import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from typing import Any, Optional, Tuple
from tqdm import tqdm
from simple_pid import PID

# Tokenizer + losses
from encodzall import ByteLevelTokenizer, PAD_BYTE
from encodzall.losses import MultipleNegativesRankingLoss, FocalLoss
from encodzall.config.training_config import TrainingConfig

# Data prep
from .data import prepare_batch_stage2
from .utils import load_token_weights, save_failure_data
from .checkpoint import save_checkpoint, load_checkpoint


def train_step_combined(
    model: nn.Module,
    tokenizer: ByteLevelTokenizer,
    anchor_inputs: tuple,
    pos_inputs: tuple,
    config: TrainingConfig,
    device: torch.device,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: Any,
    seq_recon_loss_fn: nn.Module,
    word_recon_loss_fn: nn.Module,
    contrastive_loss_fn: nn.Module,
    pid: PID,
    prob: float,
    seq_weight: float = 0.5,
    word_weight: float = 0.25,
    contrast_weight: float = 0.25,
) -> Tuple[float, float, float, float, float]:
    """
    Single-step training that combines:
      - Sequence reconstruction (anchor)
      - Word reconstruction (anchor)
      - Contrastive embedding (anchor vs positive)

    Returns:
      seq_loss, word_loss, contrast_loss, total_loss, accuracy, new_prob
    """

    optimizer.zero_grad()

    # Unpack anchor inputs
    (
        tokens_tensor,
        attention_mask_tensor,
        word_boundaries,
        sequence_ids,
        seq_target_ids,
        seq_key_padding_mask,
        word_target_ids,
        word_key_padding_mask,
    ) = anchor_inputs

    # Unpack positive inputs
    (
        pos_tokens_tensor,
        pos_attention_mask_tensor,
        pos_word_boundaries,
        pos_sequence_ids,
        _,
        _,
        _,
        _,
    ) = pos_inputs

    with autocast("cuda"):
        # -------------------------------------------------
        # Anchor pass => (noisy) => seq + word reconstruction + anchor embeddings
        # -------------------------------------------------
        sequence_ids_tensor = torch.tensor(
            sequence_ids, dtype=torch.long, device=device
        )

        anchor_outputs = model(
            x=tokens_tensor,
            sequence_ids=sequence_ids_tensor,
            seq_target_ids=seq_target_ids,
            seq_key_padding_mask=seq_key_padding_mask,
            word_target_ids=word_target_ids,
            word_key_padding_mask=word_key_padding_mask,
            attention_mask=attention_mask_tensor,
            word_boundaries=word_boundaries,
            return_embeddings_only=False,
        )
        anchor_embeddings, seq_logits, word_logits = anchor_outputs

        # Sequence reconstruction loss
        seq_loss = seq_recon_loss_fn(
            seq_logits.view(-1, seq_logits.size(-1)),
            seq_target_ids[:, 1:].long().contiguous().view(-1),
        )

        # Word reconstruction loss
        word_loss = word_recon_loss_fn(
            word_logits.view(-1, word_logits.size(-1)),
            word_target_ids[:, 1:].long().contiguous().view(-1),
        )

        # Compute accuracy for seq reconstruction (optional)
        predictions = torch.argmax(seq_logits, dim=-1)
        mask = seq_target_ids[:, 1:] != PAD_BYTE
        correct_predictions = (
            (predictions[mask] == seq_target_ids[:, 1:][mask]).sum().item()
        )
        total_predictions = mask.sum().item()
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )

        # -------------------------------------------------
        # Positive pass => (clean) => embeddings only
        # -------------------------------------------------
        pos_sequence_ids_tensor = torch.tensor(
            pos_sequence_ids, dtype=torch.long, device=device
        )
        with model.set_dropout(0.1):
            with torch.no_grad():
                pos_embeddings = model(
                    x=pos_tokens_tensor,
                    sequence_ids=pos_sequence_ids_tensor,
                    attention_mask=pos_attention_mask_tensor,
                    word_boundaries=pos_word_boundaries,
                    return_embeddings_only=True,
                )

        # Contrastive loss
        contrast_loss = contrastive_loss_fn(anchor_embeddings, pos_embeddings)

        # Combine all losses
        total_loss = (
            seq_weight * seq_loss
            + word_weight * word_loss
            + contrast_weight * contrast_loss
        )

    # Single backward pass
    scaler.scale(total_loss).backward()

    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # PID update for noise probability
    pid_output = pid(total_loss.item())
    new_prob = min(max(prob + pid_output, config.prob_min), config.prob_max)
    tokenizer.noise_config.set_prob(new_prob)

    return (
        seq_loss.item(),
        word_loss.item(),
        contrast_loss.item(),
        total_loss.item(),
        accuracy,
        new_prob,
    )


def train(
    model: nn.Module,
    tokenizer: ByteLevelTokenizer,
    dataset,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    weights_only: bool = False,
    writer: Optional[SummaryWriter] = None,
):
    """
    Single combined stage of training that uses:
      - sequence reconstruction
      - word reconstruction
      - contrastive embeddings

    """

    model.train()
    model.to(device)

    # Setup optimizer, scaler, scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-6
    )
    scaler = GradScaler("cuda")

    total_steps = math.ceil(len(dataset) / config.batch_size) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    # Losses
    weight = load_token_weights("token_weights.json", device)
    seq_recon_loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=PAD_BYTE)
    word_recon_loss_fn = FocalLoss(gamma=1.0)
    contrastive_loss_fn = MultipleNegativesRankingLoss(scale=20.0)

    # PID initialization
    pid = PID(config.pid_Kp, config.pid_Ki, config.pid_Kd, setpoint=config.target_loss)
    pid.output_limits = (config.prob_min, config.prob_max)
    prob = config.prob_initial
    tokenizer.noise_config.set_prob(prob)

    # Optionally load from checkpoint
    start_step = 0
    dataset_offset = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        if weights_only:
            load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                load_weights_only=True,
            )
        else:
            (start_step, config, pid, prob, dataset_offset) = load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                load_weights_only=False,
            )
        tokenizer.noise_config.set_prob(prob)
        if dataset_offset > 0:
            dataset = dataset.skip(dataset_offset)

    step = start_step
    failure_dir = os.path.join(config.output_dir, "failures")

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs} (Combined)")

        for batch_texts in tqdm(dataset.batch(config.batch_size), desc="Training"):
            try:
                # Prepare anchor + positive
                anchor_inputs, pos_inputs = prepare_batch_stage2(
                    batch_texts, tokenizer, config, device
                )

                # Single step with combined losses
                (
                    seq_loss_val,
                    word_loss_val,
                    contrast_loss_val,
                    total_loss_val,
                    accuracy,
                    prob,
                ) = train_step_combined(
                    model=model,
                    tokenizer=tokenizer,
                    anchor_inputs=anchor_inputs,
                    pos_inputs=pos_inputs,
                    config=config,
                    device=device,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    seq_recon_loss_fn=seq_recon_loss_fn,
                    word_recon_loss_fn=word_recon_loss_fn,
                    contrastive_loss_fn=contrastive_loss_fn,
                    pid=pid,
                    prob=prob,
                    seq_weight=0.75,  # Adjust as desired
                    word_weight=0.25,  # Adjust as desired
                    contrast_weight=1.0,  # Adjust as desired
                )

                # Logging
                if writer:
                    writer.add_scalar("Combined/Seq_Loss", seq_loss_val, step)
                    writer.add_scalar("Combined/Word_Loss", word_loss_val, step)
                    writer.add_scalar("Combined/Contrast_Loss", contrast_loss_val, step)
                    writer.add_scalar("Combined/Total_Loss", total_loss_val, step)
                    writer.add_scalar("Combined/Accuracy", accuracy, step)
                    writer.add_scalar("Combined/NoiseProb", prob, step)
                    writer.add_scalar(
                        "Combined/LearningRate",
                        optimizer.param_groups[0]["lr"],
                        step,
                    )

                # Save checkpoint every N steps
                if step > 0 and step % config.checkpoint_interval == 0:
                    dataset_offset = (step * config.batch_size) % len(dataset)
                    ckpt_path_step = os.path.join(
                        config.output_dir, f"model_step_{step}.pth"
                    )
                    save_checkpoint(
                        ckpt_path_step,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        pid,
                        step,
                        config,
                        prob,
                        dataset_offset,
                    )
                    print(f"Saved checkpoint to {ckpt_path_step}")

                step += 1

            except Exception as e:
                import traceback

                print(f"Error at step {step}: {e}")
                print(traceback.format_exc())
                try:
                    # Attempt to save relevant failure info
                    inputs = {
                        "batch_text": batch_texts,
                        "anchor_inputs": anchor_inputs,
                        "pos_inputs": pos_inputs,
                    }
                    save_failure_data(failure_dir, step, batch_texts, inputs, e)
                    if writer:
                        writer.add_text("Errors/train", f"Step {step}: {str(e)}", step)
                except:
                    print("Failed to save failure data after exception.")
                    print(traceback.format_exc())
                step += 1
                continue

        print(f"Completed epoch {epoch + 1} (Combined)")
        dataset_offset = 0

    print("Training complete (combined stage).")
