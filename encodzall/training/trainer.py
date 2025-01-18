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

from encodzall import ByteLevelTokenizer, PAD_BYTE
from encodzall.losses import MultipleNegativesRankingLoss, FocalLoss
from encodzall.config.training_config import TrainingConfig

# Import both stage-1 and stage-2 batch prep:
from .data import prepare_batch, prepare_batch_stage2

from .utils import load_token_weights, save_failure_data
from .checkpoint import save_checkpoint, load_checkpoint


def train_step_stage1(
    model: nn.Module,
    tokenizer: ByteLevelTokenizer,
    batch_inputs: tuple,
    config: TrainingConfig,
    device: torch.device,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: Any,
    recon_loss_seq: nn.Module,
    recon_loss_word: nn.Module,
    pid: PID,
    prob: float,
    seq_weight: float = 0.75,
    word_weight: float = 0.25,
) -> Tuple[float, float, float, float, float, float]:
    """
    Stage 1: Word reconstruction + sequence (character-level) reconstruction.
    No contrastive loss.
    """
    (
        tokens_tensor,
        attention_mask_tensor,
        word_boundaries,
        sequence_ids,
        seq_target_ids,
        seq_key_padding_mask,
        word_target_ids,
        word_key_padding_mask,
    ) = batch_inputs

    optimizer.zero_grad()
    with autocast("cuda"):
        sequence_ids_tensor = torch.tensor(
            sequence_ids, dtype=torch.long, device=device
        )

        # Forward pass
        outputs = model(
            x=tokens_tensor,
            sequence_ids=sequence_ids_tensor,
            seq_target_ids=seq_target_ids,
            seq_key_padding_mask=seq_key_padding_mask,
            word_target_ids=word_target_ids,
            word_key_padding_mask=word_key_padding_mask,
            attention_mask=attention_mask_tensor,
            word_boundaries=word_boundaries,
        )
        _, seq_logits, word_logits = outputs

        # Reconstruction losses
        seq_reconstruction_loss = recon_loss_seq(
            seq_logits.view(-1, seq_logits.size(-1)),
            seq_target_ids[:, 1:].long().contiguous().view(-1),
        )
        word_reconstruction_loss = recon_loss_word(
            word_logits.view(-1, word_logits.size(-1)),
            word_target_ids[:, 1:].long().contiguous().view(-1),
        )

    # Accuracy (optional, for the sequence reconstruction)
    predictions = torch.argmax(seq_logits, dim=-1)
    mask = seq_target_ids[:, 1:] != PAD_BYTE
    correct_predictions = (
        (predictions[mask] == seq_target_ids[:, 1:][mask]).sum().item()
    )
    total_predictions = mask.sum().item()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # Weighted reconstruction
    reconstruction_loss = (
        seq_weight * seq_reconstruction_loss + word_weight * word_reconstruction_loss
    ) / (seq_weight + word_weight)

    total_loss = reconstruction_loss  # No contrastive in stage 1

    # Backprop
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # PID update for noise probability
    pid_output = pid(total_loss.item())
    prob = min(max(prob + pid_output, config.prob_min), config.prob_max)
    tokenizer.noise_config.set_prob(prob)

    return (
        seq_reconstruction_loss.item(),
        word_reconstruction_loss.item(),
        0.0,  # contrastive = 0 in stage 1
        total_loss.item(),
        accuracy,
        prob,
    )


def train_step_stage2(
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
    contrastive_loss_fn: nn.Module,
    pid: PID,
    prob: float,
    seq_weight: float = 1.0,
) -> Tuple[float, float, float, float, float]:
    """
    Stage 2: Sequence reconstruction + contrastive loss.

    We do TWO forward passes:
      1) Anchor pass: noisy input, optional sequence reconstruction.
      2) Positive pass: clean input, return embeddings only (no recon).

    Then combine reconstruction + contrastive. One backward pass.
    """

    optimizer.zero_grad()
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
    # ------------------------------------------------------------------
    # 1) Anchor pass (with noise) => anchor embeddings + seq recon
    # ------------------------------------------------------------------
    with autocast("cuda"):
        sequence_ids_tensor = torch.tensor(
            sequence_ids, dtype=torch.long, device=device
        )
        anchor_out = model(
            x=tokens_tensor,
            sequence_ids=sequence_ids_tensor,
            seq_target_ids=seq_target_ids,
            seq_key_padding_mask=seq_key_padding_mask,
            # word_target_ids=word_target_ids,
            # word_key_padding_mask=word_key_padding_mask,
            attention_mask=attention_mask_tensor,
            word_boundaries=word_boundaries,
            return_embeddings_only=False,
        )
        anchor_embeddings, seq_logits, _ = anchor_out  # we ignore word_logits

        # Sequence reconstruction loss (if seq_target_ids is provided)
        seq_loss = seq_recon_loss_fn(
            seq_logits.view(-1, seq_logits.size(-1)),
            seq_target_ids[:, 1:].long().view(-1),
        )

    # Optional accuracy
    predictions = torch.argmax(seq_logits, dim=-1)
    mask = seq_target_ids[:, 1:] != PAD_BYTE
    correct_predictions = (
        (predictions[mask] == seq_target_ids[:, 1:][mask]).sum().item()
    )
    total_predictions = mask.sum().item()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # ------------------------------------------------------------------
    # 2) Positive pass (clean) => embeddings only, no reconstruction
    # ------------------------------------------------------------------
    with autocast("cuda"):
        with model.set_dropout(0.1):
            pos_sequence_ids_tensor = torch.tensor(
                pos_sequence_ids, dtype=torch.long, device=device
            )
            pos_embeddings = model(
                x=pos_tokens_tensor,
                sequence_ids=pos_sequence_ids_tensor,
                attention_mask=pos_attention_mask_tensor,
                word_boundaries=pos_word_boundaries,
                return_embeddings_only=True,
            )

    # Contrastive loss
    contrast_loss = contrastive_loss_fn(anchor_embeddings, pos_embeddings)

    # Combine
    total_loss = seq_weight * seq_loss + contrast_loss

    # Backprop
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # PID update for noise probability
    pid_output = pid(total_loss.item())
    prob = min(max(prob + pid_output, config.prob_min), config.prob_max)
    tokenizer.noise_config.set_prob(prob)

    return (
        seq_loss.item(),
        contrast_loss.item(),
        total_loss.item(),
        accuracy,
        prob,
    )


def train(
    model: nn.Module,
    tokenizer: ByteLevelTokenizer,
    dataset,
    config: TrainingConfig,
    device: torch.device,
    stage: int = 2,  # 1 or 2
    checkpoint_path: Optional[str] = None,
    weights_only: bool = False,
    writer: Optional[SummaryWriter] = None,
):
    """
    A single training function that checks if we're in stage 1 or stage 2.
    """
    model.train()
    model.to(device)

    # Optimizer, scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-6
    )
    scaler = GradScaler("cuda")
    total_steps = math.ceil(len(dataset) / config.batch_size) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    # Losses
    weight = load_token_weights("token_weights.json", device)
    recon_loss_seq = nn.CrossEntropyLoss(weight=weight, ignore_index=PAD_BYTE)
    recon_loss_word = None
    if stage == 1:
        recon_loss_word = FocalLoss(gamma=1.0)

    contrastive_loss_fn = None
    if stage == 2:
        contrastive_loss_fn = MultipleNegativesRankingLoss(scale=20.0)

    # PID
    pid = PID(config.pid_Kp, config.pid_Ki, config.pid_Kd, setpoint=config.target_loss)
    pid.output_limits = (config.prob_min, config.prob_max)
    prob = config.prob_initial
    tokenizer.noise_config.set_prob(prob)

    # Load checkpoint if exists
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
                load_weights_only=weights_only,
            )
        else:
            start_step, config, pid, prob, dataset_offset = load_checkpoint(
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
        print(f"Epoch {epoch + 1}/{config.num_epochs} [Stage {stage}]")

        for batch_texts in tqdm(dataset.batch(config.batch_size), desc="Training"):
            try:
                if stage == 1:
                    # -----------------------------
                    # STAGE 1 => Single pass
                    # -----------------------------
                    batch_inputs = prepare_batch(
                        batch_texts, tokenizer, config, device, noise_prob=None
                    )

                    (
                        seq_loss,
                        word_loss,
                        contrast_loss_val,
                        total_loss,
                        accuracy,
                        prob,
                    ) = train_step_stage1(
                        model=model,
                        tokenizer=tokenizer,
                        batch_inputs=batch_inputs,
                        config=config,
                        device=device,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        recon_loss_seq=recon_loss_seq,
                        recon_loss_word=recon_loss_word,
                        pid=pid,
                        prob=prob,
                    )

                    # Logging
                    if writer:
                        writer.add_scalar("Stage1/Seq_Loss", seq_loss, step)
                        writer.add_scalar("Stage1/Word_Loss", word_loss, step)
                        writer.add_scalar("Stage1/Total_Loss", total_loss, step)
                        writer.add_scalar("Stage1/Accuracy", accuracy, step)
                        writer.add_scalar("Stage1/NoiseProb", prob, step)
                        writer.add_scalar(
                            "Stage1/LearningRate", optimizer.param_groups[0]["lr"], step
                        )

                else:
                    # -----------------------------
                    # STAGE 2 => Anchor+Positive
                    # -----------------------------
                    anchor_inputs, pos_inputs = prepare_batch_stage2(
                        batch_texts, tokenizer, config, device
                    )
                    (
                        seq_loss,
                        contrast_loss_val,
                        total_loss,
                        accuracy,
                        prob,
                    ) = train_step_stage2(
                        model=model,
                        tokenizer=tokenizer,
                        anchor_inputs=anchor_inputs,
                        pos_inputs=pos_inputs,
                        config=config,
                        device=device,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        seq_recon_loss_fn=recon_loss_seq,
                        contrastive_loss_fn=contrastive_loss_fn,
                        pid=pid,
                        prob=prob,
                    )

                    # Logging
                    if writer:
                        writer.add_scalar("Stage2/Seq_Loss", seq_loss, step)
                        writer.add_scalar(
                            "Stage2/Contrastive_Loss", contrast_loss_val, step
                        )
                        writer.add_scalar("Stage2/Total_Loss", total_loss, step)
                        writer.add_scalar("Stage2/Accuracy", accuracy, step)
                        writer.add_scalar("Stage2/NoiseProb", prob, step)
                        writer.add_scalar(
                            "Stage2/LearningRate", optimizer.param_groups[0]["lr"], step
                        )

                # ---- Save checkpoint every N steps ----
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
                    # In stage 1, we used 'batch_inputs'; in stage 2, we used anchor/pos
                    if stage == 1:
                        inputs = {
                            "batch_text": batch_texts,
                            "batch_inputs": batch_inputs,
                        }
                    else:
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

        print(f"Completed epoch {epoch + 1} (Stage {stage})")
        dataset_offset = 0

    print(f"Training stage {stage} complete.")
