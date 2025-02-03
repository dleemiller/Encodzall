from datetime import datetime
import os
import json
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm
from typing import Optional, Tuple, Any, Dict
import argparse
from torch.utils.tensorboard import SummaryWriter
from simple_pid import PID

# Import configurations and modules
from encodzall.config.training_config import TrainingConfig
from encodzall.config.noise_config import NoiseConfig
from encodzall import encodzall_xs, encodzall_s, Encodzall, ByteLevelTokenizer, PAD_BYTE
from encodzall.losses import MultipleNegativesRankingLoss, FocalLoss


def set_seed(seed: int):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_token_weights(filepath: str, device: torch.device) -> torch.Tensor:
    """Load token weights from a JSON file."""
    with open(filepath, "r") as fh:
        weights = json.load(fh)
        idx, weight = zip(*sorted(weights.items()))
        weight = torch.tensor(weight, dtype=torch.float32).to(device)
    return weight


def prepare_batch(batch, tokenizer, config, device, noise_prob: Optional[float] = None):
    """
    Tokenize and pad the batch data.

    Args:
        batch (dict): Batch data from the dataset.
        tokenizer (ByteLevelTokenizer): Tokenizer instance.
        config (TrainingConfig): Training configuration.
        device (torch.device): Device to place tensors.
        noise_prob (float, optional): Probability of applying noise. Defaults to None.

    Returns:
        Tuple containing token_ids, attention_masks, word_boundaries, sequence_ids, target_ids, target_key_padding_mask
    """
    texts = batch["text"]
    (
        token_ids,
        attention_masks,
        word_boundaries,
        sequence_ids,
        seq_target_ids,
        word_target_ids,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for j, text in enumerate(texts):
        tokens, mask, boundaries, targets = tokenizer.tokenize(text)
        token_ids.append(tokens)
        attention_masks.append(mask)
        word_boundaries.extend(boundaries)
        seq_target_ids.append([item for sublist in targets for item in sublist])
        word_target_ids.extend(targets)
        sequence_ids.extend([j] * sum(len(x) for x in boundaries))

    tokens_tensor = torch.cat(token_ids).to(device)
    attention_mask_tensor = torch.cat(attention_masks).to(device)

    # Generate target masks for reconstruction
    seq_target_ids, seq_key_padding_mask = tokenizer.pad_targets(seq_target_ids)
    seq_target_ids = seq_target_ids.to(device)
    seq_key_padding_mask = seq_key_padding_mask.to(device)

    word_target_ids, word_key_padding_mask = tokenizer.pad_targets(word_target_ids)
    word_target_ids = word_target_ids.to(device)
    word_key_padding_mask = word_key_padding_mask.to(device)

    return (
        tokens_tensor,
        attention_mask_tensor,
        word_boundaries,
        sequence_ids,
        seq_target_ids,
        seq_key_padding_mask,
        word_target_ids,
        word_key_padding_mask,
    )


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    scaler,
    pid,
    step,
    config,
    prob,
    dataset_offset,
):
    """Save training checkpoint."""
    # Ensure the directory exists
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "pid_tunings": pid.tunings,
        "pid_setpoint": pid.setpoint,
        "pid_auto_mode": pid.auto_mode,
        "prob": prob,
        "dataset_offset": dataset_offset,
        "config": config.__dict__,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, config=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    pid = PID(
        Kp=checkpoint["pid_tunings"][0],
        Ki=checkpoint["pid_tunings"][1],
        Kd=checkpoint["pid_tunings"][2],
        setpoint=checkpoint["pid_setpoint"],
    )
    pid.auto_mode = checkpoint.get("pid_auto_mode", True)

    prob = checkpoint.get("prob", config.prob_initial if config else 0.0)
    dataset_offset = checkpoint.get("dataset_offset", 0)  # Load dataset offset

    if not config:
        saved_config = checkpoint.get("config")
        if saved_config:
            config = TrainingConfig(**saved_config)
        else:
            raise ValueError("Config missing from checkpoint and not provided.")

    # load from checkpoint step + 1
    return checkpoint["step"] + 1, config, pid, prob, dataset_offset + 1


def save_failure_data(
    failure_dir: str,
    step: int,
    batch: Dict[str, Any],
    inputs: Dict[str, Any],
    exception: Exception,
):
    """Save inputs and batch text to a file for debugging."""
    os.makedirs(failure_dir, exist_ok=True)
    failure_data = {
        "step": step,
        "batch_text": batch.get("text", []),
        "inputs": {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        },
        "exception": str(exception),
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    failure_file = os.path.join(failure_dir, f"failure_step_{step}_{timestamp}.pkl")
    with open(failure_file, "w") as f:
        pickle.dump(failure_data, f, indent=4)
    print(f"Saved failure data to {failure_file}")


def train_step(
    model: nn.Module,
    tokenizer: ByteLevelTokenizer,
    batch_inputs: tuple[str, Any],
    config: TrainingConfig,
    device: torch.device,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: Any,
    recon_loss_seq: nn.Module,
    recon_loss_word: nn.Module,
    contrastive_loss_fn: Optional[nn.Module],
    pid: PID,
    prob: float,
    seq_weight: float = 0.75,
    word_weight: float = 0.25,
) -> Tuple[float, float, float, float, float]:
    """
    Perform a single training step.

    Args:
        model (nn.Module): The model to train.
        tokenizer (ByteLevelTokenizer): The tokenizer.
        batch (dict): The batch data.
        config (TrainingConfig): Training configuration.
        device (torch.device): Device to run on.
        optimizer (optim.Optimizer): Optimizer.
        scaler (GradScaler): Gradient scaler for AMP.
        scheduler (Any): Learning rate scheduler.
        recon_loss_seq (nn.Module): Loss function for reconstruction.
        contrastive_loss_fn (Optional[nn.Module]): Loss function for contrastive learning.
        pid (PID): PID controller for noise probability.
        prob (float): Current noise probability.

    Returns:
        Tuple containing losses, accuracy, and updated noise probability.
    """
    # Prepare the batch
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
        # Assuming model returns (something, seq_logits, word_logits)
        _, seq_logits, word_logits = outputs

        # Compute reconstruction loss
        seq_reconstruction_loss = recon_loss_seq(
            seq_logits.view(-1, seq_logits.size(-1)),
            seq_target_ids[:, 1:].long().contiguous().view(-1),
        )
        word_reconstruction_loss = recon_loss_word(
            word_logits.view(-1, word_logits.size(-1)),
            word_target_ids[:, 1:].long().contiguous().view(-1),
        )

    # Compute accuracy
    predictions = torch.argmax(seq_logits, dim=-1)
    mask = seq_target_ids[:, 1:] != PAD_BYTE
    correct_predictions = (
        (predictions[mask] == seq_target_ids[:, 1:][mask]).sum().item()
    )
    total_predictions = mask.sum().item()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # Contrastive loss (if applicable)
    if contrastive_loss_fn is not None:
        with torch.no_grad():
            with model.set_dropout(0.1):
                embeddings_pos = model(
                    x=tokens_tensor,
                    sequence_ids=sequence_ids_tensor,
                    seq_target_ids=seq_target_ids,
                    seq_key_padding_mask=seq_key_padding_mask,
                    word_target_ids=word_target_ids,
                    word_key_padding_mask=word_key_padding_mask,
                    attention_mask=attention_mask_tensor,
                    return_embeddings_only=True,
                )
        contrastive_loss = contrastive_loss_fn(embeddings_anchor, embeddings_pos)
    else:
        contrastive_loss = 0.0

    # Combine Losses
    reconstruction_loss = (
        seq_weight * seq_reconstruction_loss + word_weight * word_reconstruction_loss
    ) / (seq_weight + word_weight)
    total_loss = reconstruction_loss + contrastive_loss

    # Backpropagation
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # PID Controller for Noise Probability
    pid_output = pid(total_loss.item())
    prob = min(max(prob + pid_output, config.prob_min), config.prob_max)
    tokenizer.noise_config.set_prob(prob)

    return (
        seq_reconstruction_loss.item(),
        word_reconstruction_loss.item(),
        contrastive_loss,
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
    checkpoint_path: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
):
    model.train()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-6
    )
    scaler = GradScaler("cuda")
    total_steps = math.ceil(len(dataset) / config.batch_size) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    weight = load_token_weights("token_weights.json", device)
    recon_loss_seq = nn.CrossEntropyLoss(weight=weight, ignore_index=PAD_BYTE)
    recon_loss_word = FocalLoss(gamma=1.0)

    # Initialize the contrastive loss
    # contrastive_loss_fn = MultipleNegativesRankingLoss(scale=20.0)
    contrastive_loss_fn = None

    pid = PID(config.pid_Kp, config.pid_Ki, config.pid_Kd, setpoint=config.target_loss)
    pid.output_limits = (config.prob_min, config.prob_max)
    prob = config.prob_initial
    tokenizer.noise_config.set_prob(prob)

    start_step = 0
    dataset_offset = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        start_step, config, pid, prob, dataset_offset = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, config
        )
        tokenizer.noise_config.set_prob(prob)

        # Apply offset for resumption
        if dataset_offset > 0:
            dataset = dataset.skip(dataset_offset)

    step = start_step
    failure_dir = os.path.join(config.output_dir, "failures")
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch in tqdm(dataset.batch(config.batch_size), desc=f"Training"):
            try:
                batch_inputs = prepare_batch(
                    batch, tokenizer, config, device, noise_prob=None
                )
                # Perform a training step
                (
                    seq_loss,
                    word_loss,
                    contrastive_loss,
                    total_loss,
                    accuracy,
                    prob,
                ) = train_step(
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
                    contrastive_loss_fn=contrastive_loss_fn,
                    pid=pid,
                    prob=prob,
                )

                # ----- TensorBoard Logging -----
                if writer:
                    writer.add_scalar("Seq Loss/train_reconstruction", seq_loss, step)
                    writer.add_scalar("Word Loss/train_reconstruction", word_loss, step)
                    if contrastive_loss_fn is not None:
                        writer.add_scalar(
                            "Loss/train_contrastive", contrastive_loss.item(), step
                        )
                    writer.add_scalar("Loss/train_total", total_loss, step)
                    writer.add_scalar("Prob/train", prob, step)
                    writer.add_scalar(
                        "LearningRate/train", optimizer.param_groups[0]["lr"], step
                    )
                    writer.add_scalar("Accuracy/train", accuracy, step)

                # ----- Save Checkpoint every N Steps -----
                if step > 0 and step % config.checkpoint_interval == 0:
                    dataset_offset = (step * config.batch_size) % len(dataset)
                    checkpoint_path_step = os.path.join(
                        config.output_dir, f"model_step_{step}.pth"
                    )
                    save_checkpoint(
                        checkpoint_path_step,
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
                    print(f"Saved checkpoint to {checkpoint_path_step}")

                step += 1

            except Exception as e:
                import traceback

                print(f"Error at step {step}: {e}")
                print(traceback.format_exc())
                try:
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

                    # Capture all model inputs and batch text
                    inputs = {
                        "batch_text": batch,
                        "tokens_tensor": tokens_tensor.cpu(),
                        "attention_mask_tensor": attention_mask_tensor.cpu(),
                        "word_boundaries": word_boundaries,
                        "sequence_ids": sequence_ids,
                        "seq_target_ids": seq_target_ids.cpu(),
                        "seq_key_padding_mask": seq_key_padding_mask.cpu(),
                        "word_target_ids": word_target_ids.cpu(),
                        "word_key_padding_mask": word_key_padding_mask.cpu(),
                    }
                    save_failure_data(failure_dir, step, batch, inputs, e)
                    # Optionally, log the exception to TensorBoard or other logging systems
                    if writer:
                        writer.add_text("Errors/train", f"Step {step}: {str(e)}", step)
                except:
                    print(traceback.format_exc())
                    print("Failed to save failure data")
                # Continue training
                step += 1
                continue

        print(f"Completed epoch {epoch + 1}")
        dataset_offset = 0
    print("Training complete.")


def main():
    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(
        description="Train Encodzall model with PID-controlled noise and contrastive loss"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save TensorBoard logs",
    )
    args = parser.parse_args()

    # Initialize Training Configuration
    training_config = TrainingConfig(
        num_epochs=1,  # Adjust as needed
        batch_size=80,
        learning_rate=2e-4,
        warmup_steps=2500,
        output_dir="./checkpoints",
        max_sequence_length=64,
        dataset_split="train",
        dataset_name="skymizer/fineweb-edu-dedup-45B",
        dataset_subset="default",
        # dataset_subset="20231101.en",
        # dataset_name="wikimedia/wikipedia",
        # dataset_split="train[0:1000]",
        seed=42,
        target_loss=0.2,  # Desired loss
        pid_Kp=1.0,  # Proportional gain
        pid_Ki=0.1,  # Integral gain
        pid_Kd=0.05,  # Derivative gain
        prob_initial=0.0,  # Initial total noise + mask probability
        prob_min=0.0,  # Minimum total probability
        prob_max=1.0,  # Maximum total probability
        mask_ratio=0.2,  # Fixed ratio for masking
        noise_ratio=0.8,  # Fixed ratio for noise
        checkpoint_interval=2000,
    )

    # Set seed for reproducibility
    set_seed(training_config.seed)

    # Load the dataset
    dataset = load_dataset(
        training_config.dataset_name,
        training_config.dataset_subset,
        split=training_config.dataset_split,
        cache_dir="/home/datasets/",
        download_config=DownloadConfig(resume_download=True),
    )
    # dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)

    # Initialize the tokenizer with NoiseConfig
    noise_config = NoiseConfig(
        prob=training_config.prob_initial,
        mask_ratio=training_config.mask_ratio,
        noise_ratio=training_config.noise_ratio,
    )
    tokenizer = ByteLevelTokenizer(
        max_sequence_length=training_config.max_sequence_length,
        noise_config=noise_config,
    )

    model = Encodzall(config=encodzall_xs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=unique_log_dir)

    # Start training with optional checkpoint
    train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=training_config,
        device=device,
        checkpoint_path=args.checkpoint,
        writer=writer,
    )

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
