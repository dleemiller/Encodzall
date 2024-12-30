# training_script.py
from datetime import datetime
import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm
from typing import Optional
import argparse
from torch.utils.tensorboard import SummaryWriter
from simple_pid import PID

# Import configurations and modules
from encodzall.config.training_config import TrainingConfig
from encodzall.config.noise_config import NoiseConfig
from encodzall import encodzall_xs, encodzall_s, Encodzall, ByteLevelTokenizer, PAD_BYTE
from encodzall import encodzall_xs, Encodzall, ByteLevelTokenizer, PAD_BYTE
from sentence_transformers.losses import MultipleNegativesRankingLoss


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
    token_ids, attention_masks, word_boundaries, sequence_ids, target_ids = (
        [],
        [],
        [],
        [],
        [],
    )

    for j, text in enumerate(texts):
        tokens, mask, boundaries, targets = tokenizer.tokenize(text, noise_prob=noise_prob)
        token_ids.append(tokens)
        attention_masks.append(mask)
        word_boundaries.extend(boundaries)
        target_ids.append(targets)
        sequence_ids.extend([j] * sum(len(x) for x in boundaries))

    tokens_tensor = torch.cat(token_ids).to(device)
    attention_mask_tensor = torch.cat(attention_masks).to(device)

    # Generate target masks for reconstruction
    target_ids, target_key_padding_mask = tokenizer.pad_targets(target_ids)
    target_ids = target_ids.to(device)
    target_key_padding_mask = target_key_padding_mask.to(device)

    return (
        tokens_tensor,
        attention_mask_tensor,
        word_boundaries,
        sequence_ids,
        target_ids,
        target_key_padding_mask,
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

    return checkpoint["step"], config, pid, prob, dataset_offset


def train(model, tokenizer, dataset, config, device, checkpoint_path=None, writer=None):
    model.train()
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler("cuda")
    total_steps = math.ceil(len(dataset) / config.batch_size) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    weight = load_token_weights("token_weights.json", device)
    reconstruction_loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=PAD_BYTE)

    # Initialize the contrastive loss
    contrastive_loss_fn = MultipleNegativesRankingLoss(model, scale=20.0)  # Adjust scale as needed

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
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch in tqdm(dataset.batch(config.batch_size), desc=f"Training"):
            (
                tokens_tensor,
                attention_mask_tensor,
                word_boundaries,
                sequence_ids,
                target_ids,
                target_key_padding_mask,
            ) = prepare_batch(batch, tokenizer, config, device, noise_prob=prob)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(
                    x=tokens_tensor,
                    sequence_ids=torch.tensor(
                        sequence_ids, dtype=torch.long, device=device
                    ),
                    target_ids=target_ids,
                    target_key_padding_mask=target_key_padding_mask,
                    attention_mask=attention_mask_tensor,
                    word_boundaries=word_boundaries,
                )
                # Compute reconstruction loss
                reconstruction_loss = reconstruction_loss_fn(
                    logits.view(-1, logits.size(-1)),
                    target_ids[:, 1:].contiguous().view(-1),
                )

            # ----- Contrastive Learning with Fixed Noise Probability -----
            with torch.no_grad():
                # Tokenize with fixed noise probability (e.g., 0.5) for positive examples
                (
                    tokens_tensor_pos,
                    attention_mask_tensor_pos,
                    word_boundaries_pos,
                    sequence_ids_pos,
                    target_ids_pos,
                    target_key_padding_mask_pos,
                ) = prepare_batch(batch, tokenizer, config, device, noise_prob=0.5)

                # Obtain embeddings for contrastive loss
                embeddings_pos = model(
                    x=tokens_tensor_pos,
                    sequence_ids=torch.tensor(
                        sequence_ids_pos, dtype=torch.long, device=device
                    ),
                    target_ids=target_ids_pos,
                    target_key_padding_mask=target_key_padding_mask_pos,
                    attention_mask=attention_mask_tensor_pos,
                    word_boundaries=word_boundaries_pos,
                    return_embeddings_only=True,  # Get embeddings only
                )  # Shape: (batch_size, d_model)

            # Now, obtain embeddings for the anchor (current batch)
            with autocast("cuda"):
                embeddings_anchor = model(
                    x=tokens_tensor,
                    sequence_ids=torch.tensor(
                        sequence_ids, dtype=torch.long, device=device
                    ),
                    target_ids=target_ids,
                    target_key_padding_mask=target_key_padding_mask,
                    attention_mask=attention_mask_tensor,
                    word_boundaries=word_boundaries,
                    return_embeddings_only=True,  # Get embeddings only
                )  # Shape: (batch_size, d_model)

            # Prepare sentence_features for MultipleNegativesRankingLoss
            # According to the class definition, the first element should be anchors,
            # and the second element should be positives.
            sentence_features = [
                {"sentence_embedding": embeddings_anchor},
                {"sentence_embedding": embeddings_pos},
            ]

            # Compute contrastive loss
            contrastive_loss = contrastive_loss_fn(sentence_features=sentence_features, labels=None)

            # ----- Combine Losses -----
            total_loss = reconstruction_loss + contrastive_loss  # You can weight them if needed

            # ----- Backpropagation -----
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ----- PID Controller for Noise Probability -----
            pid_output = pid(total_loss.item())
            prob = min(max(prob + pid_output, config.prob_min), config.prob_max)
            tokenizer.noise_config.set_prob(prob)

            # ----- TensorBoard Logging -----
            if writer:
                writer.add_scalar("Loss/train_reconstruction", reconstruction_loss.item(), step)
                writer.add_scalar("Loss/train_contrastive", contrastive_loss.item(), step)
                writer.add_scalar("Loss/train_total", total_loss.item(), step)
                # Optional: Add more metrics as needed
                writer.add_scalar("Prob/train", prob, step)
                writer.add_scalar(
                    "LearningRate/train", optimizer.param_groups[0]["lr"], step
                )

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
        num_epochs=2,  # Adjust as needed
        batch_size=80,
        learning_rate=5e-4,
        warmup_steps=1000,
        output_dir="./checkpoints",
        max_sequence_length=64,
        dataset_split="train",
        dataset_name="HuggingFaceTB/smollm-corpus",
        dataset_subset="fineweb-edu-dedup",
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

