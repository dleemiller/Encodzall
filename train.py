# train.py
import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Tuple, Optional
import argparse
from torch.utils.tensorboard import SummaryWriter
from simple_pid import PID

# Import configurations and modules
from encodzall.config.training_config import TrainingConfig
from encodzall.config.noise_config import NoiseConfig
from encodzall import encodzall_s, encodzall_m, encodzall_l
from encodzall import Encodzall
from encodzall import ByteLevelTokenizer, PAD_BYTE


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


def prepare_batch(
    batch: dict,
    tokenizer: ByteLevelTokenizer,
    config: TrainingConfig,
    device: torch.device,
):
    """
    Tokenize and pad the batch data.

    Args:
        batch (dict): Batch data from the dataset.
        tokenizer (ByteLevelTokenizer): Tokenizer instance.
        config (TrainingConfig): Training configuration.
        device (torch.device): Device to place tensors.

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
        tokens, mask, boundaries, targets = tokenizer.tokenize(text)
        token_ids.append(tokens)
        attention_masks.append(mask)
        word_boundaries.extend(boundaries)
        target_ids.append(targets)
        sequence_ids.extend([j] * sum([len(x) for x in boundaries]))

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


def train(
    model: nn.Module,
    tokenizer: ByteLevelTokenizer,
    dataset,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
):
    """
    Define the training loop.

    Args:
        model (nn.Module): The Encodzall model.
        tokenizer (ByteLevelTokenizer): The tokenizer.
        dataset: The training dataset.
        config (TrainingConfig): Training hyperparameters.
        device (torch.device): Device to train on.
        checkpoint_path (str, optional): Path to checkpoint to resume from.
        writer (SummaryWriter, optional): TensorBoard writer for logging.
    """
    model.train()
    model.to(device)

    # Define optimizer, scheduler, and scaler for mixed precision
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    total_steps = math.ceil(len(dataset) / config.batch_size) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    # Define loss function with token weights
    weight = load_token_weights("token_weights.json", device)
    loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=PAD_BYTE)

    # Initialize PID controller
    pid = PID(
        Kp=config.pid_Kp,
        Ki=config.pid_Ki,
        Kd=config.pid_Kd,
        setpoint=config.target_loss,
    )
    pid.output_limits = (
        0.0,
        config.prob_max,
    )  # Ensure prob stays within [prob_min, prob_max]

    # Initialize prob
    prob = config.prob_initial
    tokenizer.noise_config.set_prob(prob)  # Set initial prob in NoiseConfig

    start_epoch = 0  # Initialize starting epoch

    # If a checkpoint is provided, load it
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, loaded_config, pid, prob = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler
        )
        if loaded_config:
            config = loaded_config
        tokenizer.noise_config.set_prob(prob)
        print(f"Resuming training from epoch {start_epoch + 1} with prob={prob:.4f}")
    else:
        print("Starting training from scratch")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # Shuffle dataset at the start of each epoch
        shuffled_dataset = dataset.shuffle(seed=config.seed + epoch)

        for i, batch in enumerate(
            tqdm(shuffled_dataset.batch(config.batch_size), desc="Training")
        ):
            # Prepare batch
            (
                tokens_tensor,
                attention_mask_tensor,
                word_boundaries,
                sequence_ids,
                target_ids,
                target_key_padding_mask,
            ) = prepare_batch(batch, tokenizer, config, device)

            # Forward pass
            optimizer.zero_grad()
            with autocast():  # Mixed precision training
                output = model(
                    x=tokens_tensor,
                    sequence_ids=torch.tensor(
                        sequence_ids, dtype=torch.long, device=device
                    ),
                    target_ids=target_ids,
                    target_key_padding_mask=target_key_padding_mask,
                    attention_mask=attention_mask_tensor,
                    word_boundaries=word_boundaries,
                )

                loss = loss_fn(
                    output.view(-1, output.size(-1)),
                    target_ids[:, 1:].contiguous().view(-1),
                )

            # Calculate accuracy
            predictions = torch.argmax(output, dim=-1)
            mask = target_ids[:, 1:] != PAD_BYTE
            correct_predictions += (
                (predictions[mask] == target_ids[:, 1:][mask]).sum().item()
            )
            total_predictions += mask.sum().item()

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Adjust prob using PID controller based on current loss
            pid_output = pid(loss.item())
            prob = min(max(prob + pid_output, config.prob_min), config.prob_max)
            tokenizer.noise_config.set_prob(prob)  # Update tokenizer's prob

            # Retrieve current learning rate from optimizer
            current_lr = optimizer.param_groups[0]["lr"]

            # Log metrics to TensorBoard
            if writer:
                global_step = epoch * math.ceil(len(dataset) / config.batch_size) + i
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar(
                    "Accuracy/train",
                    (
                        correct_predictions / total_predictions
                        if total_predictions > 0
                        else 0
                    ),
                    global_step,
                )
                writer.add_scalar("Prob/train", prob, global_step)
                writer.add_scalar(
                    "LearningRate/train", current_lr, global_step
                )  # Logging Learning Rate

            # Print loss, accuracy, and prob every 10 steps
            if (i + 1) % 10 == 0:
                accuracy = (
                    correct_predictions / total_predictions
                    if total_predictions > 0
                    else 0
                )
                print(
                    f"Step {i + 1}/{math.ceil(len(dataset) / config.batch_size)}, "
                    f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}, "
                    f"Prob: {prob:.4f}"
                )
                # Reset metrics
                correct_predictions = 0
                total_predictions = 0

        avg_epoch_loss = epoch_loss / math.ceil(len(dataset) / config.batch_size)
        print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}, Prob: {prob:.4f}")
        if writer:
            writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch + 1)
            writer.add_scalar("Prob/epoch", prob, epoch + 1)

        # Save the model checkpoint at the end of the epoch
        os.makedirs(config.output_dir, exist_ok=True)
        checkpoint_path_epoch = os.path.join(
            config.output_dir, f"model_epoch_{epoch + 1}.pth"
        )
        save_checkpoint(
            checkpoint_path_epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            pid,
            epoch + 1,
            config,
            prob,
        )
        print(f"Saved model checkpoint to {checkpoint_path_epoch}")

    print("Training complete.")


def main():
    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(
        description="Train Encodzall model with PID-controlled noise"
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
        num_epochs=10,  # Adjust as needed
        batch_size=64,
        learning_rate=5e-4,
        warmup_steps=200,
        output_dir="./checkpoints",
        max_sequence_length=64,
        dataset_split="train[0:100000]",
        dataset_name="wikimedia/wikipedia",
        dataset_language="20231101.en",
        seed=42,
        target_loss=1.0,  # Desired loss
        pid_Kp=1.0,  # Proportional gain
        pid_Ki=0.1,  # Integral gain
        pid_Kd=0.05,  # Derivative gain
        prob_initial=0.0,  # Initial total noise + mask probability
        prob_min=0.0,  # Minimum total probability
        prob_max=1.0,  # Maximum total probability
        mask_ratio=0.2,  # Fixed ratio for masking
        noise_ratio=0.8,  # Fixed ratio for noise
    )

    # Set seed for reproducibility
    set_seed(training_config.seed)

    # Load the dataset
    dataset = load_dataset(
        training_config.dataset_name,
        training_config.dataset_language,
        split=training_config.dataset_split,
    )
    dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)

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
    model = Encodzall(config=encodzall_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize TensorBoard writer for logging
    writer = SummaryWriter(log_dir=args.log_dir)

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
