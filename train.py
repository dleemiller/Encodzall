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

# Import configurations and modules
from encodzall import TrainingConfig, encodzall_s, encodzall_m, encodzall_l
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
):
    """
    Define the training loop.

    Args:
        model (nn.Module): The Encodzall model.
        tokenizer (ByteLevelTokenizer): The tokenizer.
        dataset: The training dataset.
        config (TrainingConfig): Training hyperparameters.
        device (torch.device): Device to train on.
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

    for epoch in range(config.num_epochs):
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

            # Print loss and accuracy every 10 steps
            if (i + 1) % 10 == 0:
                accuracy = (
                    correct_predictions / total_predictions
                    if total_predictions > 0
                    else 0
                )
                print(
                    f"Step {i + 1}/{math.ceil(len(dataset) / config.batch_size)}, "
                    f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}"
                )
                # Reset metrics
                correct_predictions = 0
                total_predictions = 0

        avg_epoch_loss = epoch_loss / math.ceil(len(dataset) / config.batch_size)
        print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}")

        # Save the model checkpoint
        os.makedirs(config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            config.output_dir, f"model_epoch_{epoch + 1}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")


def main():
    # Initialize Training Configuration
    training_config = TrainingConfig(
        num_epochs=1,
        batch_size=64,
        learning_rate=5e-4,
        warmup_steps=200,
        output_dir="./checkpoints",
        max_sequence_length=64,
        dataset_split="train[0:100000]",
        dataset_name="wikimedia/wikipedia",
        dataset_language="20231101.en",
        seed=42,
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

    # Initialize the tokenizer and model
    tokenizer = ByteLevelTokenizer(
        max_sequence_length=training_config.max_sequence_length
    )
    model = Encodzall(config=encodzall_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Start training
    train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=training_config,
        device=device,
    )


if __name__ == "__main__":
    main()
