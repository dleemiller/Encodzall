import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import math

from word_encoder import WordTransformer
from tokenizer import ByteLevelTokenizer


# Define the training loop
def train(
    model,
    tokenizer,
    dataset,
    device,
    num_epochs,
    batch_size,
    learning_rate,
    warmup_steps,
    output_dir,
):
    model.train()
    model.to(device)

    # Define optimizer, scheduler, and scaler for mixed precision
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    total_steps = math.ceil(len(dataset) / batch_size) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0

        for i, batch in enumerate(
            tqdm(dataset.shuffle(seed=42).batch(batch_size), desc="Training")
        ):
            # Tokenize the input and create attention masks
            texts = batch["text"]
            token_ids, attention_masks, word_boundaries, sequence_ids = [], [], [], []
            for j, text in enumerate(texts):
                tokens, mask, boundaries = tokenizer.tokenize(text)
                token_ids.append(tokens)
                attention_masks.append(mask)
                word_boundaries.extend(boundaries)
                sequence_ids.extend([j] * sum([len(x) for x in boundaries]))

            tokens_tensor = torch.cat(token_ids).to(device)
            attention_mask_tensor = torch.cat(attention_masks).to(device)

            # Generate target IDs for reconstruction
            target_ids, target_key_padding_mask = tokenizer.create_targets(texts)
            target_ids = target_ids.to(device)
            target_key_padding_mask = target_key_padding_mask.to(device)

            # Forward pass
            optimizer.zero_grad()
            with autocast():  # Mixed precision training
                output = model(
                    tokens_tensor,
                    sequence_ids=torch.tensor(sequence_ids).to(device),
                    target_ids=target_ids,
                    target_key_padding_mask=target_key_padding_mask,
                    attention_mask=attention_mask_tensor,
                    word_boundaries=word_boundaries,
                )
                loss = loss_fn(output.view(-1, output.size(-1)), target_ids.view(-1))

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            print(loss.item())

        print(f"Epoch Loss: {epoch_loss / len(dataset):.4f}")

        # Save the model checkpoint
        torch.save(model.state_dict(), f"{output_dir}/model_epoch_{epoch + 1}.pth")


# Main function
if __name__ == "__main__":
    # Configurations
    num_epochs = 3
    batch_size = 16
    learning_rate = 5e-4
    warmup_steps = 1000
    output_dir = "./checkpoints"
    max_sequence_length = 64

    # Load the dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[0:10000]")
    dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)

    # Initialize the tokenizer and model
    tokenizer = ByteLevelTokenizer(max_sequence_length=max_sequence_length)
    model = WordTransformer(
        vocab_size=256,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=512,
        pooling_type="average",
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train(
        model,
        tokenizer,
        dataset,
        device,
        num_epochs,
        batch_size,
        learning_rate,
        warmup_steps,
        output_dir,
    )
