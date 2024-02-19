import torch


def gather_word_starts(
    data: torch.Tensor,
    word_starts: torch.Tensor,
    pad_id: int = 0,
    max_words: int = None,
):
    batch_size, max_seq_length, hidden_dim = data.size()

    # Calculate the maximum number of True positions across all batches if max_words is not set
    if max_words is None:
        max_words = word_starts.sum(dim=1).max().item()

    # Initialize tensors to hold the gathered positions and attention mask
    gathered = torch.zeros(
        batch_size, max_words, hidden_dim, dtype=data.dtype, device=data.device
    )
    mask = torch.zeros(batch_size, max_words, dtype=torch.bool, device=data.device)

    for i in range(batch_size):
        true_positions = data[
            i, word_starts[i]
        ]  # Gather true positions for the current batch
        true_count = true_positions.size(0)

        # Copy the positions into the 'gathered' tensor and update the mask
        gathered[i, :true_count] = true_positions[:true_count]
        mask[i, :true_count] = True

    return gathered, mask
