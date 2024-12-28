import torch
from torch import nn
import more_itertools
from typing import List, Tuple

from tokenizer import ByteLevelTokenizer


class WordPooling(nn.Module):
    def __init__(self, pooling_type: str = "average"):
        """
        Initializes the WordPooling module.

        Args:
            pooling_type (str): Type of pooling to perform ('average' or 'max').
        """
        super(WordPooling, self).__init__()
        assert pooling_type in [
            "average",
            "max",
        ], "pooling_type must be 'average' or 'max'"
        self.pooling_type = pooling_type

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        word_boundaries: list[tuple[int]],
    ) -> torch.Tensor:
        """
        Performs pooling over word vectors based on the attention mask.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_dim).
            attention_mask (torch.Tensor): Tensor of shape (batch_size, seq_len, seq_len) with boolean values.

        Returns:
            torch.Tensor: Tensor of shape (total_words, hidden_dim) containing pooled word vectors.
        """
        pooled_words = []

        for batch_idx, boundaries in enumerate(word_boundaries):
            batch = []
            for start, end in boundaries:
                word_vectors = hidden_states[
                    batch_idx, start:end, :
                ]  # Shape: (word_len, hidden_dim)
                if word_vectors.size(0) == 0:
                    continue  # Skip if no vectors to pool
                if self.pooling_type == "average":
                    pooled = word_vectors.mean(dim=0)
                else:
                    pooled, _ = word_vectors.max(dim=0)

                batch.append(pooled)
            pooled_words.append(batch)

        return torch.nested.nested_tensor(pooled_words)
