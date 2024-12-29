# word_pooling.py
import torch
import torch.nn as nn
from typing import List, Tuple


class WordPooling(nn.Module):
    def __init__(self, pooling_type: str = "average"):
        """
        Initializes the WordPooling module.

        Args:
            pooling_type (str): Type of pooling to perform ('average' or 'max').
        """
        super(WordPooling, self).__init__()
        if pooling_type not in ["average", "max"]:
            raise ValueError("pooling_type must be 'average' or 'max'")
        self.pooling_type = pooling_type

    def forward(
        self, hidden_states: torch.Tensor, word_boundaries: List[List[Tuple[int, int]]]
    ) -> torch.Tensor:
        """
        Performs pooling over word vectors based on the word boundaries.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_dim).
            word_boundaries (List[List[Tuple[int, int]]]): Word boundaries per batch element.

        Returns:
            torch.Tensor: Tensor of shape (total_words, hidden_dim) containing pooled word vectors.
        """
        pooled_words = []

        for batch_idx, boundaries in enumerate(word_boundaries):
            for start, end in boundaries:
                word_vectors = hidden_states[batch_idx, start:end, :]
                if word_vectors.size(0) == 0:
                    continue  # Skip if no vectors to pool
                if self.pooling_type == "average":
                    pooled = word_vectors.mean(dim=0)
                else:
                    pooled, _ = word_vectors.max(dim=0)
                pooled_words.append(pooled)

        if pooled_words:
            return torch.stack(pooled_words, dim=0)  # Shape: (total_words, hidden_dim)
        else:
            return torch.empty(
                0, hidden_states.size(-1), device=hidden_states.device
            )  # Return empty tensor if no words
