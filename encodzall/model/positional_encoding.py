# positional_encoding.py
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        """
        Sinusoidal positional encoding module.

        Args:
            d_model (int): Dimension of the embeddings.
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate positional encodings for a sequence length.

        Args:
            seq_length (int): Length of the input sequence.
            device (torch.device): Device where the tensor will be placed.

        Returns:
            torch.Tensor: Positional encoding tensor of shape (seq_length, d_model).
        """
        position = torch.arange(
            seq_length, dtype=torch.float32, device=device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / self.d_model)
        )
        pos_encoding = torch.zeros((seq_length, self.d_model), device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
