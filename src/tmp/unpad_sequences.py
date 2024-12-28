import torch
from torch import nn


class UnpadSequences(nn.Module):
    def __init__(self):
        """
        Initializes the UnpadSequences module.

        Args:
            d_model (int): Dimension of the word vectors.
        """
        super(UnpadSequences, self).__init__()

    def forward(
        self, batch: list[torch.Tensor], sequence_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ """
        unique_ids, inverse_indices = torch.unique(sequence_ids, return_inverse=True)
        grouped_indices = [
            torch.nonzero(inverse_indices == idx).squeeze(-1)
            for idx in range(len(unique_ids))
        ]
        grouped_tensors = [batch[indices] for indices in grouped_indices]
        nested_tensor = torch.nested.nested_tensor(grouped_tensors)
        padded_tensor = torch.nested.to_padded_tensor(nested_tensor, padding=0)
        padding_mask = torch.all(padded_tensor == 0, dim=-1)
        return padded_tensor, padding_mask
