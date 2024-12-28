import torch
from torch import nn


class ZeroPadding(nn.Module):
    def __init__(self, d_model: int):
        """
        Initializes the ZeroPadding module.

        Args:
            d_model (int): Dimension of the word vectors.
        """
        super(ZeroPadding, self).__init__()
        padding_vector = torch.zeros(d_model, requires_grad=False)
        self.register_buffer("padding_vector", padding_vector)

    def forward(
        self, batch_pooled_vectors: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pads pooled word vectors to the maximum length in the batch.

        Args:
            batch_pooled_vectors (List[torch.Tensor]): List of tensors, each of shape (M_i, d_model).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - padded_memory: Tensor of shape (N, M_max, d_model).
                - memory_key_padding_mask: Tensor of shape (N, M_max), where True indicates padding.
        """
        batch_size = len(batch_pooled_vectors)
        d_model = self.padding_vector.size(0)
        M_max = max([vec.size(0) for vec in batch_pooled_vectors])

        # Initialize padded memory with padding vectors
        padded_memory = torch.stack(
            [
                (
                    torch.cat(
                        [
                            vec,
                            self.padding_vector.unsqueeze(0).repeat(
                                M_max - vec.size(0), 1
                            ),
                        ],
                        dim=0,
                    )
                    if vec.size(0) < M_max
                    else vec[:M_max]
                )
                for vec in batch_pooled_vectors
            ],
            dim=0,
        )  # Shape: (N, M_max, d_model)

        # Create padding masks
        memory_key_padding_mask = torch.tensor(
            [
                (
                    [False] * vec.size(0) + [True] * (M_max - vec.size(0))
                    if vec.size(0) < M_max
                    else [False] * M_max
                )
                for vec in batch_pooled_vectors
            ],
            dtype=torch.bool,
            device=self.padding_vector.device,
        )  # Shape: (N, M_max)

        return padded_memory, memory_key_padding_mask
