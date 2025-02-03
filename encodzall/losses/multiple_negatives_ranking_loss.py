import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Optional

from .focal_loss import FocalLoss


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0) -> None:
        super().__init__()
        self.scale = scale
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = FocalLoss(gamma=1)

    def forward(
        self,
        anchors: Tensor,
        positives: Tensor,
        hard_negatives: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = anchors.size(0)

        # Compute pairwise cosine similarities
        # dim=1 means compute similarity along embedding dimension
        similarity_matrix = F.cosine_similarity(
            anchors.unsqueeze(1),  # shape: (batch_size, 1, embedding_dim)
            positives.unsqueeze(0).detach(),  # shape: (1, batch_size, embedding_dim)
            dim=2,
        )

        similarity_matrix = similarity_matrix * self.scale
        labels = torch.arange(0, similarity_matrix.size(0), device=anchors.device)

        return self.loss_fn(similarity_matrix, labels)

    def get_config_dict(self) -> dict:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
