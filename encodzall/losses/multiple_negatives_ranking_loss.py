import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Optional


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct=F.cosine_similarity) -> None:
        """
        Custom loss that takes precomputed anchor and positive embeddings.

        Args:
            scale: Scalar to scale the similarity scores.
            similarity_fct: Function to compute similarity between embeddings.
        """
        super().__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        anchors: Tensor,
        positives: Tensor,
        hard_negatives: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the loss.

        Args:
            anchors: Tensor of shape (batch_size, embedding_dim).
            positives: Tensor of shape (batch_size, embedding_dim).
            hard_negatives: (Optional) Tensor of shape (batch_size, num_hard_negatives, embedding_dim).

        Returns:
            Scalar loss value.
        """
        batch_size = anchors.size(0)

        # normalize
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)

        # Compute similarity between anchors and all positives
        positives = positives.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        similarity_matrix = self.similarity_fct(
            anchors.unsqueeze(1), positives.expand(-1, batch_size, -1)
        )  # (batch_size, batch_size)

        # Scale the similarities
        similarity_matrix = similarity_matrix * self.scale

        # The label for each anchor is the index of its positive in the batch
        labels = torch.arange(batch_size).to(anchors.device)

        loss = self.cross_entropy_loss(similarity_matrix, labels)
        return loss

    def get_config_dict(self) -> dict:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
