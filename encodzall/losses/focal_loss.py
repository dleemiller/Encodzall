import torch
import torch.nn as nn
import torch.nn.functional as F
from encodzall import PAD_BYTE

class FocalLoss(nn.Module):
    """
    Focal Loss (https://arxiv.org/abs/1708.02002)

    Args:
        alpha (float): Weighting factor for the rare class. Defaults to 1.0.
        gamma (float): Focusing parameter to down-weight easy examples
                       and focus training on hard negatives. Defaults to 0.5.
        reduction (str): Specifies the reduction to apply to the output.
                         Options: 'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """
    def __init__(self, alpha=1.0, gamma=0.5, reduction='mean', ignore_index=PAD_BYTE):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index=ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Focal Loss between `inputs` and `targets`.
        
        Args:
            inputs (Tensor): Logits from the model of shape (N, C) for multi-class
                             or (N, 1) / (N,) for binary.
            targets (Tensor): Ground-truth class indices of shape (N,) for multi-class
                              or (N,) for binary. 
                             
        Returns:
            Tensor: Scalar loss if `reduction` is 'mean' or 'sum', otherwise
                    the per-sample loss of shape (N,).
        """
        # Compute the cross-entropy loss per sample (without reduction).
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        
        # Convert cross-entropy loss to probability of being correct: p_t = exp(-CE).
        pt = torch.exp(-ce_loss)
        
        # The Focal Loss formula: FL = alpha * (1 - p_t)^gamma * CE
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

