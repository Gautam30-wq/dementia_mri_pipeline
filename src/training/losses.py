import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalFocalLoss(nn.Module):
    """
    Composite loss function combining:
    1. Inverse-frequency-weighted focal loss (handles class imbalance & hard examples).
    2. Squared Earth Mover's Distance (EMD) loss (penalizes ordinal severity distance).
    """
    def __init__(self, alpha_weights: torch.Tensor = None, gamma: float = 2.0, emd_weight: float = 1.0):
        super(OrdinalFocalLoss, self).__init__()
        self.alpha_weights = alpha_weights  # Tensor of inverse frequency weights per class
        self.gamma = gamma                  # Focusing parameter for hard examples
        self.emd_weight = emd_weight        # Mixing coefficient for the EMD ordinal penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        num_classes = logits.size(-1)
        
        # --- 1. Focal Loss Component with Class Weighting ---
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of the true class
        
        if self.alpha_weights is not None:
            alpha_t = self.alpha_weights[targets].to(logits.device)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        focal_loss = focal_loss.mean()

        # --- 2. Squared EMD Ordinal Loss Component ---
        # Convert discrete integer targets to one-hot label distributions
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
        
        # Compute Cumulative Distribution Functions (CDFs) along the ordered stage axis
        cdf_pred = torch.cumsum(probs, dim=-1)
        cdf_true = torch.cumsum(one_hot_targets, dim=-1)
        
        # Compute squared differences between predicted and true CDFs
        emd_loss = torch.sum((cdf_pred - cdf_true) ** 2, dim=-1).mean()

        # --- 3. Composite Objective ---
        total_loss = focal_loss + (self.emd_weight * emd_loss)
        return total_loss