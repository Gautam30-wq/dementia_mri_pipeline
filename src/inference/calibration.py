import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class TemperatureScaler(nn.Module):
    """
    Fits a single scalar temperature parameter (T) on a held-out calibration set 
    to minimize Negative Log-Likelihood (NLL).
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        # Initialize temperature to 1.5 (empirically a good starting point)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Dividing logits by T flattens overconfident probability distributions
        # without altering the arg-max prediction[cite: 2].
        return logits / self.temperature

    def fit(self, valid_logits: torch.Tensor, valid_labels: torch.Tensor):
        """
        Optimizes the temperature parameter using LBFGS.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(valid_logits), valid_labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_closure)
        print(f"Calibrated Temperature (T): {self.temperature.item():.4f}")

def fuse_branches(cnn_probs: np.ndarray, rad_probs: np.ndarray, 
                  cnn_uncertainty: np.ndarray, rad_uncertainty: np.ndarray, 
                  alpha: float = 0.6) -> tuple:
    """
    Fuses the CNN and radiomics branches using a convex combination[cite: 2].
    
    Args:
        cnn_probs: Calibrated probabilities from the CNN.
        rad_probs: Probabilities from the XGBoost/RF classifier.
        cnn_uncertainty: Normalized MC Dropout entropy.
        rad_uncertainty: Normalized tree-ensemble variance or softmax entropy.
        alpha: Mixing coefficient balancing CNN vs. Radiomics evidence (e.g., 0.6 weights CNN at 60%).
    """
    # Fuse class probabilities
    fused_probs = (alpha * cnn_probs) + ((1 - alpha) * rad_probs)
    
    # Predict final stage via arg-max
    fused_predictions = np.argmax(fused_probs, axis=-1)
    
    # Fuse uncertainty metrics using the identical mixing coefficient[cite: 2]
    fused_uncertainty = (alpha * cnn_uncertainty) + ((1 - alpha) * rad_uncertainty)
    
    # Derive a single confidence score (inverse of fused uncertainty)
    fused_confidence = 1.0 - fused_uncertainty
    
    return fused_probs, fused_predictions, fused_confidence