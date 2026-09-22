import torch
import numpy as np
from scipy.stats import entropy

# --- CNN Branch Uncertainty ---

def compute_mc_dropout_uncertainty(model: torch.nn.Module, input_tensor: torch.Tensor, num_passes: int = 30) -> tuple:
    """
    Computes MC Dropout uncertainty metrics for the CNN branch.
    Relies on spatial dropout layers remaining active (training=True) inside the model definition.
    """
    model.eval() # Freezes BatchNorm statistics while custom dropout remains active
    
    stochastic_preds = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=-1)
            stochastic_preds.append(probs.cpu().numpy())
            
    # Shape: (num_passes, batch_size, num_classes)
    stochastic_preds = np.stack(stochastic_preds) 
    
    # Averaged predictive distribution
    mean_probs = np.mean(stochastic_preds, axis=0)
    
    # Predictive Entropy (Total Uncertainty)
    predictive_entropy = entropy(mean_probs, axis=-1)
    
    # Expected Entropy (Aleatoric component)
    expected_entropy = np.mean(entropy(stochastic_preds, axis=-1), axis=0)
    
    # Mutual Information / BALD score (Epistemic component)
    bald_score = predictive_entropy - expected_entropy
    
    return mean_probs, predictive_entropy, bald_score

# --- Radiomics Branch Uncertainty ---

def compute_rf_uncertainty(rf_pipeline, X_input: np.ndarray) -> tuple:
    """
    Computes uncertainty for the Random Forest radiomics branch using tree-ensemble variance.
    """
    rf_model = rf_pipeline.named_steps['classifier']
    preprocessed_X = rf_pipeline[:-1].transform(X_input)
    
    tree_preds = []
    for tree in rf_model.estimators_:
        tree_preds.append(tree.predict_proba(preprocessed_X))
        
    # Shape: (num_trees, batch_size, num_classes)
    tree_preds = np.stack(tree_preds)
    
    mean_probs = np.mean(tree_preds, axis=0)
    
    # Variance of predicted probabilities across constituent trees (infinitesimal jackknife proxy)
    variance = np.var(tree_preds, axis=0)
    mean_variance = np.mean(variance, axis=-1) 
    
    return mean_probs, mean_variance

def compute_xgboost_uncertainty(xgb_pipeline, X_input: np.ndarray) -> tuple:
    """
    Computes uncertainty for the XGBoost radiomics branch using softmax entropy.
    """
    # XGBoost does not natively support tree-level variance extraction like RF, 
    # so the entropy of the probability distribution is used.
    mean_probs = xgb_pipeline.predict_proba(X_input)
    softmax_entropy = entropy(mean_probs, axis=-1)
    
    return mean_probs, softmax_entropy