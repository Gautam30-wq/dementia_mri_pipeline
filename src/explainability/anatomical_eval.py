import numpy as np

def compute_anatomical_overlap(grad_cam_mask: np.ndarray, reference_atlas_mask: np.ndarray) -> dict:
    """
    Calculates Dice Similarity Coefficient and Intersection-over-Union (IoU) 
    between the model's ROI and a verified anatomical atlas[cite: 1, 2].
    """
    # Ensure boolean arrays
    pred = grad_cam_mask.astype(bool)
    ref = reference_atlas_mask.astype(bool)
    
    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    
    iou = intersection / union if union > 0 else 0.0
    dice = (2. * intersection) / (pred.sum() + ref.sum()) if (pred.sum() + ref.sum()) > 0 else 0.0
    
    return {'Dice': dice, 'IoU': iou}