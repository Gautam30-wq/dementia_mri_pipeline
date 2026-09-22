import torch
import torch.nn.functional as F
import numpy as np
import cv2 # Used for spatial resizing of the heatmap

class GradCAM:
    """
    Extracts spatial attribution heatmaps from a trained CNN.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register PyTorch hooks to capture the forward feature maps and backward gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> tuple:
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass (ensure spatial dropout is bypassed during explanation generation)
        logits = self.model(input_tensor)
        
        # If no specific class is targeted, explain the model's top prediction
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()
            
        class_score = logits[0, target_class]
        class_score.backward(retain_graph=True)
        
        # 1. Compute channel-wise importance weights via global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 2. Compute the weighted combination of the feature map channels
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        
        # 3. Apply ReLU rectification to retain only positive-influence regions
        cam = F.relu(cam)
        
        # 4. Resize to match the original input resolution (e.g., 224x224) and normalize to [0, 1]
        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        if np.max(cam) != 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
            
        return cam, target_class

def extract_roi_mask(heatmap: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """
    Applies per-image quantile thresholding to generate a binary ROI mask.
    Isolates the most salient sub-region (e.g., hippocampal or cortical-thinning areas).
    """
    # Calculate the threshold intensity value at the specified percentile
    threshold = np.percentile(heatmap, percentile)
    
    # Generate binary mask: 1 for pixels >= threshold, 0 otherwise
    roi_mask = (heatmap >= threshold).astype(np.uint8)
    
    return roi_mask

# Example instantiation mapping to the CustomCNN's final stage:
# cam_extractor = GradCAM(model, model.stage4)
# heatmap, pred_class = cam_extractor.generate_heatmap(image_tensor)
# roi_mask = extract_roi_mask(heatmap, percentile=90.0)