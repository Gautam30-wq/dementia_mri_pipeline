import torch
from torchvision import transforms

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Data augmentation and transformation pipeline for the training set.
    Includes lightweight augmentations to prevent overfitting on MRI scans.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Slight affine transformations simulate patient positioning differences in MRI machines
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        # Standard normalization for grayscale medical images
        # Maps the [0.0, 1.0] tensor range to a zero-mean, unit-variance distribution
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_val_test_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Transformation pipeline for validation and testing sets.
    Strictly deterministic; no random augmentations are applied.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])