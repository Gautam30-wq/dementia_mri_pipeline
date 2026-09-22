import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation gating to recalibrate channel-wise features.
    Emphasizes channels sensitive to dementia biomarkers (e.g., ventricular enlargement).
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResSEBlock(nn.Module):
    """
    Residual block with SE gating and persistent spatial dropout for MC Dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_rate: float = 0.2):
        super(ResSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout_rate = dropout_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        # Spatial dropout kept strictly active (training=True) for MC Dropout inference
        if self.dropout_rate > 0:
            out = F.dropout2d(out, p=self.dropout_rate, training=True)
            
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CustomCNN(nn.Module):
    """
    Two-stage Custom CNN for multi-class MRI dementia staging.
    """
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.2):
        super(CustomCNN, self).__init__()
        
        # Convolutional Stem (grayscale input = 1 channel)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual-SE Stages
        self.stage1 = self._make_layer(64, 64, stride=1, blocks=2, dropout_rate=dropout_rate)
        self.stage2 = self._make_layer(64, 128, stride=2, blocks=2, dropout_rate=dropout_rate)
        self.stage3 = self._make_layer(128, 256, stride=2, blocks=2, dropout_rate=dropout_rate)
        self.stage4 = self._make_layer(256, 512, stride=2, blocks=2, dropout_rate=dropout_rate)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), # Conventional dropout before the head
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels: int, out_channels: int, stride: int, blocks: int, dropout_rate: float) -> nn.Sequential:
        layers = []
        layers.append(ResSEBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, blocks):
            layers.append(ResSEBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        features = self.stage4(x) # Shape: (B, 512, H, W) - Used for Grad-CAM extraction
        
        pooled = self.global_avg_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Standard dropout explicitly set to training=True for downstream uncertainty calculation
        pooled = F.dropout(pooled, p=self.classifier[0].p, training=True) 
        logits = self.classifier[1](pooled)
        
        if return_features:
            return logits, features
        return logits