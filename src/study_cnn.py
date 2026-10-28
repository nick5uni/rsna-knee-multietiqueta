"""CNN propia por estudio. Requiere PyTorch en Kaggle."""
import torch
from torch import nn


class StudyCNN(nn.Module):
    def __init__(self, num_labels=12):
        super().__init__()
        blocks = []
        channels = 1
        for width in (16, 32, 64):
            blocks.extend([nn.Conv2d(channels, width, 3, padding=1),
                           nn.GroupNorm(4, width), nn.ReLU(), nn.MaxPool2d(2)])
            channels = width
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(64, num_labels)

    def forward(self, images, slice_mask):
        if images.ndim != 5 or images.shape[2] != 1 or slice_mask.shape != images.shape[:2]:
            raise ValueError('Entrada [B,S,1,H,W] y mascara [B,S].')
        if slice_mask.dtype != torch.bool or not bool(slice_mask.any(dim=1).all()):
            raise ValueError('Cada estudio requiere al menos un corte valido y mascara booleana.')
        valid_features = self.features(images[slice_mask])
        features = valid_features.new_zeros((*slice_mask.shape, valid_features.shape[-1]))
        features[slice_mask] = valid_features
        pooled = features.sum(dim=1)/slice_mask.sum(dim=1, keepdim=True)
        return self.classifier(pooled)  # logits; BCEWithLogits, sin sigmoid previo
