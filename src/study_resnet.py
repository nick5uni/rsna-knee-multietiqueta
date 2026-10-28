"""ResNet18 por estudio con la interfaz comun (images, slice_mask)."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class StudyResNet18(nn.Module):
    def __init__(self, num_labels=12, weights="imagenet"):
        super().__init__()
        if num_labels != 12:
            raise ValueError("El contrato requiere doce salidas.")
        if weights not in ("imagenet", None):
            raise ValueError("weights debe ser imagenet o None para pruebas tecnicas.")
        selected = ResNet18_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
        # Si la descarga falla, propagar el error: no usar pesos aleatorios en silencio.
        self.backbone = resnet18(weights=selected)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_labels)
        self.initialization = "ImageNet_IMAGENET1K_V1" if selected is not None else "random_technical_only"
        self.register_buffer("imagenet_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images, slice_mask):
        if (images.ndim != 5 or images.shape[2] != 1
                or tuple(slice_mask.shape) != tuple(images.shape[:2])):
            raise ValueError("Entrada [B,S,1,H,W] y mascara [B,S].")
        if slice_mask.dtype != torch.bool or not bool(slice_mask.any(dim=1).all()):
            raise ValueError("Cada estudio requiere un corte valido y mascara booleana.")
        valid = images[slice_mask]
        if not valid.is_floating_point() or not bool(torch.isfinite(valid).all()):
            raise ValueError("Cortes validos deben ser flotantes y finitos.")
        if not bool(((valid >= 0) & (valid <= 1)).all()):
            raise ValueError("El lector debe entregar valores en 0..1, sin normalizacion ImageNet previa.")
        rgb = valid.repeat(1, 3, 1, 1)
        features_valid = self.backbone((rgb-self.imagenet_mean)/self.imagenet_std)
        features = features_valid.new_zeros((*slice_mask.shape, 512))
        features[slice_mask] = features_valid
        pooled = features.sum(dim=1)/slice_mask.sum(dim=1, keepdim=True)
        return self.classifier(pooled)  # logits; no sigmoid antes de BCE/focal
