import os

import torch
from torch import nn


class DINOv3Classifier(nn.Module):
    """Classification model wrapping DINOv3 backbone with a linear head."""

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        # in_features: int = getattr(backbone, "embed_dim", 384)
        self.head = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat: torch.Tensor = self.backbone(x)
        return self.head(feat)


def load_dinov3_weights(model: nn.Module, weights_path: str) -> None:
    """Loads official DINOv3 pretrain weights into DinoVisionTransformer backbone.

    Args:
        model (nn.Module): Target DinoVisionTransformer model.
        weights_path (str): Path to DINOv3 checkpoint file.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)


def get_dinov3_vits16(
    num_classes: int,
    pretrained_weights: str,
) -> nn.Module:
    """Creates a DINOv3 ViT-S/16 model for multiclass classification using PyTorch Hub.

    Args:
        num_classes (int): Number of target classes.
        pretrained_weights (str): Path to pretrained weights file.

    Returns:
        nn.Module: The modified DINOv3 ViT-S/16 classification model.
    """
    backbone: nn.Module = torch.hub.load(
        "facebookresearch/dinov3",
        "dinov3_vits16",
        pretrained=False,
        trust_repo=True,
    )

    load_dinov3_weights(model=backbone, weights_path=pretrained_weights)

    return DINOv3Classifier(backbone=backbone, num_classes=num_classes)
