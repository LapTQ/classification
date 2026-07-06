from torch import nn
from torchvision import models


def get_vit_b_16(num_classes: int) -> nn.Module:
    """Creates a ViT-B/16 model for multiclass classification.

    Args:
        num_classes (int): Number of target classes.

    Returns:
        nn.Module: The modified ViT model.
    """
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model
