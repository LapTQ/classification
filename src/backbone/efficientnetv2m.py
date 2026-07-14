import torch
import torch.nn as nn
from torchvision import models


def get_efficientnetv2m(num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
