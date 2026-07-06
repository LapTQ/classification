from typing import Optional

import torchvision.transforms as T
from torch import nn


def create_backbone(cfg: dict) -> nn.Module:

    model_name = cfg["model"]
    num_classes = len(cfg["classes"])

    if model_name == "efficientnetv2s":
        from src.backbone.efficientnetv2s import get_efficientnetv2s

        backbone = get_efficientnetv2s(num_classes=num_classes)
    elif model_name == "resnet50":
        from src.backbone.resnet50 import get_resnet50

        backbone = get_resnet50(num_classes=num_classes)
    elif model_name == "vit_b_16":
        from src.backbone.vit_b_16 import get_vit_b_16

        backbone = get_vit_b_16(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not found")

    return backbone


def create_transform(aug_list: list) -> T.Compose:
    from src.core.augmentations import ChannelShuffle, Downscale, GaussNoise, SquarePad

    assert aug_list

    transforms = []
    for aug in aug_list:
        name = list(aug.keys())[0]
        params = aug[name] or {}

        if name == "RandomHorizontalFlip":
            transforms.append(T.RandomHorizontalFlip())
        elif name == "RandAugment":
            num_ops = params["num_ops"]
            magnitude = params["magnitude"]
            interpolation = getattr(T.InterpolationMode, params["interpolation"])
            transforms.append(
                T.RandAugment(
                    num_ops=num_ops, magnitude=magnitude, interpolation=interpolation
                )
            )
        elif name == "Downscale":
            scale_range = tuple(params["scale_range"])
            p = params["p"]
            transforms.append(Downscale(scale_range=scale_range, p=p))
        elif name == "ChannelShuffle":
            p = params["p"]
            transforms.append(ChannelShuffle(p=p))
        elif name == "ColorJitter":
            hue = params["hue"]
            transforms.append(T.ColorJitter(hue=hue))
        elif name == "RandomGrayscale":
            p = params["p"]
            transforms.append(T.RandomGrayscale(p=p))
        elif name == "Resize":
            size = (
                tuple(params["size"])
                if isinstance(params["size"], list)
                else params["size"]
            )
            transforms.append(T.Resize(size=size))
        elif name == "ToTensor":
            transforms.append(T.ToTensor())
        elif name == "Normalize":
            mean = params["mean"]
            std = params["std"]
            transforms.append(T.Normalize(mean=mean, std=std))
        elif name == "SquarePad":
            transforms.append(SquarePad())
        elif name == "GaussNoise":
            p = params["p"]
            transforms.append(GaussNoise(p=p))
        elif name == "RandomResizedCrop":
            size = (
                tuple(params["size"])
                if isinstance(params["size"], list)
                else params["size"]
            )
            scale = tuple(params["scale"])
            transforms.append(T.RandomResizedCrop(size=size, scale=scale))
        else:
            raise ValueError(f"Unknown augmentation {name}")

    return T.Compose(transforms)
