import torchvision.transforms as T
from torch import nn
from src.core.rescaler import IBatchRescaler


def create_backbone(cfg: dict) -> nn.Module:

    model_name = cfg["model"]
    num_classes = len(cfg["classes"])

    if model_name == "efficientnetv2s":
        from src.backbone.efficientnetv2s import get_efficientnetv2s

        backbone = get_efficientnetv2s(num_classes=num_classes)
    elif model_name == "efficientnetv2m":
        from src.backbone.efficientnetv2m import get_efficientnetv2m

        backbone = get_efficientnetv2m(num_classes=num_classes)
    elif model_name == "resnet50":
        from src.backbone.resnet50 import get_resnet50

        backbone = get_resnet50(num_classes=num_classes)
    elif model_name == "vit_b_16":
        from src.backbone.vit_b_16 import get_vit_b_16

        backbone = get_vit_b_16(num_classes=num_classes)
    elif model_name == "dinov3_vits16":
        from src.backbone.dinov3_vits16 import get_dinov3_vits16

        backbone = get_dinov3_vits16(
            num_classes=num_classes,
            pretrained_weights=cfg["dinov3_pretrain_weights"],
        )
    else:
        raise ValueError(f"Model {model_name} not found")

    return backbone


def create_transform(aug_list: list) -> T.Compose:
    from src.core.augmentations import ChannelShuffle, Downscale, GaussNoise, SquarePad, GaussianBlur

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
        elif name == "GaussianBlur":
            blur_limit = params["blur_limit"]
            p = params["p"]
            transforms.append(GaussianBlur(blur_limit=blur_limit, p=p))
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


def create_rescaler(cfg: dict) -> IBatchRescaler | None:
    if cfg["multiscale"] is None:
        return None

    ms_cfg: dict = cfg["multiscale"]
    rescaler_type: str = ms_cfg["type"]

    if rescaler_type == "random":
        from src.core.rescaler import RandomBatchRescaler
        
        return RandomBatchRescaler(
            sizes=ms_cfg["sizes"],
            step_interval=ms_cfg["step_interval"],
        )
    elif rescaler_type == "progressive":
        from src.core.rescaler import ProgressiveBatchRescaler

        return ProgressiveBatchRescaler(
            milestones=ms_cfg["milestones"],
            step_interval=ms_cfg["step_interval"],
        )
    else:
        raise ValueError(f"Unknown rescaler type: {rescaler_type}")


