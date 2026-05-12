import torchvision.transforms.functional as F
from PIL import Image
import torch
from typing import Tuple
import albumentations as A
import numpy as np


class SquarePad:
    """
    Pad the image to a square.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, 0, "constant")

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class BaseAlbumentationConverter:

    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        result = self.transform(image=img_np)["image"]
        return Image.fromarray(result)


class ChannelShuffle(BaseAlbumentationConverter):
    def __init__(self, p=0.5):
        self.transform = A.ChannelShuffle(p=p)


class Downscale(BaseAlbumentationConverter):
    def __init__(self, scale_range, p=0.5):
        self.transform = A.Downscale(scale_range=scale_range, p=p)


class GaussNoise(BaseAlbumentationConverter):
    def __init__(self, std_range=(0.1, 0.1), p=0.5):
        self.transform = A.GaussNoise(std_range=std_range, p=p)
