import os
import random
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as T
from src.core.augmentations import SquarePad, ChannelShuffle, Downscale, GaussNoise


class ActionDataset(Dataset):
    def __init__(
        self,
        class_to_files: Dict[str, Union[str, List[str]]],
        classes: List[str],
        transform: Optional[T.Compose] = None,
    ) -> None:
        """
        Args:
            class_to_files: dict {class_name: [list of .txt paths]}
            classes: list of class names in specific order
            transform: torchvision transforms
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for idx, cls_name in enumerate(classes):
            txt_files = class_to_files.get(cls_name, [])
            # Support both string and list of strings
            if isinstance(txt_files, str):
                txt_files = [txt_files]

            for txt_file in txt_files:
                with open(txt_file, "r") as f:
                    paths = [line.strip() for line in f if line.strip()]
                    self.image_paths.extend(paths)
                    self.labels.extend([idx] * len(paths))

        idxs = list(range(len(self.image_paths)))
        random.seed(42)
        random.shuffle(idxs)
        self.image_paths = [self.image_paths[i] for i in idxs]
        self.labels = [self.labels[i] for i in idxs]

        print(
            f"Dataset initialized with {len(self.image_paths)} images across {len(classes)} classes."
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label, img_path


class ClassifyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_cfg: Dict[str, Any],
        val_cfg: Dict[str, Any],
        classes: List[str],
        train_transform: Optional[T.Compose] = None,
        val_transform: Optional[T.Compose] = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        self.classes = classes
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_ds = ActionDataset(
                self.train_cfg, self.classes, self.train_transform
            )
            self.val_ds = ActionDataset(self.val_cfg, self.classes, self.val_transform)
        if stage == "test":
            self.test_ds = ActionDataset(self.val_cfg, self.classes, self.val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
