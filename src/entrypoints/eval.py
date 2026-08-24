import os
import yaml
from typing import Any, Dict, Union
import torch
import pytorch_lightning as pl
from src.core.model import ClassifyModel
from src.core.data import ClassifyDataModule
from src.entrypoints.bootstrap import create_backbone, create_transform

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = "models/checkpoints/fs26/action_recognition/classification/v22.efficientv2s.for_CNN_8_classes_manually_selected+flux_set_1_2+masked/weights/best-epoch=29-val_f1=0.435.ckpt"
# =====================================================


def evaluate_model(ckpt_path: str) -> None:
    # 1. Load Config (tìm config.yaml ở thư mục cha của checkpoint)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    accelerator = cfg["accelerator"]
    devices = cfg["devices"]

    # 2. Data Module
    val_transform = create_transform(cfg["val_augment"])

    dm = ClassifyDataModule(
        train_cfg={},
        val_cfg=cfg["val_data"],
        classes=cfg["classes"],
        val_transform=val_transform,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    # 3. Load Model
    backbone = create_backbone(cfg)
    model = ClassifyModel.load_from_checkpoint(ckpt_path, model=backbone)
    model.class_names = cfg["classes"]

    # 4. Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=run_dir,
    )

    # 5. Evaluate
    print(f"Evaluating checkpoint: {ckpt_path}")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    evaluate_model(CKPT_PATH)
