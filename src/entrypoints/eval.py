import os
import yaml
from typing import Any, Dict, Union
import torch
import pytorch_lightning as pl
from src.core.model import ActionResNet50
from src.core.data import ActionDataModule

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = "/home/laptq/laptq-fs26-shoplifting-detection/runs/classification/fs26/v3.2dcnn.cluster-CNN-10--cut-l4/weights/best-epoch=02-val_acc=0.333.ckpt"
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
    dm = ActionDataModule(
        train_cfg={},
        val_cfg=cfg["val_data"],
        classes=cfg["classes"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
    )

    # 3. Load Model
    model = ActionResNet50.load_from_checkpoint(ckpt_path)
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
