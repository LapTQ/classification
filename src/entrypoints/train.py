import os
import yaml
from typing import Optional, Dict, Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from src.core.model import ActionResNet50
from src.core.data import ActionDataModule
from src.core.utils import get_run_dir, visualize_batch

# ================= CẤU HÌNH TRỰC TIẾP =================
CONFIG_PATH = "configs/v3.2dcnn.cluster-CNN-10--cut.yaml"  # Path tới file cấu hình
# =====================================================


def train_model(config_path: str) -> None:
    # 1. Load Initial Config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resume_path = cfg.get("resume_path")
    device = cfg.get("device", "auto")

    # 2. Setup Run Directory & Load Actual Config (if resume)
    if resume_path:
        print(f"Resuming from {resume_path}")
        run_dir = resume_path
        config_file = os.path.join(resume_path, "config.yaml")
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            # Re-apply device from the command/main config if needed, 
            # or keep it from resume config. Usually device is environmental.
            cfg["device"] = device 
    else:
        run_dir = get_run_dir(cfg["output_dir"])
        # Lưu lại config để phục vụ resume sau này
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True)

    print(f"Output directory: {run_dir}")

    # 3. Data Module
    dm = ActionDataModule(
        train_cfg=cfg["train_data"],
        val_cfg=cfg["val_data"],
        classes=cfg["classes"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
    )
    dm.setup(stage="fit")

    # Visualize 3 batches
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        visualize_batch(
            batch,
            cfg["classes"],
            os.path.join(run_dir, f"train_batch_{i}.jpg"),
            f"Train Batch {i}",
        )
    for i, batch in enumerate(val_loader):
        if i >= 3:
            break
        visualize_batch(
            batch,
            cfg["classes"],
            os.path.join(run_dir, f"val_batch_{i}.jpg"),
            f"Val Batch {i}",
        )

    # 4. Model
    model = ActionResNet50(
        num_classes=len(cfg["classes"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )

    # 5. Callbacks & Loggers
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "weights"),
        filename="best-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_last=True,
    )

    loggers = [
        CSVLogger(run_dir, name="logs"),
    ]

    # 5. Trainer
    # Xử lý device logic
    if device == "cpu":
        accelerator = "cpu"
        devices = "auto"
    else:
        accelerator = "gpu"
        devices = [device]

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=cfg.get("patience", 10)),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=loggers,
        default_root_dir=run_dir,
        precision=cfg.get("precision", 32),
    )

    # 6. Fit
    ckpt_path = os.path.join(run_dir, "weights", "last.ckpt") if resume_path else None
    if ckpt_path and not os.path.exists(ckpt_path):
        print(f"Warning: last.ckpt not found at {ckpt_path}. Training from scratch.")
        ckpt_path = None

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    train_model(CONFIG_PATH)
