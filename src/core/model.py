import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision import models


class ClassifyModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        class_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)
        self.class_names = class_names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = self.train_acc(logits, y)
        f1 = self.train_f1(logits, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        self.log(
            "train_f1",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log(
            "val_acc", self.val_acc, on_epoch=True, prog_bar=True, batch_size=x.shape[0]
        )
        self.log(
            "val_f1", self.val_f1, on_epoch=True, prog_bar=True, batch_size=x.shape[0]
        )
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.test_cm.update(logits, y)
        self.log("test_loss", loss, on_epoch=True, batch_size=x.shape[0])
        self.log("test_acc", self.test_acc, on_epoch=True, batch_size=x.shape[0])
        self.log("test_f1", self.test_f1, on_epoch=True, batch_size=x.shape[0])
        return loss

    def on_test_epoch_end(self) -> None:
        cm = self.test_cm.compute()
        self.test_cm.reset()

        # Normalize by row (Actual class)
        cm_sum = cm.sum(dim=1, keepdim=True)
        cm_norm = cm.float() / (cm_sum + 1e-7)
        cm_norm = cm_norm.cpu().numpy()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names if self.class_names else "auto",
            yticklabels=self.class_names if self.class_names else "auto",
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Normalized Confusion Matrix")

        save_path = os.path.join(self.trainer.default_root_dir, "confusion_matrix.png")
        plt.savefig(save_path, bbox_inches="tight")
        print(f"\n[INFO] Normalized confusion matrix saved to: {save_path}")
        plt.close()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # 1. Warmup scheduler: Tăng LR từ lr/10 lên lr trong 5 epochs đầu
        warmup_epochs = 3
        main_epochs = self.trainer.max_epochs - warmup_epochs

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )

        # 2. Cosine Annealing scheduler: Giảm LR theo đường cosine
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=self.hparams.lr * 0.01
        )

        # 3. Kết hợp 2 scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
