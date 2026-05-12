from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import torchmetrics


class ActionResNet50(pl.LightningModule):
    def __init__(
        self, num_classes: int, lr: float = 1e-4, weight_decay: float = 1e-5
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = self.train_acc(logits, y)
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
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_acc(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log(
            "val_acc", self.val_acc, on_epoch=True, prog_bar=True, batch_size=x.shape[0]
        )
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log("test_loss", loss, on_epoch=True, batch_size=x.shape[0])
        self.log("test_acc", self.val_acc, on_epoch=True, batch_size=x.shape[0])

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
