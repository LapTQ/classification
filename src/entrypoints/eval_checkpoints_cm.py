import argparse
import csv
import os
from pathlib import Path
import re
import sys
from typing import Any

# Đảm bảo import được src.* khi chạy trực tiếp từ bất kỳ thư mục nào
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm
import yaml

from src.core.data import ActionDataset
from src.core.model import ClassifyModel
from src.entrypoints.bootstrap import create_backbone, create_transform

# ================= CẤU HÌNH TRỰC TIẾP =================
CHECKPOINT_DIR = "models/checkpoints/fs26/action_recognition/classification/v31.efficientv2s.for_CNN_8_classes_manually_selected+flux_set_1_2_3_4_5_7_8_9_10_11_12+sensenova_set_1_2_2"
OUTPUT_DIR = "data/tmp/cm"
DEVICE = "cuda:0"  # GPU device để inference ("cuda:0", "cuda:3", "cpu", ...)
BATCH_SIZE = 128  # Nếu None thì dùng batch_size trong config.yaml
NUM_WORKERS = 16  # Nếu None thì dùng num_workers trong config.yaml
# =====================================================


def get_checkpoint_files(weights_dir: str) -> list[str]:
    """Lấy danh sách các checkpoint .ckpt và sắp xếp theo thứ tự epoch."""
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Thư mục weights không tồn tại: {weights_dir}")

    all_files: list[str] = [
        f for f in os.listdir(weights_dir) if f.endswith(".ckpt")
    ]
    if not all_files:
        raise FileNotFoundError(f"Không tìm thấy file .ckpt nào trong {weights_dir}")

    def sort_key(name: str) -> tuple[int, int, str]:
        match = re.search(r"epoch=(\d+)", name)
        if "best" in name:
            priority = 1
        elif "last" in name:
            priority = 2
        else:
            priority = 0
        epoch_idx = int(match.group(1)) if match else 9999
        return (priority, epoch_idx, name)

    return sorted(all_files, key=sort_key)


def plot_and_save_confusion_matrix(
    cm_tensor: torch.Tensor,
    class_names: list[str],
    save_path: str,
    title: str,
    acc: float,
    f1: float,
) -> None:
    """Vẽ và lưu normalized confusion matrix ra file ảnh."""
    # Chuẩn hóa theo hàng (Actual class)
    cm_sum = cm_tensor.sum(dim=1, keepdim=True)
    cm_norm = cm_tensor.float() / (cm_sum + 1e-7)
    cm_norm_np = cm_norm.cpu().numpy()

    label_mapping: dict[str, str] = {
        "cho_tay_vao_tui_quan": "Hand into pant pocket",
        "cho_tay_vao_tui_ao": "Hand into shirt pocket",
        "cho_tay_vao_tui_deo_tren_nguoi": "Hand into wearing bag",
        "cho_tay_vao_tui_cam_tren_tay": "Hand into holding bag",
        "cho_tay_vao_gio_xe_hang": "Hand into basket/cart",
        "cho_tay_vao_ke": "Hand into shelf",
        "tay_cam_san_pham": "Holding product",
        "tay_khong_cam_san_pham": "Not holding product",
    }

    display_labels: list[str] = [
        label_mapping[name] if name in label_mapping else name
        for name in class_names
    ]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm_np,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"Normalized Confusion Matrix - {title}\nAccuracy: {acc:.4f} | Macro F1: {f1:.4f}"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def evaluate_all_checkpoints(
    ckpt_dir: str,
    output_dir: str,
    device_str: str = "cuda:0",
    batch_size: int | None = None,
    num_workers: int | None = None,
    skip_existing: bool = False,
    max_checkpoints: int | None = None,
) -> None:
    """Duyệt qua tất cả checkpoint, tính confusion matrix và lưu ảnh vào output_dir."""
    if os.path.isdir(os.path.join(ckpt_dir, "weights")):
        weights_dir = os.path.join(ckpt_dir, "weights")
        run_dir = ckpt_dir
    else:
        weights_dir = ckpt_dir
        run_dir = os.path.dirname(ckpt_dir)

    config_file = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_files = get_checkpoint_files(weights_dir)
    if max_checkpoints is not None:
        checkpoint_files = checkpoint_files[:max_checkpoints]
    print(
        f"[INFO] Tìm thấy {len(checkpoint_files)} checkpoints để đánh giá trong {weights_dir}"
    )

    actual_batch_size: int = batch_size if batch_size is not None else cfg["batch_size"]
    actual_num_workers: int = num_workers if num_workers is not None else cfg["num_workers"]

    # 1. Khởi tạo DataModule & DataLoader
    classes: list[str] = cfg["classes"]
    num_classes: int = len(classes)
    val_transform = create_transform(cfg["val_augment"])
    val_ds = ActionDataset(cfg["val_data"], classes, val_transform)

    device = torch.device(
        device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu"
    )
    print(f"[INFO] Sử dụng device: {device}")

    val_loader = DataLoader(
        val_ds,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=actual_num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Khởi tạo model backbone & metrics
    backbone = create_backbone(cfg)
    model = ClassifyModel(model=backbone, num_classes=num_classes)
    model.to(device)
    model.eval()

    acc_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes
    ).to(device)
    f1_metric = torchmetrics.F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    # Bảng tổng hợp kết quả
    summary_records: list[dict[str, Any]] = []
    summary_csv_path = os.path.join(output_dir, "metrics_summary.csv")

    if skip_existing and os.path.exists(summary_csv_path):
        with open(summary_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary_records.append(
                    {
                        "checkpoint": row["checkpoint"],
                        "accuracy": float(row["accuracy"]),
                        "macro_f1": float(row["macro_f1"]),
                        "image_path": row["image_path"],
                    }
                )

    existing_ckpts: set[str] = {r["checkpoint"] for r in summary_records}

    for idx, ckpt_name in enumerate(checkpoint_files, start=1):
        ckpt_stem = os.path.splitext(ckpt_name)[0]
        cm_image_path = os.path.join(output_dir, f"{ckpt_stem}.png")

        if skip_existing and (ckpt_name in existing_ckpts or os.path.exists(cm_image_path)):
            print(
                f"[{idx}/{len(checkpoint_files)}] Bỏ qua {ckpt_name} (đã tồn tại)"
            )
            continue

        ckpt_path = os.path.join(weights_dir, ckpt_name)
        print(f"\n[{idx}/{len(checkpoint_files)}] Đang đánh giá: {ckpt_name}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        acc_metric.reset()
        f1_metric.reset()
        cm_metric.reset()

        with torch.no_grad():
            for x, y, _ in tqdm(
                val_loader,
                desc=f"Inference {ckpt_name}",
                leave=False,
            ):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                acc_metric.update(logits, y)
                f1_metric.update(logits, y)
                cm_metric.update(logits, y)

        acc_val: float = float(acc_metric.compute().item())
        f1_val: float = float(f1_metric.compute().item())
        cm_tensor: torch.Tensor = cm_metric.compute()

        plot_and_save_confusion_matrix(
            cm_tensor=cm_tensor,
            class_names=classes,
            save_path=cm_image_path,
            title=ckpt_stem,
            acc=acc_val,
            f1=f1_val,
        )
        print(
            f" => Acc: {acc_val:.4f} | Macro F1: {f1_val:.4f} | Đã lưu CM: {cm_image_path}"
        )

        summary_records.append(
            {
                "checkpoint": ckpt_name,
                "accuracy": acc_val,
                "macro_f1": f1_val,
                "image_path": cm_image_path,
            }
        )

    # Lưu metrics summary ra CSV
    if summary_records:
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["checkpoint", "accuracy", "macro_f1", "image_path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_records)
        print(f"\n[INFO] Đã lưu bảng tổng hợp metrics tại: {summary_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá và tạo Confusion Matrix cho tất cả checkpoint trong thư mục run."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=CHECKPOINT_DIR,
        help="Đường dẫn thư mục run hoặc thư mục weights chứa các file .ckpt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Thư mục lưu ảnh confusion matrix",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Device để inference (ví dụ: cuda:0, cuda:3, cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (nếu None thì lấy theo config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Số lượng workers cho DataLoader",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Bỏ qua các checkpoint đã có file ảnh confusion matrix",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Giới hạn số lượng checkpoint đánh giá (dùng để test nhanh)",
    )

    args = parser.parse_args()

    evaluate_all_checkpoints(
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        skip_existing=args.skip_existing,
        max_checkpoints=args.max_checkpoints,
    )


if __name__ == "__main__":
    main()
