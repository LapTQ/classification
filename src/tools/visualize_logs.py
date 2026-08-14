import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def visualize_training(run_dir: str) -> None:
    # Tìm file metrics.csv (thường ở logs/version_0/metrics.csv)
    csv_pattern = os.path.join(run_dir, "logs", "version_*", "metrics.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"Error: Không tìm thấy file metrics.csv trong {run_dir}")
        return

    # Lấy file metrics mới nhất (thường chỉ có 1 version)
    csv_path = max(csv_files, key=os.path.getmtime)
    print(f"Đang đọc dữ liệu từ: {csv_path}")

    df = pd.read_csv(csv_path)

    # Sửa lỗi: Điền các giá trị epoch bị trống (do Lightning log LR ở hàng riêng ngay trước epoch mới)
    # Dùng bfill() để hàng chứa LR mới được gán đúng vào epoch tiếp theo
    df["epoch"] = df["epoch"].bfill().ffill().fillna(0)

    # Gom dữ liệu theo epoch
    metrics_by_epoch = df.groupby("epoch").max().reset_index()

    epochs = metrics_by_epoch["epoch"]

    plt.figure(figsize=(24, 5))

    # 1. Plot Loss
    plt.subplot(1, 4, 1)
    if "train_loss_epoch" in metrics_by_epoch:
        plt.plot(epochs, metrics_by_epoch["train_loss_epoch"], "-o", label="Train Loss")
    elif "train_loss" in metrics_by_epoch:
        plt.plot(epochs, metrics_by_epoch["train_loss"], "-o", label="Train Loss")

    if "val_loss" in metrics_by_epoch:
        plt.plot(epochs, metrics_by_epoch["val_loss"], "-o", label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Plot Accuracy
    plt.subplot(1, 4, 2)
    if "train_acc" in metrics_by_epoch:
        plt.plot(epochs, metrics_by_epoch["train_acc"], "-o", label="Train Acc")
    if "val_acc" in metrics_by_epoch:
        plt.plot(epochs, metrics_by_epoch["val_acc"], "-o", label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # 3. Plot F1 Score
    plt.subplot(1, 4, 3)
    if "train_f1" in metrics_by_epoch:
        plt.plot(epochs, metrics_by_epoch["train_f1"], "-o", label="Train F1")
    if "val_f1" in metrics_by_epoch:
        plt.plot(epochs, metrics_by_epoch["val_f1"], "-o", label="Val F1")
    plt.title("F1 Score Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    # 4. Plot Learning Rate
    plt.subplot(1, 4, 4)
    lr_cols = [c for c in metrics_by_epoch.columns if c.startswith("lr")]
    if lr_cols:
        # Lấy cột LR đầu tiên tìm thấy
        plt.plot(
            epochs,
            metrics_by_epoch[lr_cols[0]],
            "-o",
            color="green",
            label="Learning Rate",
        )
    plt.title("Learning Rate Curve")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    output_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu đồ thị tại: {output_path}")


if __name__ == "__main__":
    # Dán đường dẫn thư mục run của bạn vào đây
    RUN_DIR = "models/checkpoints/fs26/action_recognition/classification/v21.efficientv2s.for_CNN_8_classes_manually_selected+flux_set_1.v2"

    visualize_training(RUN_DIR)
