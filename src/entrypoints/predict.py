import os
import yaml
from typing import List, Tuple, Any
import torch
from PIL import Image
import torchvision.transforms as T
from src.core.model import ClassifyModel
from src.entrypoints.bootstrap import create_backbone, create_transform
from tqdm import tqdm

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = "models/checkpoints/fs26/action_recognition/classification/v21.efficientv2s.for_CNN_8_classes_manually_selected+flux_set_1.val_easy_medium/weights/best-epoch=34-val_f1=0.434.ckpt"
INPUT_PATHS = [
    "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.manually_selected/cho_tay_vao_tui_quan/val.easy.txt",
    "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.manually_selected/cho_tay_vao_tui_ao/val.easy.txt",
    "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.manually_selected/cho_tay_vao_tui_deo_tren_nguoi/val.easy.txt",
    "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.manually_selected/cho_tay_vao_tui_cam_tren_tay/val.easy.txt",
]  # Path tới file ảnh, file .txt hoặc thư mục
DEVICE = "cuda:2"
BATCH_SIZE = 64
OUTPUT_PATH = "/home/laptq/laptq-fs26-shoplifting-detection/outputs/trivials/predict_2dcnn_action/predictions.txt"
# =====================================================


def predict_model(ckpt_path: str, input_paths: List[str], output_path: str, device: str, batch_size: int = 64) -> None:
    # 2. Load Model & Config
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg["classes"]
    backbone = create_backbone(cfg)
    model = ClassifyModel.load_from_checkpoint(ckpt_path, model=backbone).to(device)
    model.eval()

    # 3. Transform
    transform = create_transform(cfg["val_augment"])

    # 4. Collect Images
    image_paths = []
    for in_path in input_paths:
        if os.path.isfile(in_path):
            if in_path.endswith(".txt"):
                with open(in_path, "r") as f:
                    image_paths.extend([l.strip() for l in f if l.strip()])
            else:
                image_paths.append(in_path)
        elif os.path.isdir(in_path):
            for root, _, files in os.walk(in_path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(root, f))

    # 5. Predict
    print(f"Predicting {len(image_paths)} images...")
    results = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = []
            valid_paths = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = transform(img)
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            if not batch_tensors:
                continue

            img_tensors = torch.stack(batch_tensors).to(device)
            logits = model(img_tensors)
            probs = torch.softmax(logits, dim=1)
            scores, pred_idxs = torch.max(probs, dim=1)

            for path, label_idx, score in zip(valid_paths, pred_idxs.cpu().tolist(), scores.cpu().tolist()):
                label = classes[label_idx]
                results.append((path, label, score))

    # 6. Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p, l, s in results:
            f.write(f"{p}\t{l}\t{s:.4f}\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    predict_model(CKPT_PATH, INPUT_PATHS, OUTPUT_PATH, DEVICE, BATCH_SIZE)
