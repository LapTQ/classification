import os
import yaml
from typing import List, Tuple, Any
import torch
from PIL import Image
import torchvision.transforms as T
from src.core.model import ClassifyModel
from src.core.augmentations import SquarePad
from tqdm import tqdm

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = "/home/laptq/laptq-fs26-shoplifting-detection/runs/classification/fs26/v4.2dcnn.cluster-CNN-8--cut-l4/weights/best-epoch=03-val_acc=0.396.ckpt"
INPUT_PATHS = [
    "/home/laptq/laptq-fs26-shoplifting-detection/outputs/train_data/classify_rgb/my_format/labels_actions--cluster-CNN-8--cut-l4/cho_tay_vao_tui_quan/val.txt",
    "/home/laptq/laptq-fs26-shoplifting-detection/outputs/train_data/classify_rgb/my_format/labels_actions--cluster-CNN-8--cut-l4/cho_tay_vao_tui_ao/val.txt",
    "/home/laptq/laptq-fs26-shoplifting-detection/outputs/train_data/classify_rgb/my_format/labels_actions--cluster-CNN-8--cut-l4/cho_tay_vao_tui_deo_tren_nguoi/val.txt",
    "/home/laptq/laptq-fs26-shoplifting-detection/outputs/train_data/classify_rgb/my_format/labels_actions--cluster-CNN-8--cut-l4/cho_tay_vao_tui_cam_tren_tay/val.txt",
    # "/home/laptq/laptq-fs26-shoplifting-detection/outputs/train_data/classify_rgb/my_format/labels_actions--cluster-CNN-8--cut-l4/cho_tay_vao_gio_xe_hang/val.txt",
    # "/home/laptq/laptq-fs26-shoplifting-detection/outputs/train_data/classify_rgb/my_format/labels_actions--cluster-CNN-8--cut-l4/cho_tay_vao_ke/val.txt",
    # "/home/laptq/laptq-fs26-shoplifting-detection/outputs/train_data/classify_rgb/my_format/labels_actions--cluster-CNN-8--cut-l4/tay_cam_san_pham/val.txt",
    # "/home/laptq/laptq-fs26-shoplifting-detection/outputs/train_data/classify_rgb/my_format/labels_actions--cluster-CNN-8--cut-l4/tay_khong_cam_san_pham/val.txt",
]  # Path tới file ảnh, file .txt hoặc thư mục
DEVICE = "cuda:5"
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
    model = ClassifyModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    # 3. Transform
    transform = T.Compose(
        [
            SquarePad(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

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
