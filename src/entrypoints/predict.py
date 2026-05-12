import os
import yaml
from typing import List, Tuple, Any
import torch
from PIL import Image
import torchvision.transforms as T
from src.core.model import ActionResNet50
from src.core.augmentations import SquarePad

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = "outputs/train_classification/train_20260506_154000_0/weights/best-epoch=05-val_acc=0.850.ckpt"
INPUT_PATH = "data/test_images"  # Path tới file ảnh, file .txt hoặc thư mục
OUTPUT_PATH = "predictions.txt"
# =====================================================


def predict_model(ckpt_path: str, input_path: str, output_path: str) -> None:
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model & Config
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg["classes"]
    model = ActionResNet50.load_from_checkpoint(ckpt_path).to(device)
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
    if os.path.isfile(input_path):
        if input_path.endswith(".txt"):
            with open(input_path, "r") as f:
                image_paths = [l.strip() for l in f if l.strip()]
        else:
            image_paths = [input_path]
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, f))

    # 5. Predict
    print(f"Predicting {len(image_paths)} images...")
    results = []

    with torch.no_grad():
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                score, pred_idx = torch.max(probs, dim=1)

                label = classes[pred_idx.item()]
                results.append((path, label, score.item()))
            except Exception as e:
                print(f"Error predicting {path}: {e}")

    # 6. Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for p, l, s in results:
            f.write(f"{p}\t{l}\t{s:.4f}\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    predict_model(CKPT_PATH, INPUT_PATH, OUTPUT_PATH)
