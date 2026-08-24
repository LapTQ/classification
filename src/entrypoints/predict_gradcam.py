import os
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

from src.core.model import ClassifyModel
from src.entrypoints.bootstrap import create_backbone, create_transform

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = "models/checkpoints/fs26/action_recognition/classification/v22.efficientv2s.for_CNN_8_classes_manually_selected+flux_set_1_2+masked/weights/best-epoch=29-val_f1=0.435.ckpt"
INPUT_PATHS = [
    # "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.manually_selected/cho_tay_vao_tui_cam_tren_tay/train.txt",

    # "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.manually_selected/cho_tay_vao_tui_quan/val.easy.txt",
    # "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.manually_selected/cho_tay_vao_tui_ao/val.easy.txt",
    # "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.manually_selected/cho_tay_vao_tui_deo_tren_nguoi/val.easy.txt",
    # "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.manually_selected/cho_tay_vao_tui_cam_tren_tay/val.easy.txt",

    # "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.gen-flux--set-1/cho_tay_vao_tui_cam_tren_tay/train.txt"
    "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.gen-flux--set-1.masked+grid2/cho_tay_vao_tui_cam_tren_tay/train.txt"
]  # Path tới file ảnh, file .txt hoặc thư mục
DEVICE = "cuda:1"
BATCH_SIZE = 16
OUTPUT_DIR = "data/tmp/predict_2dcnn_action_gradcam/v22"
# =====================================================


def get_norm_stats(cfg: dict) -> tuple[list[float], list[float]]:
    for aug in cfg.get("val_augment", []):
        if "Normalize" in aug:
            return aug["Normalize"]["mean"], aug["Normalize"]["std"]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def predict_gradcam(
    ckpt_path: str,
    input_paths: list[str],
    output_dir: str,
    device: str,
    batch_size: int = 64,
) -> None:
    # 2. Load Model & Config
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes: list[str] = cfg["classes"]
    mean_val, std_val = get_norm_stats(cfg)
    mean_t = torch.tensor(mean_val, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std_val, device=device).view(1, 3, 1, 1)

    backbone = create_backbone(cfg)
    model = ClassifyModel.load_from_checkpoint(ckpt_path, model=backbone).to(device)
    model.eval()

    # Enable gradients for Grad-CAM
    for param in model.parameters():
        param.requires_grad = True

    # 3. Transform
    transform = create_transform(cfg["val_augment"])

    # 4. Collect Images
    image_paths: list[str] = []
    for in_path in input_paths:
        if os.path.isfile(in_path):
            if in_path.endswith(".txt"):
                with open(in_path, "r", encoding="utf-8") as f:
                    image_paths.extend([line.strip() for line in f if line.strip()])
            else:
                image_paths.append(in_path)
        elif os.path.isdir(in_path):
            for root, _, files in os.walk(in_path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(root, f))

    # 5. Predict & Grad-CAM
    print(f"Generating Grad-CAM for {len(image_paths)} images...")
    os.makedirs(output_dir, exist_ok=True)

    # Target layers for Grad-CAM
    if hasattr(model.model, "features"):
        target_layers = [model.model.features[-1]]
    elif hasattr(model.model, "layer4"):  # ResNet
        target_layers = [model.model.layer4[-1]]
    else:
        target_layers = [list(model.model.children())[-2]]

    cam = GradCAM(model=model, target_layers=target_layers)

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

        # Compute predicted classes
        with torch.no_grad():
            logits = model(img_tensors)
            probs = torch.softmax(logits, dim=1)
            scores, pred_idxs = torch.max(probs, dim=1)

        # Generate Grad-CAM for the batch
        grayscale_cams = cam(input_tensor=img_tensors, targets=None)

        # Denormalize input tensors to get the exact RGB image seen by the model
        denorm_imgs = (img_tensors * std_t + mean_t).clamp(0.0, 1.0)
        denorm_np = (denorm_imgs.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)

        for j in range(len(valid_paths)):
            path = valid_paths[j]
            grayscale_cam = grayscale_cams[j, :]
            model_rgb_img = np.float32(denorm_np[j]) / 255.0

            # Generate overlay (cam_image)
            cam_image = show_cam_on_image(model_rgb_img, grayscale_cam, use_rgb=True)

            # Create the standalone heatmap image
            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Combine horizontally: Original (Model Input) | Heatmap | Overlay
            combined = np.hstack((denorm_np[j], heatmap, cam_image))
            result_img = Image.fromarray(combined)

            label_idx: int = int(pred_idxs[j].item())
            label: str = classes[label_idx]
            score: float = float(scores[j].item())

            global_idx: int = i + j
            out_name = f"{global_idx:05d}_{label}_{score:.4f}.jpg"
            out_path = os.path.join(output_dir, out_name)
            result_img.save(out_path)


if __name__ == "__main__":
    predict_gradcam(CKPT_PATH, INPUT_PATHS, OUTPUT_DIR, DEVICE, BATCH_SIZE)
