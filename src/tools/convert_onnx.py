import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

# Đảm bảo thư mục gốc dự án nằm trong sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.model import ClassifyModel
from src.entrypoints.bootstrap import create_backbone

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH: str = "models/checkpoints/fs26/action_recognition/classification/v39.v37+merge_class_2_3/weights/best-epoch=23-val_f1=0.661.ckpt"
OUTPUT_PATH: str = "models/checkpoints/fs26/action_recognition/classification/v39.v37+merge_class_2_3/weights/best-epoch=23-val_f1=0.661.ckpt.onnx"
DUMMY_INPUT: torch.Tensor = torch.randn(1, 3, 224, 224)
INPUT_NAMES: List[str] = ["input"]
OUTPUT_NAMES: List[str] = ["output"]
DYNAMIC_AXES: Dict[str, Dict[int, str]] = {
    "input": {0: "batch"},
    "output": {0: "batch"},
}
OPSET_VERSION: int = 17
# =====================================================


def convert_onnx(
    ckpt_path: str,
    output_path: str,
    dummy_input: torch.Tensor,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    opset_version: int = 17,
) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")

    print(f"Loading model checkpoint from {ckpt_path}...")

    # 1. Load Config (tìm config.yaml ở thư mục cha của checkpoint)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    backbone = create_backbone(cfg)

    # 2. Load model from checkpoint
    model = ClassifyModel.load_from_checkpoint(
        ckpt_path, model=backbone, map_location="cpu"
    )
    model.eval()

    # 3. Ensure parent directory of output_path exists
    parent_dir: str = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent_dir, exist_ok=True)

    # 4. Export model sang ONNX format
    print(f"Exporting model to ONNX format at {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("ONNX conversion completed successfully!")


if __name__ == "__main__":
    convert_onnx(
        ckpt_path=CKPT_PATH,
        output_path=OUTPUT_PATH,
        dummy_input=DUMMY_INPUT,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=DYNAMIC_AXES,
        opset_version=OPSET_VERSION,
    )
