import os
from datetime import datetime
from typing import List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T


def get_run_dir(run_dir: str) -> str:
    """
    Tạo thư mục run. Nếu đã tồn tại, tự động thêm hậu tố _1, _2, ... _n
    dựa trên hậu tố lớn nhất hiện có.
    """
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    parent_dir = os.path.dirname(os.path.abspath(run_dir))
    base_name = os.path.basename(run_dir)

    import re

    pattern = re.compile(rf"^{re.escape(base_name)}(?:_(\d+))?$")
    max_suffix = 0
    found_base = False

    if os.path.exists(parent_dir):
        for d in os.listdir(parent_dir):
            match = pattern.match(d)
            if match:
                suffix = match.group(1)
                if suffix is None:
                    found_base = True
                else:
                    max_suffix = max(max_suffix, int(suffix))

    if not found_base and max_suffix == 0:
        # Trường hợp hy hữu: run_dir không tồn tại nhưng parent_dir có các thư mục trùng pattern
        # Nhưng ở đây run_dir đã exists nên ít nhất found_base sẽ là True
        pass

    new_suffix = max_suffix + 1
    new_run_dir = os.path.join(parent_dir, f"{base_name}_{new_suffix}")
    os.makedirs(new_run_dir, exist_ok=True)
    return new_run_dir


def visualize_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
    classes: List[str],
    output_path: str,
    title: str = "Batch",
) -> None:
    imgs, labels, _ = batch

    # Số lượng ảnh muốn hiển thị (max 16)
    n = min(len(imgs), 16)
    imgs = imgs[:n]
    labels = labels[:n]

    # Unnormalize (ImageNet standards)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    plt.figure(figsize=(12, 12))
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        # Denormalize
        img = imgs[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        plt.imshow(img)
        class_name = (
            classes[labels[i]] if labels[i] < len(classes) else f"ID:{labels[i]}"
        )
        plt.title(class_name, fontsize=8)
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization to {output_path}")
