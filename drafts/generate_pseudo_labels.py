from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm
import yaml

from src.core.model import ClassifyModel
from src.entrypoints.bootstrap import create_backbone


def load_and_resize(
    args: Tuple[str, Tuple[int, int]]
) -> Tuple[str, Optional[Image.Image]]:
    """Load image from disk and resize it to target dimensions.

    Since PIL's resize operation is written in C, it releases the Python GIL,
    enabling true concurrent execution across multiple threads.

    Args:
        args (Tuple[str, Tuple[int, int]]): Tuple of image path and target size
          (width, height).

    Returns:
        Tuple[str, Optional[Image.Image]]: Tuple of image path and PIL Image.
    """
    path, resize_target = args
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(resize_target, Image.BILINEAR)
        return path, img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return path, None


def thread_batch_generator(
    image_paths: List[str],
    resize_target: Tuple[int, int],
    batch_size: int,
    num_threads: int = 16,
) -> Generator[Tuple[List[str], List[Image.Image]], None, None]:
    """Generates batches of images loaded and resized concurrently in worker threads.

    Args:
        image_paths (List[str]): List of image paths to process.
        resize_target (Tuple[int, int]): Target dimensions (width, height).
        batch_size (int): Size of each batch.
        num_threads (int): Number of worker threads.

    Yields:
        Generator[Tuple[List[str], List[Image.Image]], None, None]: Batches of
        paths and PIL Images.
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tasks = [(path, resize_target) for path in batch_paths]
            results = list(executor.map(load_and_resize, tasks))

            valid_paths: List[str] = []
            valid_imgs: List[Image.Image] = []
            for path, img in results:
                if img is not None:
                    valid_paths.append(path)
                    valid_imgs.append(img)

            if valid_imgs:
                yield valid_paths, valid_imgs


def read_image_paths(file_path: str) -> List[str]:
    """Read image paths from a comma-separated text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        List[str]: List of extracted image paths.
    """
    image_paths: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split(",")
            if parts:
                img_path = parts[0].strip()
                if img_path:
                    image_paths.append(img_path)
    return image_paths


def predict_pseudo_labels(
    ckpt_path: str,
    active_inputs: List[Tuple[str, str]],
    output_dir: str,
    device: str,
    batch_size: int = 64,
    class_mapping: Optional[Dict[str, str]] = None,
    selected_classes: Optional[List[str]] = None,
) -> None:
    """Predict and generate pseudo-label txt files using the checkpoint model.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        active_inputs (List[Tuple[str, str]]): List of (input path, prefix)
          tuples.
        output_dir (str): Base directory for saving output files.
        device (str): Device to run inference on.
        batch_size (int): Batch size for inference.
        class_mapping (Optional[Dict[str, str]]): Dictionary mapping old class
          names to new class names before saving. If not mapped, original name
          is preserved.
        selected_classes (Optional[List[str]]): List of allowed classes to use
          after mapping. Unselected classes will be assigned an empty label.
    """
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg["classes"]
    selected_classes_set = (
        set(selected_classes) if selected_classes is not None else None
    )
    backbone = create_backbone(cfg)
    model = ClassifyModel.load_from_checkpoint(ckpt_path, model=backbone, map_location=device)
    model.eval()

    # Extract target size and normalization parameters from the config
    resize_target = (192, 256)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for aug in cfg["val_augment"]:
        name = list(aug.keys())[0]
        if name == "Resize":
            size = aug[name]["size"]
            if isinstance(size, list) and len(size) == 2:
                # YAML size is [height, width], PIL resize needs (width, height)
                resize_target = (size[1], size[0])
        elif name == "Normalize":
            mean = aug[name]["mean"]
            std = aug[name]["std"]

    post_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    for in_path, prefix in active_inputs:
        if not os.path.exists(in_path):
            print(f"Input file not found: {in_path}")
            continue

        print(f"Processing input file: {in_path}")
        image_paths = read_image_paths(in_path)
        num_images = len(image_paths)
        print(f"Loaded {num_images} image paths.")

        results: List[Tuple[str, str]] = []
        num_batches = (num_images + batch_size - 1) // batch_size

        with torch.no_grad():
            batch_gen = thread_batch_generator(
                image_paths=image_paths,
                resize_target=resize_target,
                batch_size=batch_size,
                num_threads=16,
            )
            for batch_paths, batch_imgs in tqdm(
                batch_gen, total=num_batches, desc="Inference Progress"
            ):
                batch_tensors = [post_transform(img) for img in batch_imgs]
                img_tensors = torch.stack(batch_tensors).to(device)

                logits = model(img_tensors)
                probs = torch.softmax(logits, dim=1)
                pred_idxs = torch.argmax(probs, dim=1)

                for path, label_idx in zip(
                    batch_paths, pred_idxs.cpu().tolist()
                ):
                    label = classes[label_idx]
                    if class_mapping is not None and label in class_mapping:
                        label = class_mapping[label]

                    if (
                        selected_classes_set is not None
                        and label not in selected_classes_set
                    ):
                        label = ""

                    results.append((path, label))

        sub_path = in_path
        if in_path.startswith(prefix):
            sub_path = in_path[len(prefix):]
            if sub_path.startswith("/"):
                sub_path = sub_path[1:]

        output_path = os.path.join(output_dir, sub_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for path, label in results:
                f.write(f"{path},{label}\n")
        print(f"Results saved to {output_path}")


def main() -> None:
    """Main execution function to define paths and run labeling."""

    # ================================================================================================
    # GENERATE PERSON-VIEW PSEUDO LABELS 
    # ================================================================================================
    ckpt_path = "models/checkpoints/fs26/person_view/classification/v19.person_view.efficientnetv2m.satudora10k+pa100k/weights/best-epoch=07-val_acc=0.881.ckpt"
    output_dir = "data/tmp/person_view_labels/pseudo/"
    device = "cuda:0"
    batch_size = 128

    # Optional: map old class name to new class name before saving
    class_mapping: Optional[Dict[str, str]] = None

    # Optional: list of classes to use (after mapping). Other classes will be empty
    selected_classes: Optional[List[str]] = None

    print("OK")

    active_inputs: List[Tuple[str, str]] = [
        # ==================== NHÓM A: person attribute data ====================
        # (
        #     "data/processed/fs26/person_attributes/classification_multilabel/CIA_combined_with_geneated_editting.txt",
        #     "data/processed/fs26/person_attributes/classification_multilabel",
        # ),
        # (
        #     "data/processed/fs26/person_attributes/classification_multilabel/CIA_original.txt",
        #     "data/processed/fs26/person_attributes/classification_multilabel",
        # ),
        # (
        #     "data/processed/fs26/person_attributes/classification_multilabel/Satudora_test.txt",
        #     "data/processed/fs26/person_attributes/classification_multilabel",
        # ),
        # ==================== NHÓM B: action recognition data ====================
        # (
        #     "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar.txt",
        #     "data/tmp/action_recognition_labels/gt",
        # ),
        # (
        #     "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_quan/train_resagepar.txt",
        #     "data/tmp/action_recognition_labels/gt",
        # )
        # (
        #     "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar_v2.txt",
        #     "data/tmp/action_recognition_labels/gt",
        # ),
        # (
        #     "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_quan/train_resagepar_v2.txt",
        #     "data/tmp/action_recognition_labels/gt",
        # ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.easy.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.medium.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.easy.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.medium.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.easy.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.medium.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.easy.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.medium.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.easy.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.medium.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.easy.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.medium.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        #
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_gio_xe_hang/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_ke/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_cam_san_pham/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_khong_cam_san_pham/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        #
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-3/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-4/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-5/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-7/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-8/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-9/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-10/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-11/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-12/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        #
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
        (
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/action_recognition_labels/gt",
        ),
    ]


    # # ================================================================================================ 
    # # GENERATE ACTION_RECOGNITION PSEUDO LABELS 
    # # ================================================================================================
    # ckpt_path = "models/checkpoints/fs26/action_recognition/classification/v31.efficientv2s.for_CNN_8_classes_manually_selected+flux_set_1_2_3_4_5_7_8_9_10_11_12+sensenova_set_1_2/weights/best-epoch=31-val_f1=0.566.ckpt"
    # output_dir = "data/tmp/action_recognition_labels/pseudo"
    # device = "cuda:0"
    # batch_size = 128

    # # Optional: map old class name to new class name before saving
    # class_mapping: Optional[Dict[str, str]] = {
    #     "cho_tay_vao_tui_ao": "cho_tay_vao_tui_ao_tui_deo_tui_cam_tay",
    #     "cho_tay_vao_tui_deo_tren_nguoi": "cho_tay_vao_tui_ao_tui_deo_tui_cam_tay",
    #     "cho_tay_vao_tui_cam_tren_tay": "cho_tay_vao_tui_ao_tui_deo_tui_cam_tay",
    # }

    # # Optional: list of classes to use (after mapping). Other classes will be empty
    # selected_classes: Optional[List[str]] = [
    #     "cho_tay_vao_tui_quan",
    #     "cho_tay_vao_tui_ao_tui_deo_tui_cam_tay",
    #     # "cho_tay_vao_gio_xe_hang",
    #     "cho_tay_vao_ke",
    #     "tay_cam_san_pham",
    #     # "tay_khong_cam_san_pham",
    # ]

    # active_inputs: List[Tuple[str, str]] = [
    #     # ==================== NHÓM A: person view data ====================
    #     (
    #         "data/tmp/person_view_labels/gt/back/satudora10k--train.txt",
    #         "data/tmp/person_view_labels/gt/",
    #     ),
    #     (
    #         "data/tmp/person_view_labels/gt/front/satudora10k--train.txt",
    #         "data/tmp/person_view_labels/gt/",
    #     ),
    #     (
    #         "data/tmp/person_view_labels/gt/side/satudora10k--train.txt",
    #         "data/tmp/person_view_labels/gt/",
    #     ),
    #     (
    #         "data/tmp/person_view_labels/gt/back/satudora10k--val.txt",
    #         "data/tmp/person_view_labels/gt/",
    #     ),
    #     (
    #         "data/tmp/person_view_labels/gt/front/satudora10k--val.txt",
    #         "data/tmp/person_view_labels/gt/",
    #     ),
    #     (
    #         "data/tmp/person_view_labels/gt/side/satudora10k--val.txt",
    #         "data/tmp/person_view_labels/gt/",
    #     ),
    #     # ==================== NHÓM B: person attribute data ====================
    #     (
    #         "data/processed/fs26/person_attributes/classification_multilabel/pa100k_train.txt",
    #         "data/processed/fs26/person_attributes/classification_multilabel/"
    #     ),
    #     (
    #         "data/processed/fs26/person_attributes/classification_multilabel/pa100k_val.txt",
    #         "data/processed/fs26/person_attributes/classification_multilabel/"
    #     ),
    #     (
    #         "data/processed/fs26/person_attributes/classification_multilabel/pa100k_test.txt",
    #         "data/processed/fs26/person_attributes/classification_multilabel/"
    #     ),
    #     (
    #         "data/processed/fs26/person_attributes/classification_multilabel/CIA_combined_with_geneated_editting.txt",
    #         "data/processed/fs26/person_attributes/classification_multilabel/"
    #     ),
    #     (
    #         "data/processed/fs26/person_attributes/classification_multilabel/CIA_original.txt",
    #         "data/processed/fs26/person_attributes/classification_multilabel/"
    #     ),
    #     (
    #         "data/processed/fs26/person_attributes/classification_multilabel/Satudora_test.txt",
    #         "data/processed/fs26/person_attributes/classification_multilabel/"
    #     ),
    # ]


    # ===========================================================================================================================================
    predict_pseudo_labels(
        ckpt_path=ckpt_path,
        active_inputs=active_inputs,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        class_mapping=class_mapping,
        selected_classes=selected_classes,
    )


if __name__ == "__main__":
    main()
