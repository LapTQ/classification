import os
from pathlib import Path
from typing import List, Tuple


def get_class_directories(raw_dir: Path) -> List[str]:
    classes: List[str] = [
        d.name for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    return sorted(classes)


def get_image_files(class_dir: Path, extensions: Tuple[str, ...]) -> List[str]:
    files: List[str] = [
        f.name for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions
    ]
    return sorted(files)


def process_dataset(
    raw_dataset_dir: str,
    output_processed_dir: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> None:
    raw_dir_path: Path = Path(raw_dataset_dir)
    out_dir_path: Path = Path(output_processed_dir)

    classes: List[str] = get_class_directories(raw_dir_path)

    for class_name in classes:
        class_raw_dir: Path = raw_dir_path / class_name
        class_out_dir: Path = out_dir_path / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        image_files: List[str] = get_image_files(class_raw_dir, extensions)

        relative_paths: List[str] = [
            f"{raw_dataset_dir}/{class_name}/{img_name}" for img_name in image_files
        ]

        output_txt_file: Path = class_out_dir / "train.txt"
        with open(output_txt_file, "w", encoding="utf-8") as f:
            for path_str in relative_paths:
                f.write(f"{path_str}\n")

        print(f"Created {output_txt_file} with {len(relative_paths)} image paths.")


def main() -> None:
    raw_dir: str = "data/raw/fs26/action_recognition/gen-flux--set-1"
    output_dir: str = (
        "data/processed/fs26/action_recognition/classification/action.for_CNN.8_classes.gen-flux--set-1"
    )
    process_dataset(raw_dataset_dir=raw_dir, output_processed_dir=output_dir)


if __name__ == "__main__":
    main()
