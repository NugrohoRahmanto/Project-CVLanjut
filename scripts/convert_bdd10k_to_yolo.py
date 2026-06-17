from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


BDD10K_NAMES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BDD10K detection JSON annotations to Ultralytics YOLO txt labels.")
    parser.add_argument("--data-root", default="", help="BDD10K root with train/val/test folders and labels JSON files. If set, converts available splits and prepares images symlinks.")
    parser.add_argument("--input-json", default="")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--output-label-dir", default="")
    parser.add_argument("--classes", default=",".join(BDD10K_NAMES))
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio taken from the original BDD train split when using --data-root.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic train/val split when using --data-root.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_json_candidates(missing_path: Path) -> list[Path]:
    roots = []
    for root in (missing_path.parent, missing_path.parent.parent, Path("data/bdd10k"), Path(".")):
        if root.exists() and root not in roots:
            roots.append(root)
    patterns = [
        missing_path.name,
        "*labels*images*train*.json",
        "*labels*train*.json",
        "*train*.json",
    ]
    candidates: list[Path] = []
    for root in roots:
        for pattern in patterns:
            for candidate in root.rglob(pattern):
                if candidate.is_file() and candidate not in candidates:
                    candidates.append(candidate)
    return sorted(candidates)


def resolve_input_json(path: Path) -> Path:
    if path.exists():
        return path
    candidates = find_json_candidates(path)
    if len(candidates) == 1:
        print(f"Input JSON not found at {path}. Using discovered JSON: {candidates[0]}")
        return candidates[0]
    message = [f"Input JSON not found: {path}"]
    if candidates:
        message.append("Found possible JSON files. Re-run with one of these paths:")
        message.extend(f"  --input-json {candidate}" for candidate in candidates[:20])
    else:
        message.extend(
            [
                "No candidate train JSON was found under data/bdd10k or the current directory.",
                "Check the extracted Kaggle folder structure with:",
                "  find data/bdd10k -type f -name '*.json' | sort | head -50",
            ]
        )
    raise FileNotFoundError("\n".join(message))


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def find_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in exts)


def clear_directory(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_images(images: list[Path], target_dir: Path) -> None:
    clear_directory(target_dir)
    copied = 0
    for image in images:
        target = target_dir / image.name
        try:
            target.symlink_to(image.resolve())
        except OSError:
            shutil.copy2(image, target)
            copied += 1
    if copied:
        print(f"Symlink failed for {copied} images in {target_dir}; copied files instead.")


def prepare_ultralytics_image_splits(data_root: Path, val_ratio: float, seed: int) -> tuple[int, int, int]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    original_train = data_root / "train"
    original_val = data_root / "val"
    if not original_train.exists():
        raise FileNotFoundError(f"Original BDD train image directory not found: {original_train}")
    if not original_val.exists():
        raise FileNotFoundError(f"Original BDD val image directory not found: {original_val}")

    train_images_all = find_images(original_train)
    final_test_images = find_images(original_val)
    if not train_images_all:
        raise RuntimeError(f"No images found in {original_train}")
    if not final_test_images:
        raise RuntimeError(f"No images found in {original_val}")

    shuffled = train_images_all[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_images = sorted(shuffled[:val_count])
    train_images = sorted(shuffled[val_count:])
    if not train_images:
        raise RuntimeError("Train split is empty after applying --val-ratio.")

    link_images(train_images, data_root / "images" / "train")
    link_images(val_images, data_root / "images" / "val")
    link_images(final_test_images, data_root / "images" / "test")
    return len(train_images), len(val_images), len(final_test_images)


def ensure_data_yaml(data_root: Path) -> None:
    yaml_path = data_root / "bdd10k.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(BDD10K_NAMES))
    yaml_path.write_text(
        f"path: {data_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    print(f"Wrote data yaml: {yaml_path}")


def convert_records(input_json: Path, image_dir: Path, output_dir: Path, class_to_id: dict[str, int], overwrite: bool) -> tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(input_json.read_text(encoding="utf-8"))
    converted = 0
    skipped_missing = 0
    skipped_invalid = 0
    for item in tqdm(records, desc=f"Converting {input_json.name}"):
        image_path = image_dir / item["name"]
        label_path = output_dir / Path(item["name"]).with_suffix(".txt")
        if label_path.exists() and not overwrite:
            continue
        if not image_path.exists():
            skipped_missing += 1
            continue
        width, height = image_size(image_path)
        lines: list[str] = []
        for label in item.get("labels", []):
            category = label.get("category")
            box = label.get("box2d")
            if category not in class_to_id or not box:
                continue
            converted_box = convert_box(box, width, height)
            if converted_box is None:
                skipped_invalid += 1
                continue
            lines.append(f"{class_to_id[category]} " + " ".join(f"{value:.6f}" for value in converted_box))
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted += 1
    return converted, skipped_missing, skipped_invalid


def convert_data_root(args: argparse.Namespace, class_to_id: dict[str, int]) -> None:
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    train_count, val_count, test_count = prepare_ultralytics_image_splits(data_root, args.val_ratio, args.seed)
    print(
        "Prepared Ultralytics image splits: "
        f"train={train_count} from original train, "
        f"val={val_count} from original train, "
        f"test={test_count} from original val"
    )
    ensure_data_yaml(data_root)

    total_converted = 0
    total_missing = 0
    total_invalid = 0

    conversion_plan = [
        ("train", data_root / "labels" / "bdd100k_labels_images_train.json", data_root / "images" / "train", data_root / "labels" / "train"),
        ("val", data_root / "labels" / "bdd100k_labels_images_train.json", data_root / "images" / "val", data_root / "labels" / "val"),
        ("test", data_root / "labels" / "bdd100k_labels_images_val.json", data_root / "images" / "test", data_root / "labels" / "test"),
    ]
    for split, input_json, image_dir, output_dir in conversion_plan:
        if not input_json.exists():
            raise FileNotFoundError(f"Required annotation JSON for {split} not found: {input_json}")
        clear_directory(output_dir)
        converted, missing, invalid = convert_records(input_json, image_dir, output_dir, class_to_id, args.overwrite)
        print(f"{split}: converted={converted} missing_images={missing} invalid_boxes={invalid} labels={output_dir}")
        total_converted += converted
        total_missing += missing
        total_invalid += invalid
    print(f"Total converted images: {total_converted}")
    print(f"Total skipped missing images: {total_missing}")
    print(f"Total skipped invalid boxes: {total_invalid}")


def convert_box(box: dict, width: int, height: int) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(box["x1"]), width))
    y1 = max(0.0, min(float(box["y1"]), height))
    x2 = max(0.0, min(float(box["x2"]), width))
    y2 = max(0.0, min(float(box["y2"]), height))
    if x2 <= x1 or y2 <= y1:
        return None
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return xc, yc, bw, bh


def main() -> None:
    args = parse_args()
    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    if args.data_root:
        convert_data_root(args, class_to_id)
        return

    if not args.input_json or not args.image_dir or not args.output_label_dir:
        raise ValueError("Either use --data-root, or provide --input-json, --image-dir, and --output-label-dir.")

    input_json = resolve_input_json(Path(args.input_json))
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_label_dir)
    converted, skipped_missing, skipped_invalid = convert_records(input_json, image_dir, output_dir, class_to_id, args.overwrite)

    print(f"Converted images: {converted}")
    print(f"Skipped missing images: {skipped_missing}")
    print(f"Skipped invalid boxes: {skipped_invalid}")
    print(f"Labels written to: {output_dir}")


if __name__ == "__main__":
    main()
