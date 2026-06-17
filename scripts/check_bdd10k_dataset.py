from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BDD10K dataset in Ultralytics YOLO format.")
    parser.add_argument("--data-yaml", default="data/bdd10k/bdd10k.yaml")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing data yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def normalize_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise ValueError("names must be a dict or list")


def resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def label_root_from_image_root(image_root: Path) -> Path:
    return Path(str(image_root).replace("/images/", "/labels/"))


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data_yaml)
    config = load_yaml(data_yaml)
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root)
        if not root.exists():
            root = Path(config.get("path", data_yaml.parent))
    names = normalize_names(config.get("names", {}))
    max_class_id = max(names) if names else 9

    class_counts: Counter[int] = Counter()
    missing_labels: list[Path] = []
    empty_labels: list[Path] = []
    invalid_boxes: list[str] = []
    invalid_classes: list[str] = []
    split_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        image_dir = resolve(root, config.get(split))
        if image_dir is None:
            print(f"{split}: not configured")
            continue
        label_dir = label_root_from_image_root(image_dir)
        print(f"{split} image dir: {image_dir} exists={image_dir.exists()}")
        print(f"{split} label dir: {label_dir} exists={label_dir.exists()}")
        images = image_files(image_dir)
        split_counts[split] = len(images)
        for image in images:
            label_path = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            if not label_path.exists():
                missing_labels.append(label_path)
                continue
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels.append(label_path)
            for line_no, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    invalid_boxes.append(f"{label_path}:{line_no}: expected 5 columns")
                    continue
                try:
                    cls = int(float(parts[0]))
                    x, y, w, h = [float(v) for v in parts[1:]]
                except ValueError:
                    invalid_boxes.append(f"{label_path}:{line_no}: non-numeric value")
                    continue
                if cls < 0 or cls > max_class_id:
                    invalid_classes.append(f"{label_path}:{line_no}: class {cls}")
                else:
                    class_counts[cls] += 1
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid_boxes.append(f"{label_path}:{line_no}: invalid xywh {x} {y} {w} {h}")

    print("\nSummary")
    print(f"data yaml: {data_yaml} exists={data_yaml.exists()}")
    for split, count in split_counts.items():
        print(f"{split} images: {count}")
    print(f"missing label files: {len(missing_labels)}")
    print(f"empty label files: {len(empty_labels)}")
    print(f"invalid class rows: {len(invalid_classes)}")
    print(f"invalid bbox rows: {len(invalid_boxes)}")
    print("\nAnnotations per class")
    for class_id, name in names.items():
        print(f"{class_id}: {name}: {class_counts[class_id]}")
    if missing_labels[:10]:
        print("\nFirst missing labels")
        for path in missing_labels[:10]:
            print(path)
    if invalid_classes[:10]:
        print("\nFirst invalid classes")
        for item in invalid_classes[:10]:
            print(item)
    if invalid_boxes[:10]:
        print("\nFirst invalid boxes")
        for item in invalid_boxes[:10]:
            print(item)


if __name__ == "__main__":
    main()
