from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-label-dir", required=True)
    parser.add_argument("--classes", default=",".join(BDD10K_NAMES))
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
    input_json = resolve_input_json(Path(args.input_json))
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_label_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    records = json.loads(input_json.read_text(encoding="utf-8"))
    converted = 0
    skipped_missing = 0
    skipped_invalid = 0
    for item in tqdm(records, desc="Converting BDD10K"):
        image_path = image_dir / item["name"]
        label_path = output_dir / Path(item["name"]).with_suffix(".txt")
        if label_path.exists() and not args.overwrite:
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

    print(f"Converted images: {converted}")
    print(f"Skipped missing images: {skipped_missing}")
    print(f"Skipped invalid boxes: {skipped_invalid}")
    print(f"Labels written to: {output_dir}")


if __name__ == "__main__":
    main()
