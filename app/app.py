from __future__ import annotations

import hashlib
import logging
import sys
import tempfile
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = "pedestrian,rider,car,truck,bus,train,motorcycle,bicycle,traffic light,traffic sign"
BDD10K_ALL_PROMPTS = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
BDD10K_KNOWN_PROMPTS = ["car", "bus", "truck"]
BDD10K_UNKNOWN_PROMPTS = ["pedestrian", "rider", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
LOW_INFERENCE_CONF = 0.01
MAX_DISPLAY_MODELS = 5
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("streamlit_inference_app")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = setup_logger()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str
    weight: str
    description: str
    scale: str
    scheme: str
    prompt_mode: str = "none"
    needs_prompt: bool = False
    color: tuple[int, int, int] = (0, 170, 70)


SCALE_LABELS = {"s": "Small", "m": "Medium", "l": "Large"}
SCALE_ORDER = {"s": 0, "m": 1, "l": 2}


def infer_scale_from_name(name: str) -> str | None:
    lowered = name.lower()
    patterns = (
        ("s", ("_s_", "_s", "yolov8s", "-s")),
        ("m", ("_m_", "_m", "yolov8m", "-m")),
        ("l", ("_l_", "_l", "yolov8l", "-l")),
    )
    for scale, tokens in patterns:
        if any(token in lowered for token in tokens):
            return scale
    return None


def infer_training_config_from_name(name: str) -> str | None:
    lowered = name.lower()
    if "3class" in lowered:
        return "3class"
    if "10class" in lowered:
        return "10class"
    return None


def latest_checkpoint(root: Path, model_kind: str, training_config: str, scale: str) -> Path | None:
    if not root.exists():
        return None
    candidates: list[Path] = []
    for checkpoint in root.glob("*/weights/best.pt"):
        run_name = checkpoint.parents[1].name
        if infer_training_config_from_name(run_name) == training_config and infer_scale_from_name(run_name) == scale:
            candidates.append(checkpoint)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.parents[1].name)
    LOGGER.info("Detected %s %s %s checkpoint=%s", model_kind, training_config, scale, candidates[-1])
    return candidates[-1]


def latest_checkpoint_from_roots(roots: list[Path], model_kind: str, training_config: str, scale: str) -> Path | None:
    candidates = [checkpoint for root in roots if (checkpoint := latest_checkpoint(root, model_kind, training_config, scale))]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.parents[1].name)
    return candidates[-1]


def is_retention_checkpoint(checkpoint: Path) -> bool:
    return "yoloworld_bdd10k_retention" in checkpoint.parts or "retention" in checkpoint.parents[1].name.lower()


def discover_model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    yoloworld_roots = [ROOT / "runs/yoloworld_bdd10k", ROOT / "runs/yoloworld_bdd10k_retention"]
    for scale in ("s", "m", "l"):
        scale_label = SCALE_LABELS[scale]
        yworld_10 = latest_checkpoint_from_roots(yoloworld_roots, "yoloworld", "10class", scale)
        if yworld_10:
            specs.append(
                ModelSpec(
                    key=f"yoloworld_{scale}_10class",
                    label=f"YOLO-World {scale_label} Fine-Tuned 10-Class",
                    kind="yoloworld",
                    weight=str(yworld_10.relative_to(ROOT)),
                    description="Scheme 1: YOLO-World all-class fine-tuned checkpoint.",
                    scale=scale,
                    scheme="yoloworld_10class",
                    prompt_mode="all",
                    needs_prompt=True,
                    color=(185, 45, 185),
                )
            )

        yworld_3 = latest_checkpoint_from_roots(yoloworld_roots, "yoloworld", "3class", scale)
        if yworld_3:
            is_retention = is_retention_checkpoint(yworld_3)
            specs.append(
                ModelSpec(
                    key=f"yoloworld_{scale}_3class",
                    label=(
                        f"YOLO-World {scale_label} Retention 3-Class"
                        if is_retention
                        else f"YOLO-World {scale_label} Fine-Tuned 3-Class"
                    ),
                    kind="yoloworld",
                    weight=str(yworld_3.relative_to(ROOT)),
                    description=(
                        "Scheme 2: YOLO-World 3-class retention checkpoint with teacher pseudo-labels for open-vocabulary preservation."
                        if is_retention
                        else "Scheme 2: pure YOLO-World 3-class fine-tuned checkpoint; prompts are applied directly to this model."
                    ),
                    scale=scale,
                    scheme="yoloworld_3class",
                    prompt_mode="user",
                    needs_prompt=True,
                    color=(185, 45, 185),
                )
            )

        pretrained_name = f"yolov8{scale}-world.pt"
        specs.append(
            ModelSpec(
                key=f"yoloworld_{scale}_pretrained",
                label=f"YOLO-World {scale_label} Pretrained Zero-Shot",
                kind="yoloworld",
                weight=pretrained_name,
                description="Pure pretrained YOLO-World baseline without BDD10K training. Ultralytics will download the weight if it is not available locally.",
                scale=scale,
                scheme="yoloworld_pretrained",
                prompt_mode="user",
                needs_prompt=True,
                color=(185, 45, 185),
            )
        )

        yolo_10 = latest_checkpoint(ROOT / "runs/yolo_bdd10k", "yolo", "10class", scale)
        if yolo_10:
            specs.append(
                ModelSpec(
                    key=f"yolo_{scale}_10class",
                    label=f"Standard YOLO {scale_label} Fine-Tuned 10-Class",
                    kind="yolo",
                    weight=str(yolo_10.relative_to(ROOT)),
                    description="Scheme 3: closed-set YOLO all-class fine-tuned checkpoint.",
                    scale=scale,
                    scheme="yolo_10class",
                    color=(0, 170, 70),
                )
            )

        yolo_3 = latest_checkpoint(ROOT / "runs/yolo_bdd10k", "yolo", "3class", scale)
        if yolo_3:
            specs.append(
                ModelSpec(
                    key=f"yolo_{scale}_3class",
                    label=f"Standard YOLO {scale_label} Fine-Tuned 3-Class",
                    kind="yolo",
                    weight=str(yolo_3.relative_to(ROOT)),
                    description="Scheme 4: closed-set YOLO known-class fine-tuned checkpoint.",
                    scale=scale,
                    scheme="yolo_3class",
                    color=(0, 170, 70),
                )
            )

    return sorted(specs, key=lambda spec: (SCALE_ORDER.get(spec.scale, 99), spec.scheme))


@st.cache_data(show_spinner=False)
def cached_model_specs() -> list[ModelSpec]:
    return discover_model_specs()


def parse_prompts(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def prompts_for_spec(spec: ModelSpec, user_prompts: list[str], custom_prompts_for_all_yoloworld: bool) -> list[str]:
    if not spec.needs_prompt:
        return []
    if custom_prompts_for_all_yoloworld:
        return user_prompts
    if spec.prompt_mode == "all":
        return BDD10K_ALL_PROMPTS
    if spec.prompt_mode == "known":
        return BDD10K_KNOWN_PROMPTS
    return user_prompts


def resolve_weight(weight: str) -> str:
    path = ROOT / weight
    return str(path) if path.exists() else weight


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@st.cache_resource(show_spinner=False)
def load_model(kind: str, weight: str) -> Any:
    LOGGER.info("Loading model kind=%s weight=%s resolved=%s", kind, weight, resolve_weight(weight))
    if kind == "yoloworld":
        from ultralytics import YOLOWorld

        return YOLOWorld(resolve_weight(weight))

    from ultralytics import YOLO

    return YOLO(resolve_weight(weight))


def set_yoloworld_prompts(model: Any, prompts: list[str], device: str) -> None:
    LOGGER.info("Setting YOLO-World prompts count=%s prompts=%s", len(prompts), prompts)
    try:
        model.set_classes(prompts)
    except RuntimeError as exc:
        if "Expected all tensors to be on the same device" not in str(exc):
            raise
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to("cpu")
        model.set_classes(prompts)
    move_model_to_device(model, device)


def move_model_to_device(model: Any, device: str) -> None:
    if not device or device == "cpu":
        LOGGER.info("Using CPU/no explicit CUDA move for model=%s", type(model).__name__)
        return
    module = getattr(model, "model", None)
    if module is not None and hasattr(module, "to"):
        target = f"cuda:{device}" if str(device).isdigit() else device
        LOGGER.info("Moving model=%s to device=%s", type(model).__name__, target)
        module.to(target)


def save_upload_to_temp(uploaded_file: Any) -> tuple[Path, bytes]:
    data = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name), data


def media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in IMAGE_EXTS:
        return "image"
    return "unknown"


def extract_video_frames(video_path: Path, frame_stride: int, max_frames: int) -> tuple[list[Image.Image], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output_fps = max(1.0, source_fps / max(frame_stride, 1))
    frames: list[Image.Image] = []
    frame_index = 0
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        frame_index += 1
    capture.release()
    if not frames:
        raise RuntimeError("No frames were extracted from the uploaded video.")
    LOGGER.info(
        "Video frames extracted path=%s frames=%s source_fps=%.3f output_fps=%.3f stride=%s max_frames=%s",
        video_path,
        len(frames),
        source_fps,
        output_fps,
        frame_stride,
        max_frames,
    )
    return frames, output_fps


def save_frames_to_temp_images(frames: list[Image.Image]) -> list[Path]:
    frame_dir = Path(tempfile.mkdtemp(prefix="streamlit_video_frames_"))
    paths: list[Path] = []
    for idx, frame in enumerate(frames):
        path = frame_dir / f"frame_{idx:06d}.jpg"
        frame.save(path, quality=95)
        paths.append(path)
    return paths


def result_to_detections(result: Any, names: dict[int, str], color: tuple[int, int, int]) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    detections: list[dict[str, Any]] = []
    for box, conf, cls in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
        class_id = int(cls)
        if isinstance(names, dict):
            label = names.get(class_id, class_id)
        elif isinstance(names, list) and 0 <= class_id < len(names):
            label = names[class_id]
        else:
            label = class_id
        detections.append(
            {
                "box": [float(v) for v in box],
                "confidence": float(conf),
                "class_id": class_id,
                "label": str(label),
                "color": color,
            }
        )
    return detections


def predict_detections_for_model(
    model: Any,
    spec: ModelSpec,
    source_paths: list[Path],
    imgsz: int,
    iou: float,
    device: str,
    color: tuple[int, int, int],
) -> list[list[dict[str, Any]]]:
    LOGGER.info("Predict start key=%s sources=%s", spec.key, len(source_paths))
    results = model.predict(
        source=[str(path) for path in source_paths],
        conf=LOW_INFERENCE_CONF,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=False,
        verbose=False,
    )
    frame_detections: list[list[dict[str, Any]]] = []
    for result in results:
        names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}
        frame_detections.append(result_to_detections(result, names, color))
    return frame_detections


def log_detection_summary(spec: ModelSpec, frame_detections: list[list[dict[str, Any]]]) -> None:
    detections = [det for frame in frame_detections for det in frame]
    top_labels = sorted(
        ((det["label"], det["confidence"]) for det in detections),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    LOGGER.info(
        "Predict done key=%s frames=%s raw_boxes=%s top=%s",
        spec.key,
        len(frame_detections),
        len(detections),
        [(label, round(conf, 4)) for label, conf in top_labels],
    )


def run_inference(
    source_paths: list[Path],
    selected_specs: list[ModelSpec],
    prompts: list[str],
    custom_prompts_for_all_yoloworld: bool,
    imgsz: int,
    iou: float,
    device: str,
) -> dict[str, list[list[dict[str, Any]]]]:
    outputs: dict[str, list[list[dict[str, Any]]]] = {}
    progress = st.progress(0, text="Running inference...")
    LOGGER.info(
        "Inference request sources=%s models=%s prompt_count=%s imgsz=%s iou=%s device=%s low_conf=%s",
        len(source_paths),
        [spec.label for spec in selected_specs],
        len(prompts),
        imgsz,
        iou,
        device,
        LOW_INFERENCE_CONF,
    )

    for index, spec in enumerate(selected_specs, start=1):
        progress.progress((index - 1) / len(selected_specs), text=f"Loading {spec.label}")
        LOGGER.info("Model start key=%s label=%s kind=%s weight=%s", spec.key, spec.label, spec.kind, spec.weight)
        model = load_model(spec.kind, spec.weight)
        if spec.needs_prompt:
            active_prompts = prompts_for_spec(spec, prompts, custom_prompts_for_all_yoloworld)
            LOGGER.info("Active prompts for key=%s mode=%s prompts=%s", spec.key, spec.prompt_mode, active_prompts)
            set_yoloworld_prompts(model, active_prompts, device)
        else:
            move_model_to_device(model, device)

        progress.progress((index - 1) / len(selected_specs), text=f"Inference: {spec.label}")
        frame_detections = predict_detections_for_model(
            model=model,
            spec=spec,
            source_paths=source_paths,
            imgsz=imgsz,
            iou=iou,
            device=device,
            color=spec.color,
        )
        outputs[spec.key] = frame_detections
        log_detection_summary(spec, frame_detections)

    progress.progress(1.0, text="Inference complete")
    progress.empty()
    LOGGER.info("Inference request complete sources=%s", len(source_paths))
    return outputs


def draw_detections(image: Image.Image, detections: list[dict[str, Any]], threshold: float) -> Image.Image:
    canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]
    for det in detections:
        if det["confidence"] < threshold:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        color = tuple(int(v) for v in det["color"])
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        y_text = max(y1, th + baseline + 5)
        cv2.rectangle(canvas, (x1, y_text - th - baseline - 5), (min(x1 + tw + 8, width - 1), y_text + baseline), color, -1)
        cv2.putText(canvas, label, (x1 + 4, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def render_video(frames: list[Image.Image], frame_detections: list[list[dict[str, Any]]], threshold: float, fps: float) -> tuple[bytes, bytes]:
    output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)
    first = frames[0].convert("RGB")
    width, height = first.size
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create annotated video writer.")
    for idx, frame in enumerate(frames):
        detections = frame_detections[idx] if idx < len(frame_detections) else []
        rendered = draw_detections(frame, detections, threshold)
        writer.write(cv2.cvtColor(np.array(rendered.convert("RGB")), cv2.COLOR_RGB2BGR))
    writer.release()
    mp4_bytes = output_path.read_bytes()

    gif_buffer = BytesIO()
    gif_frames = []
    gif_max_frames = min(len(frames), 80)
    for idx, frame in enumerate(frames[:gif_max_frames]):
        detections = frame_detections[idx] if idx < len(frame_detections) else []
        rendered = draw_detections(frame, detections, threshold)
        if rendered.width > 960:
            ratio = 960 / rendered.width
            rendered = rendered.resize((960, int(rendered.height * ratio)))
        gif_frames.append(rendered.convert("P", palette=Image.ADAPTIVE))
    duration_ms = int(1000 / max(fps, 1.0))
    gif_frames[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return mp4_bytes, gif_buffer.getvalue()


def selected_model_specs(selected_keys: list[str], available_specs: list[ModelSpec]) -> list[ModelSpec]:
    by_key = {spec.key: spec for spec in available_specs}
    return [by_key[key] for key in selected_keys if key in by_key]


def render_model_controls(available_specs: list[ModelSpec]) -> tuple[list[str], str]:
    st.subheader("Model Selection")
    scale_choice = st.radio(
        "Model scale",
        options=["l", "m", "s", "all"],
        format_func=lambda value: {"s": "Small", "m": "Medium", "l": "Large", "all": "All detected"}.get(value, value),
        horizontal=True,
    )
    specs = [spec for spec in available_specs if scale_choice == "all" or spec.scale == scale_choice]
    if not specs:
        st.warning("No checkpoints detected for the selected scale.")

    selected: list[str] = []
    default_scale = "l" if scale_choice == "all" else scale_choice
    default_keys = {f"yoloworld_{default_scale}_pretrained", f"yoloworld_{default_scale}_3class", f"yolo_{default_scale}_3class"}
    for spec in specs:
        default = spec.key in default_keys
        checked = st.checkbox(spec.label, value=default, help=spec.description, key=f"select_{spec.key}")
        if checked:
            selected.append(spec.key)

    if len(selected) > MAX_DISPLAY_MODELS:
        st.error(f"Select at most {MAX_DISPLAY_MODELS} models.")
    return selected, scale_choice


def main() -> None:
    st.set_page_config(page_title="Autonomous Driving Detector Comparison", layout="wide")
    st.title("Open-Vocabulary Hazard Perception for Autonomous Driving")
    st.caption("Compare detected YOLO-World and Standard YOLO Small/Medium/Large models on the same input.")
    available_specs = cached_model_specs()

    with st.sidebar:
        if st.button("Refresh detected runs"):
            cached_model_specs.clear()
            st.rerun()
        selected_keys, scale_choice = render_model_controls(available_specs)
        threshold = st.slider("Displayed confidence threshold", 0.01, 0.95, 0.25, 0.01)
        imgsz = st.select_slider("Image size", options=[320, 480, 640, 800, 960, 1280], value=640)
        iou = st.slider("NMS IoU threshold", 0.10, 0.95, 0.70, 0.01)
        device = st.text_input("Device", value="0", help="Use 0 for CUDA GPU 0, cpu for CPU.")
        custom_prompts_for_all_yoloworld = st.checkbox(
            "Use custom prompt for every YOLO-World model",
            value=False,
            help=(
                "Default off keeps prompts aligned with each scheme: 10-class fine-tuned uses all BDD10K classes; "
                "3-class fine-tuned and pretrained YOLO-World use the custom prompt directly."
            ),
        )
        st.divider()
        st.subheader("Video")
        frame_stride = st.number_input("Frame stride", min_value=1, max_value=120, value=5, step=1)
        max_video_frames = st.number_input("Max processed frames", min_value=1, max_value=600, value=120, step=1)

    uploaded_file = st.file_uploader(
        "Upload road-scene image or video",
        type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "mov", "avi", "mkv", "webm"],
    )
    prompt_text = st.text_input(
        "Custom YOLO-World prompts, separated by comma",
        value=DEFAULT_PROMPTS,
        help=(
            "Minimum one prompt. By default this is used by pretrained YOLO-World. "
            "Fine-tuned YOLO-World uses scheme prompts unless the sidebar custom-prompt option is enabled."
        ),
    )

    selected_specs = selected_model_specs(selected_keys[:MAX_DISPLAY_MODELS], available_specs)
    prompts = parse_prompts(prompt_text)
    selected_requires_prompt = any(spec.needs_prompt for spec in selected_specs)

    run_disabled = (
        uploaded_file is None
        or not selected_specs
        or len(selected_keys) > MAX_DISPLAY_MODELS
        or (selected_requires_prompt and not prompts)
    )

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_clicked = st.button("Run Inference", disabled=run_disabled, type="primary")
    with col_info:
        if selected_requires_prompt and not prompts:
            st.warning("YOLO-World needs at least one comma-separated prompt before inference.")
        elif not selected_specs:
            st.info("Select one or more models.")
        elif len(selected_keys) > MAX_DISPLAY_MODELS:
            st.warning(f"Only {MAX_DISPLAY_MODELS} models can be displayed together.")

    if uploaded_file is None:
        st.info("Upload an image to start.")
        return

    media_path, media_bytes = save_upload_to_temp(uploaded_file)
    current_media_kind = media_kind(media_path)
    media_hash = file_digest(media_bytes)
    if current_media_kind == "unknown":
        st.error("Unsupported upload type.")
        return

    st.subheader("Input")
    if current_media_kind == "image":
        image = Image.open(media_path).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        source_paths = [media_path]
        video_frames: list[Image.Image] = []
        video_fps = 0.0
    else:
        st.video(media_bytes, format="video/mp4")
        video_frames, video_fps = extract_video_frames(media_path, int(frame_stride), int(max_video_frames))
        source_paths = save_frames_to_temp_images(video_frames)
        st.caption(f"Video mode: processing {len(source_paths)} sampled frames at output FPS {video_fps:.2f}.")

    inference_signature = {
        "media_hash": media_hash,
        "media_kind": current_media_kind,
        "selected_keys": selected_keys[:MAX_DISPLAY_MODELS],
        "scale_choice": scale_choice,
        "prompts": prompts,
        "custom_prompts_for_all_yoloworld": custom_prompts_for_all_yoloworld,
        "imgsz": imgsz,
        "iou": iou,
        "device": device,
        "frame_stride": int(frame_stride) if current_media_kind == "video" else None,
        "max_video_frames": int(max_video_frames) if current_media_kind == "video" else None,
    }

    if run_clicked:
        LOGGER.info(
            "Run button clicked upload=%s media_kind=%s selected=%s threshold=%s prompts=%s",
            uploaded_file.name,
            current_media_kind,
            selected_keys[:MAX_DISPLAY_MODELS],
            threshold,
            prompts,
        )
        with st.spinner("Running selected models..."):
            st.session_state["detections"] = run_inference(
                source_paths=source_paths,
                selected_specs=selected_specs,
                prompts=prompts,
                custom_prompts_for_all_yoloworld=custom_prompts_for_all_yoloworld,
                imgsz=imgsz,
                iou=iou,
                device=device,
            )
            st.session_state["inference_signature"] = inference_signature
            st.session_state["media_kind"] = current_media_kind
            st.session_state["media_bytes"] = media_bytes
            st.session_state["video_frames"] = video_frames
            st.session_state["video_fps"] = video_fps

    detections_by_model = st.session_state.get("detections")
    if not detections_by_model:
        return

    st.subheader("Inference Results")
    if st.session_state.get("inference_signature") != inference_signature:
        st.warning("Settings changed after the last inference. Click Run Inference to refresh model outputs. The threshold slider still filters the previous outputs.")

    stored_media_kind = st.session_state.get("media_kind", current_media_kind)
    columns = st.columns(min(len(selected_specs), 3))
    for idx, spec in enumerate(selected_specs):
        per_source_detections = detections_by_model.get(spec.key, [])
        visible_count = sum(1 for frame in per_source_detections for det in frame if det["confidence"] >= threshold)
        with columns[idx % len(columns)]:
            if stored_media_kind == "image":
                image = Image.open(media_path).convert("RGB")
                rendered = draw_detections(image, per_source_detections[0] if per_source_detections else [], threshold)
                st.image(rendered, caption=f"{spec.label} | boxes: {visible_count}", use_container_width=True)
            else:
                frames = st.session_state.get("video_frames") or video_frames
                fps = float(st.session_state.get("video_fps") or video_fps or 10.0)
                video_bytes, gif_bytes = render_video(frames, per_source_detections, threshold, fps)
                st.video(video_bytes, format="video/mp4")
                st.caption(f"{spec.label} | boxes: {visible_count}")
                with st.expander("GIF fallback / download"):
                    st.image(gif_bytes, caption="Fallback preview if MP4 playback is unavailable.")
                    st.download_button(
                        "Download annotated MP4",
                        data=video_bytes,
                        file_name=f"{spec.key}_annotated.mp4",
                        mime="video/mp4",
                        key=f"download_{spec.key}",
                    )

    with st.expander("Detection table"):
        rows = []
        for spec in selected_specs:
            for frame_idx, frame_detections in enumerate(detections_by_model.get(spec.key, [])):
                for det in frame_detections:
                    if det["confidence"] >= threshold:
                        rows.append(
                            {
                                "model": spec.label,
                                "frame": frame_idx,
                                "label": det["label"],
                                "confidence": round(det["confidence"], 4),
                                "box_xyxy": [round(v, 1) for v in det["box"]],
                            }
                        )
        st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
