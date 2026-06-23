# Streamlit Inference App

Interactive dashboard for comparing YOLO-based detectors on one uploaded road-scene image or video.

## Run

```bash
bash run_app.sh
```

Custom host or port:

```bash
HOST=0.0.0.0 PORT=8502 bash run_app.sh
```

If Streamlit is not installed yet:

```bash
uv add streamlit
```

or:

```bash
.venv/bin/pip install streamlit
```

## Included Model Panels

The app automatically scans these folders:

```text
runs/yoloworld_bdd10k/*/weights/best.pt
runs/yolo_bdd10k/*/weights/best.pt
```

It detects Small, Medium, and Large checkpoints from run names and exposes a scale switch in the sidebar:

```text
Small
Medium
Large
All detected
```

Detected schemes:

```text
YOLO-World Fine-Tuned 10-Class
YOLO-World Fine-Tuned 3-Class
YOLO-World Pretrained Zero-Shot
Standard YOLO Fine-Tuned 10-Class
Standard YOLO Fine-Tuned 3-Class
```

YOLO-World models require at least one comma-separated prompt before inference.

The confidence slider filters already-computed detections, so changing the threshold is immediate after the first inference run.

## Video Inference

The app accepts `mp4`, `mov`, `avi`, `mkv`, and `webm`.

Video inference samples frames before running the selected models. Use the sidebar controls to limit compute:

```text
Frame stride         : process every N-th frame
Max processed frames : hard cap for processed frames
```

After inference, the threshold slider re-renders the annotated video from stored detections without running the models again.

## Console Logger

The terminal running `bash run_app.sh` logs:

```text
selected models
YOLO-World prompts
model loading
device movement
inference start
raw bbox count
top detected labels per model
```
