# YOLO-World BDD10K Pipeline

Pipeline ini menyiapkan training, evaluation, prediction, dan export YOLO-World untuk BDD10K menggunakan library `ultralytics`. Notebook dipakai untuk workflow interaktif dan dokumentasi eksperimen, sedangkan `scripts/run_train_yoloworld_bdd10k.py` adalah runner utama untuk terminal atau `nohup`.

## Quick Start Dari Nol

### 1. Clone Repository

```bash
git clone <url-repo-anda>
cd cvlanjut-project
```

### 2. Install uv

Jika `uv` belum terpasang:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Tutup dan buka terminal lagi, atau jalankan instruksi PATH yang muncul dari installer. Cek versi:

```bash
uv --version
```

### 3. Install Dependency

Project ini memakai Python `>=3.10,<3.11`.

```bash
uv sync
```

Setelah selesai, semua command runtime training memakai script bash:

```bash
bash run_train_yoloworld_bdd10k.sh
```

### 4. Download Dataset BDD10K

Download zip dari Kaggle, unzip ke folder tujuan, lalu hapus zip otomatis:

```bash
mkdir -p data/bdd10k
curl -L -o ./bdd10k.zip \
  https://www.kaggle.com/api/v1/datasets/download/aadityadamle/bdd10k
unzip -q ./bdd10k.zip -d data/bdd10k
rm -f ./bdd10k.zip
```

Jika Kaggle meminta autentikasi, pastikan credential Kaggle sudah tersedia di `~/.kaggle/kaggle.json` atau environment Kaggle Anda sudah login.

Dataset akhir yang diharapkan oleh pipeline:

```text
data/bdd10k/
  images/train|val|test/
  labels/bdd100k_labels_images_train.json
  labels/bdd100k_labels_images_val.json
  bdd10k.yaml
```

Jika hasil unzip memiliki struktur nested, pindahkan isi foldernya sampai sesuai struktur di atas. Pipeline akan mencoba auto-convert JSON BDD ke label YOLO saat training pertama dijalankan.

### 5. Smoke Test GPU

```bash
bash run_train_yoloworld_bdd10k.sh \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model yolov8s-world.pt \
  --output-dir runs/yoloworld_bdd10k \
  --experiment-name smoke_yoloworld_bdd10k \
  --timestamp-output \
  --epochs 1 \
  --batch-size 2 \
  --imgsz 320 \
  --workers 0 \
  --device 0 \
  --amp \
  --patience 1
```

Command langsung kembali ke prompt karena proses berjalan background dengan `nohup`. Pantau log:

```bash
tail -f training.log
```

## Struktur Pipeline Berbasis Prompt

### Fase A: Data Preparation & Fine-Tuning

Sumber data: BDD100K/BDD10K subset.

Prompt latih default:

```python
["car", "bus", "truck"]
```

Runner membuat dataset sementara berisi hanya anotasi 3 kelas target di:

```text
runs/yoloworld_bdd10k/<experiment-name>/dataset_known/
```

Class id di-remap menjadi 0..2. Modul text/CLIP/prompt encoder dibekukan secara best-effort dengan default `--freeze-text-encoder`, lalu training mengoptimalkan detector/head untuk bbox 3 kelas target.

### Fase B: Inference Known + Zero-Shot Unknown

Prompt known dari model fine-tuned:

```python
["car", "bus", "truck"]
```

Prompt unknown dari model YOLO-World pretrained zero-shot:

```python
["pedestrian", "rider", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
```

Secara default sample evaluation memakai dua branch:

1. Checkpoint fine-tuned untuk mendeteksi known class `car`, `bus`, `truck`.
2. Weight pretrained `yolov8s-world.pt` melalui `--zero-shot-unknown-model` untuk mendeteksi prompt unknown.

Ini penting karena checkpoint yang sudah fine-tune pada 3 kelas sering menjadi terlalu spesifik dan tidak lagi memilih prompt unknown dengan baik. Branch zero-shot menjaga kemampuan open-vocabulary dari YOLO-World pretrained.

### Fase C: Post-Processing Unknown

Jika indeks prediksi berada pada `0..2`, bounding box disimpan dengan label spesifiknya. Jika indeks prediksi berada pada `3..9`, bounding box tetap disimpan tetapi label final diganti menjadi:

```text
Unknown Object
```

Hasil post-processing inference ditulis ke:

```text
runs/yoloworld_bdd10k/<experiment-name>/predictions/prediction_postprocessed.json
```

Setelah training/eval, pipeline juga menyimpan contoh visual performa ke:

```text
runs/yoloworld_bdd10k/<experiment-name>/eval_samples/
```

Gambar evaluasi dibuat dua sisi: kiri adalah ground truth, kanan adalah hasil inference. Bounding box ground truth digambar biru-oranye. Bounding box inference untuk known class dari model fine-tuned digambar hijau. Bounding box inference untuk branch zero-shot unknown digambar merah dan label akhirnya ditulis sebagai `Unknown Object`.

## Struktur Project

```text
data/bdd10k/
  images/train|val|test/
  labels/train|val|test/
  bdd10k.yaml
notebooks/train_yoloworld_bdd10k.ipynb
scripts/convert_bdd10k_to_yolo.py
scripts/check_bdd10k_dataset.py
scripts/run_train_yoloworld_bdd10k.py
scripts/yoloworld_bdd10k_pipeline.py
runs/yoloworld_bdd10k/<experiment-name>/
training.log
```

Semua output eksperimen masuk ke:

```text
runs/yoloworld_bdd10k/<experiment-name>/
```

Isi pentingnya:

```text
train.log
args.json
config_used.yaml
run_summary.json
weights/
results.csv
metrics_summary.json
evaluation.json
eval_samples/
predictions/
```

File `training.log` di root project selalu ditimpa setiap kali runner dijalankan.

## Dataset BDD10K

Default class names di `data/bdd10k/bdd10k.yaml`:

```yaml
names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
```

Struktur YOLO yang diharapkan:

```text
data/bdd10k/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  bdd10k.yaml
```

Jika gambar masih berada langsung di `data/bdd10k/train`, `data/bdd10k/val`, dan `data/bdd10k/test`, pindahkan atau symlink ke `data/bdd10k/images/<split>` agar kompatibel dengan Ultralytics.

## Install Dependency

```bash
uv sync
```

Project ini memiliki `pyproject.toml` dengan Python `>=3.10,<3.11`. Jika `uv` belum menemukan Python 3.10, siapkan interpreter 3.10 dulu lalu jalankan:

```bash
uv sync
```

## Convert Dataset

```bash
.venv/bin/python scripts/convert_bdd10k_to_yolo.py \
  --input-json data/bdd10k/labels/bdd100k_labels_images_train.json \
  --image-dir data/bdd10k/images/train \
  --output-label-dir data/bdd10k/labels/train
```

Ulangi untuk split val/test jika JSON tersedia. Converter mempertahankan mapping 10 kelas BDD asli. Filtering ke 3 prompt latih dilakukan oleh runner di dalam folder eksperimen agar `data/bdd10k/bdd10k.yaml` tetap konsisten.

Jika `data/bdd10k/labels/train/*.txt` belum ada, runner akan mencoba auto-convert dari `data/bdd10k/labels/bdd100k_labels_images_train.json` untuk gambar yang namanya cocok dengan `data/bdd10k/images/train`.

Pada dataset lokal saat ini, JSON val tidak cocok dengan nama gambar di `data/bdd10k/images/val`. Jika filtered validation annotation kosong, runner otomatis memakai `images/train` sebagai split val agar Ultralytics YOLO-World tidak gagal saat membangun dataloader.

## Check Dataset

```bash
.venv/bin/python scripts/check_bdd10k_dataset.py \
  --data-yaml data/bdd10k/bdd10k.yaml
```

Checker melaporkan folder image/label, jumlah image train/val/test, missing label, empty label, invalid class id, invalid bbox, dan distribusi annotation per class.

## Known vs Unknown

Gunakan `--known-classes` untuk memilih kelas yang dipelajari secara supervised. Contoh:

Default `--known-classes` sekarang adalah `car,bus,truck`.

Runner akan membuat dataset sementara di folder eksperimen:

```text
runs/yoloworld_bdd10k/<experiment-name>/dataset_known/
```

Label YOLO hanya menyimpan class known dan class id di-remap menjadi 0..N-1. Kelas lain tidak dipakai sebagai target training dan dicatat sebagai unknown/ignored. Untuk inference open-vocabulary, prompt known dan default `--unknown-prompts pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign` dikirim ke `model.set_classes(...)` jika API Ultralytics tersedia.

Catatan: ini tidak membuat YOLO-World belajar class supervised bernama `unknown`; annotation kelas lain disembunyikan saat training agar model fine-tune pada known classes, lalu kemampuan open-vocabulary pretrained YOLO-World dipakai saat predict/eval.

## Training dari Notebook

Buka:

```text
notebooks/train_yoloworld_bdd10k.ipynb
```

Notebook memanggil logic yang sama dari `scripts/yoloworld_bdd10k_pipeline.py`.

Eksekusi notebook dari terminal:

```bash
jupyter nbconvert --to notebook --execute notebooks/train_yoloworld_bdd10k.ipynb \
  --output executed_train_yoloworld_bdd10k.ipynb
```

Alternatif papermill:

```bash
papermill notebooks/train_yoloworld_bdd10k.ipynb notebooks/executed_train_yoloworld_bdd10k.ipynb
```

## Training Tanpa Flag GPU

```bash
bash run_train_yoloworld_bdd10k.sh
```

Command ini langsung selesai di terminal dan training berjalan di background dengan `nohup`. Default pipeline:

```text
data-yaml: data/bdd10k/bdd10k.yaml
model: yolov8s-world.pt
output-dir: runs/yoloworld_bdd10k
experiment-name: yoloworld_bdd10k_finetune
known-classes: car,bus,truck
unknown-prompts: pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign
epochs: 50
batch-size: 8
imgsz: 640
device: 0
amp: true
```

Gunakan ini hanya jika GPU tersedia sebagai device `0`.

## Training Dengan Flag GPU

```bash
bash run_train_yoloworld_bdd10k.sh \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model yolov8s-world.pt \
  --output-dir runs/yoloworld_bdd10k \
  --experiment-name yoloworld_bdd10k_finetune \
  --timestamp-output \
  --epochs 50 \
  --batch-size 8 \
  --imgsz 640 \
  --lr0 1e-4 \
  --device 0 \
  --workers 8 \
  --amp
```

Ultralytics akan mengunduh pretrained weight resmi jika nama seperti `yolov8s-world.pt` belum tersedia lokal. Untuk custom training, model `yolov8s-worldv2.pt` juga dapat dipakai:

```bash
--model yolov8s-worldv2.pt
```

## Training Background

Semua command training di atas memakai [run_train_yoloworld_bdd10k.sh](run_train_yoloworld_bdd10k.sh). Script ini menjalankan Python di background dengan `nohup`, tanpa logger langsung di terminal. Setelah command dimasukkan, terminal langsung kembali ke prompt dan boleh ditutup.

## Resume

```bash
bash run_train_yoloworld_bdd10k.sh \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model runs/yoloworld_bdd10k/yoloworld_bdd10k_finetune/weights/last.pt \
  --output-dir runs/yoloworld_bdd10k \
  --experiment-name yoloworld_bdd10k_finetune \
  --device 0 \
  --amp \
  --resume
```

## Eval-only

```bash
bash run_train_yoloworld_bdd10k.sh \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model runs/yoloworld_bdd10k/yoloworld_bdd10k_finetune/weights/best.pt \
  --eval-only \
  --imgsz 640 \
  --batch-size 8 \
  --device 0 \
  --zero-shot-unknown-model yolov8s-world.pt \
  --unknown-conf-thres 0.05
```

Metric `model.val()` tetap dihitung oleh Ultralytics berdasarkan dataset/yaml. Gambar di `eval_samples/` memakai dua branch: checkpoint fine-tuned untuk known class dan pretrained YOLO-World untuk zero-shot unknown.

## Predict-only

```bash
bash run_train_yoloworld_bdd10k.sh \
  --model runs/yoloworld_bdd10k/yoloworld_bdd10k_finetune/weights/best.pt \
  --predict-only \
  --source data/bdd10k/images/val \
  --conf-thres 0.25 \
  --unknown-conf-thres 0.05 \
  --zero-shot-unknown-model yolov8s-world.pt \
  --iou-thres 0.7 \
  --device 0
```

Prediction tersimpan di:

```text
runs/yoloworld_bdd10k/<experiment-name>/predictions/
runs/yoloworld_bdd10k/<experiment-name>/predictions_unknown/
```

`predictions/prediction_postprocessed.json` menggabungkan bbox known dan unknown. Field `source_model` bernilai `fine_tuned_known` untuk checkpoint hasil training, atau `pretrained_zero_shot_unknown` untuk bbox dari branch unknown.

## Export Model

```bash
bash run_train_yoloworld_bdd10k.sh \
  --model runs/yoloworld_bdd10k/yoloworld_bdd10k_finetune/weights/best.pt \
  --predict-only \
  --source data/bdd10k/images/val \
  --export \
  --export-format onnx
```

Format export mengikuti dukungan Ultralytics, misalnya `onnx`, `torchscript`, atau `openvino`.

## Lokasi Log

```text
training.log
runs/yoloworld_bdd10k/<experiment-name>/train.log
runs/yoloworld_bdd10k/<experiment-name>/console.log
```

`training.log` di root selalu ditimpa setiap run baru. Folder eksperimen timestamped menyimpan `train.log` untuk logger pipeline dan `console.log` untuk stdout/stderr dari Ultralytics.

Log mencatat command, parsed arguments, dataset yaml, model weight, output dir, experiment name, device, seed, epoch, batch size, image size, learning rate, optimizer, AMP, resume, freeze, hasil train/eval/predict/export, checkpoint, dan traceback jika error.

`train.log` dan `training.log` mencatat fase yang sedang berjalan, misalnya dataset loading, model loading, training start, evaluation start, inference start, export start, serta finish marker. Saat training berjalan, callback Ultralytics menulis progress batch per epoch dengan format seperti:

```text
Training progress: epoch=1/50 batch=175/1750 elapsed=1m22s eta=11m05s avg_loss=[...]
```

Detail progress bar asli dari Ultralytics tetap tersimpan live di `console.log`. Setelah training selesai, pipeline membaca `results.csv` dan menulis ringkasan metrik setiap epoch ke `train.log` dan `training.log`. Di akhir run, logger menulis:

```text
Notebook finished. elapsed_seconds=<detik> experiment_dir=<folder_run>
```

## Visual Eval Samples

Default setelah training/eval, pipeline menyimpan 16 gambar contoh dari `data/bdd10k/images/val` ke `eval_samples/`. Setiap gambar output berisi panel kiri ground truth dan panel kanan inference. Atur jumlah atau sumber gambar dengan:

```bash
bash run_train_yoloworld_bdd10k.sh \
  --sample-source data/bdd10k/images/val \
  --sample-count 24
```

Matikan visual sample jika hanya ingin training cepat:

```bash
bash run_train_yoloworld_bdd10k.sh --no-save-eval-samples
```

Untuk melihat run terbaru:

```bash
ls -td runs/yoloworld_bdd10k/* | head -1
tail -f training.log
```

## Smoke Test

Setelah label YOLO tersedia, jalankan smoke test 1 epoch di GPU:

```bash
bash run_train_yoloworld_bdd10k.sh \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model yolov8s-world.pt \
  --output-dir runs/yoloworld_bdd10k \
  --experiment-name smoke_yoloworld_bdd10k \
  --timestamp-output \
  --epochs 1 \
  --batch-size 2 \
  --imgsz 320 \
  --workers 0 \
  --device 0 \
  --amp \
  --patience 1
```
