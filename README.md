# YOLO-World BDD10K Pipeline

Pipeline ini menjalankan training, evaluation, prediction, visual evaluation, dan export YOLO-World untuk BDD10K memakai `ultralytics`. Training dijalankan lewat script bash background dengan `nohup`, sehingga terminal langsung kembali ke prompt setelah command dimasukkan.

Default eksperimen:

```text
Known supervised classes : car, bus, truck
Unknown zero-shot prompts: pedestrian, rider, train, motorcycle, bicycle, traffic light, traffic sign
Training model           : yolov8s-world.pt atau checkpoint YOLO-World lain
Unknown model            : yolov8s-world.pt pretrained
Output root              : runs/yoloworld_bdd10k/
Root log                 : training.log
```

## 1. Setup

Clone repository:

```bash
git clone <url-repo-anda>
cd cvlanjut-project
```

Install `uv` jika belum ada:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Tutup dan buka terminal lagi, lalu cek:

```bash
uv --version
```

Install dependency project:

```bash
uv sync
```

Project memakai Python `>=3.10,<3.11`. Setelah `uv sync`, runner bash otomatis memakai `.venv/bin/python` jika tersedia.

## 2. Download Dataset

Download BDD10K dari Kaggle, unzip langsung ke folder `data`, lalu hapus zip.

```bash
curl -L -o ./bdd10k.zip \
  https://www.kaggle.com/api/v1/datasets/download/aadityadamle/bdd10k
unzip -q ./bdd10k.zip -d data
rm -f ./bdd10k.zip
```

Jika Kaggle meminta autentikasi, pastikan credential tersedia di `~/.kaggle/kaggle.json`.

Struktur yang diharapkan:

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

Converter pada langkah berikutnya akan membuat `data/bdd10k/images/train`, `images/val`, dan `images/test` secara otomatis.

Cek lokasi file annotation JSON setelah unzip:

```bash
find data/bdd10k -type f -name '*.json' | sort | head -50
```

Jika `bdd100k_labels_images_train.json` berada di subfolder lain, pakai path yang ditemukan pada flag `--input-json`.

Default class mapping di `data/bdd10k/bdd10k.yaml`:

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

## 3. Convert dan Check Dataset

Setelah unzip dengan command download di atas, dataset Kaggle berisi image mentah di:

```text
data/bdd10k/train
data/bdd10k/val
data/bdd10k/test
```

Karena split `test` dari Kaggle tidak memiliki annotation JSON, pipeline menyiapkan split Ultralytics seperti ini:

```text
Ultralytics train: subset dari data/bdd10k/train
Ultralytics val  : subset dari data/bdd10k/train untuk validasi saat training
Ultralytics test : seluruh data/bdd10k/val untuk metrik performa akhir
```

Path yang dipakai `data/bdd10k/bdd10k.yaml` tetap:

```text
data/bdd10k/images/train
data/bdd10k/images/val
data/bdd10k/images/test
```

Jalankan converter mode otomatis. Command ini membuat split image, convert JSON ke YOLO label, serta menulis `bdd10k.yaml`:

```bash
.venv/bin/python scripts/convert_bdd10k_to_yolo.py \
  --data-root data/bdd10k
```

Default `--val-ratio` adalah `0.2`, artinya 20% dari original train dipakai sebagai validation. Jika ingin mengubah rasio:

```bash
.venv/bin/python scripts/convert_bdd10k_to_yolo.py \
  --data-root data/bdd10k \
  --val-ratio 0.2 \
  --seed 42
```

Converter mempertahankan 10 kelas BDD asli. Filtering ke known class dilakukan otomatis oleh runner di folder eksperimen.

Jika ingin convert split manual, format ini tetap didukung:

```bash
.venv/bin/python scripts/convert_bdd10k_to_yolo.py \
  --input-json data/bdd10k/labels/bdd100k_labels_images_train.json \
  --image-dir data/bdd10k/images/train \
  --output-label-dir data/bdd10k/labels/train
```

Check dataset:

```bash
.venv/bin/python scripts/check_bdd10k_dataset.py \
  --data-yaml data/bdd10k/bdd10k.yaml
```

Checker melaporkan folder image/label, jumlah image, missing label, empty label, invalid class id, invalid bbox, dan distribusi annotation per class.

Jika muncul error `Input JSON not found`, cek lokasi JSON sebenarnya:

```bash
find data/bdd10k -type f -name '*.json' | sort
```

Catatan: `data/bdd10k/images/test` bukan berasal dari folder raw `data/bdd10k/test`, tetapi dari raw `data/bdd10k/val` agar test akhir memiliki ground truth dan metrik performa bisa dihitung.

## 4. Cara Kerja Pipeline

### Training Known Class

Training supervised hanya memakai class berikut secara default:

```python
["car", "bus", "truck"]
```

Runner membuat dataset sementara:

```text
runs/yoloworld_bdd10k/<experiment-name>/dataset_known/
```

Di dataset sementara ini, label hanya berisi known class dan class id di-remap menjadi `0..2`. Text/CLIP/prompt encoder dibekukan secara best-effort dengan default `--freeze-text-encoder`.

### Zero-Shot Unknown

Model fine-tuned 3 kelas sering menjadi terlalu spesifik dan tidak lagi memilih prompt unknown dengan baik. Karena itu pipeline memakai dua branch untuk evaluation sample dan predict:

```text
Branch known  : checkpoint fine-tuned, prompt car,bus,truck
Branch unknown: YOLO-World pretrained, prompt unknown zero-shot
```

Default unknown prompt:

```python
["pedestrian", "rider", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
```

Semua bbox dari branch unknown disimpan sebagai:

```text
Unknown Object
```

Pada visual evaluation:

```text
Kiri  : ground truth
Kanan : inference
Hijau : known class
Merah : Unknown Object
```

## 5. Training

Semua command di bawah memakai [run_train_yoloworld_bdd10k.sh](run_train_yoloworld_bdd10k.sh). Script menjalankan proses dengan `nohup` di background tanpa logger langsung di terminal.

Training tanpa flag:

```bash
bash run_train_yoloworld_bdd10k.sh
```

Default training tanpa flag:

```text
data-yaml      : data/bdd10k/bdd10k.yaml
model          : yolov8s-world.pt
output-dir     : runs/yoloworld_bdd10k
experiment-name: yoloworld_bdd10k_finetune
epochs         : 50
batch-size     : 8
imgsz          : 640
device         : 0
amp            : true
known-classes  : car,bus,truck
```

Training dengan flag:

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

Smoke test GPU:

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

Resume:

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

## 6. Evaluation

Eval-only:

```bash
bash run_train_yoloworld_bdd10k.sh \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model runs/yoloworld_bdd10k/yoloworld_bdd10k_finetune/weights/best.pt \
  --output-dir runs/yoloworld_bdd10k \
  --experiment-name eval_yoloworld_bdd10k_finetune \
  --timestamp-output \
  --eval-only \
  --eval-split test \
  --imgsz 640 \
  --batch-size 8 \
  --device 0 \
  --sample-source data/bdd10k/images/test \
  --sample-count 24 \
  --zero-shot-unknown-model yolov8s-world.pt \
  --unknown-conf-thres 0.05
```

`model.val()` menghitung metric Ultralytics pada split `test` secara default. Dalam pipeline ini, split `test` berasal dari raw BDD `val`, sehingga tetap memiliki ground truth. Gambar di `evaluation/images/` dibuat oleh dua branch: checkpoint fine-tuned untuk known class dan YOLO-World pretrained untuk unknown zero-shot.

Matikan visual sample jika hanya ingin metric:

```bash
bash run_train_yoloworld_bdd10k.sh \
  --model runs/yoloworld_bdd10k/yoloworld_bdd10k_finetune/weights/best.pt \
  --eval-only \
  --no-save-eval-samples \
  --device 0
```

## 7. Prediction

Predict dengan default prompt:

```bash
bash run_train_yoloworld_bdd10k.sh \
  --model runs/yoloworld_bdd10k/yoloworld_bdd10k_finetune/weights/best.pt \
  --output-dir runs/yoloworld_bdd10k \
  --experiment-name predict_yoloworld_bdd10k \
  --timestamp-output \
  --predict-only \
  --source data/bdd10k/images/val \
  --conf-thres 0.25 \
  --unknown-conf-thres 0.05 \
  --zero-shot-unknown-model yolov8s-world.pt \
  --iou-thres 0.7 \
  --device 0
```

Predict dengan custom prompt:

```bash
bash run_train_yoloworld_bdd10k.sh \
  --model runs/yoloworld_bdd10k/yoloworld_bdd10k_finetune/weights/best.pt \
  --output-dir runs/yoloworld_bdd10k \
  --experiment-name predict_custom_prompt \
  --timestamp-output \
  --predict-only \
  --source path/to/image_or_folder \
  --known-classes car,bus,truck \
  --unknown-prompts "person,animal,road debris,traffic cone,object,obstacle,item" \
  --zero-shot-unknown-model yolov8s-world.pt \
  --conf-thres 0.25 \
  --unknown-conf-thres 0.05 \
  --iou-thres 0.7 \
  --device 0
```

Output prediction:

```text
runs/yoloworld_bdd10k/<experiment-name>/predictions/
runs/yoloworld_bdd10k/<experiment-name>/predictions_unknown/
runs/yoloworld_bdd10k/<experiment-name>/predictions/prediction_postprocessed.json
```

`prediction_postprocessed.json` menggabungkan known dan unknown. Field `source_model` menunjukkan asal bbox:

```text
fine_tuned_known
pretrained_zero_shot_unknown
```

## 8. Notebook

Notebook utama:

```text
notebooks/train_yoloworld_bdd10k.ipynb
```

Notebook memakai logic yang sama dari `scripts/yoloworld_bdd10k_pipeline.py`. Jika dijalankan tanpa flag, notebook memakai default config.

Execute notebook dari terminal:

```bash
jupyter nbconvert --to notebook --execute notebooks/train_yoloworld_bdd10k.ipynb \
  --output executed_train_yoloworld_bdd10k.ipynb
```

Atau memakai papermill:

```bash
papermill notebooks/train_yoloworld_bdd10k.ipynb notebooks/executed_train_yoloworld_bdd10k.ipynb
```

## 9. Export

Export model mengikuti format yang didukung Ultralytics, misalnya `onnx`, `torchscript`, atau `openvino`.

```bash
bash run_train_yoloworld_bdd10k.sh \
  --model runs/yoloworld_bdd10k/yoloworld_bdd10k_finetune/weights/best.pt \
  --predict-only \
  --source data/bdd10k/images/val \
  --export \
  --export-format onnx \
  --device 0
```

## 10. Output dan Log

Setiap run disimpan di:

```text
runs/yoloworld_bdd10k/<experiment-name>/
```

Isi penting:

```text
logs/
  train.log
  console.log
configs/
  args.json
  config_used.yaml
  run_summary.json
  ultralytics_args.yaml
metrics/
  training_history.csv
  metrics_summary.json
  evaluation.json
  final_metrics.csv
  confidence_by_label.csv
  confidence_by_label.png
evaluation/
  images/
  sample_predictions.json
weights/
  best.pt
  last.pt
predictions/
predictions_unknown/
```

`metrics/training_history.csv` berisi history training per epoch. `metrics/final_metrics.csv` berisi performa akhir. `metrics/confidence_by_label.png` adalah bar chart confidence rata-rata dari deteksi yang dijalankan pada evaluation/predict samples. `weights/best.pt` dan `weights/last.pt` menyimpan model terbaik dan checkpoint terakhir.

Jika ingin confidence chart dari seluruh test image, jalankan `predict-only` dengan `--source data/bdd10k/images/test`, atau naikkan `--sample-count` pada eval sampai mencakup jumlah gambar test yang ingin dianalisis.

Log utama:

```text
training.log
runs/yoloworld_bdd10k/<experiment-name>/logs/train.log
runs/yoloworld_bdd10k/<experiment-name>/logs/console.log
```

`training.log` di root selalu ditimpa setiap run baru. `train.log` menyimpan logger pipeline permanen per eksperimen. `console.log` menyimpan stdout/stderr Ultralytics.

Monitor run:

```bash
tail -f training.log
```

Lihat run terbaru:

```bash
ls -td runs/yoloworld_bdd10k/* | head -1
```

Log mencatat command, parsed arguments, dataset yaml, model weight, output dir, experiment name, device, seed, epoch, batch size, image size, learning rate, optimizer, AMP, resume, freeze, hasil train/eval/predict/export, checkpoint, traceback jika error, dan finish marker:

```text
Notebook finished. elapsed_seconds=<detik> experiment_dir=<folder_run>
```

## 11. Troubleshooting

Jika CUDA tidak tersedia tetapi `--device 0` dipakai, pipeline akan memberi error informatif. Gunakan GPU yang benar atau ubah ke `--device cpu` untuk test kecil.

Jika bbox unknown tidak muncul dari checkpoint fine-tuned, gunakan branch zero-shot pretrained default:

```bash
--zero-shot-unknown-model yolov8s-world.pt --unknown-conf-thres 0.05
```

Jika ground truth pada panel kiri kosong, pastikan image sample memiliki label pasangan di `data/bdd10k/labels/<split>/<image_name>.txt`.

Jika training atau eval berjalan di background dan ingin dihentikan:

```bash
ps aux | grep run_train_yoloworld_bdd10k.py
kill <PID>
```
