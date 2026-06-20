Berikut adalah revisi lengkap dari templat tabel dan deskripsi visualisasi, disesuaikan dengan instruksi Anda untuk memisahkan skema berdasarkan *training data*, merestrukturisasi Tabel 3, menghapus metrik yang tidak diperlukan pada Tabel efisiensi, dan menggunakan terminologi bahasa Inggris standar untuk istilah teknis agar relevan dengan format jurnal IEEE.

### 1. Templat Tabel dan Deskripsi Fungsional

**Tabel 1: Evaluation Metrics for Models Trained on All Classes (Scheme 1 & 3)**
Tabel ini khusus menampilkan model yang dilatih menggunakan seluruh dataset (10 kelas). Tabel ini berfungsi sebagai *baseline* batas atas (*upper bound*) untuk membandingkan kapasitas maksimal arsitektur YOLO-World melawan Standard YOLO ketika keduanya tidak dihadapkan pada skenario *zero-shot*.

| Scheme ID | Model & Training Config | Scale | $mAP_{50}$ | $mAP_{50-95}$ | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Scheme 1** | YOLO-World (All Class) | s <br>

<br> m <br>

<br> l |  |  |  |  |  |
| **Scheme 3** | Standard YOLO (All Class) | s <br>

<br> m <br>

<br> l |  |  |  |  |  |

**Tabel 2: Zero-Shot Evaluation Metrics for Models Trained on Known Classes (Scheme 2 & 4)**
Tabel ini membandingkan langsung kedua model yang di-*train* hanya pada himpunan kelas terbatas (3 *Known Classes*). Evaluasi dilakukan secara terpisah untuk mengukur seberapa baik YOLO-World menangani *Unknown Classes* melalui *text prompts*, dan menegaskan kegagalan Standard YOLO (dengan nilai 0.00) pada skenario *closed-set*.

| Scheme ID | Model & Training Config | Scale | Evaluation Target | $mAP_{50}$ | $mAP_{50-95}$ | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Scheme 2** | YOLO-World (Known Class) | s <br>

<br> m <br>

<br> l | Unknown Classes (Zero-Shot) |  |  |  |  |  |
| **Scheme 4** | Standard YOLO (Known Class) | s <br>

<br> m <br>

<br> l | Unknown Classes (Zero-Shot) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Tabel 3: Per-Class $mAP_{50}$ Comparison on Unknown Classes**
Sesuai revisi Anda, tabel ini menjabarkan metrik $mAP_{50}$ per kelas tak dikenal. Kolom mempresentasikan varian ukuran dari masing-masing arsitektur. Format ini sangat efisien untuk memperlihatkan kelas mana (misal: *pedestrian* vs *traffic light*) yang paling sulit diprediksi secara *zero-shot* oleh YOLO-World, sekaligus menyandingkannya dengan kolom Standard YOLO yang bernilai 0.00.

| Unknown Classes | YW-Small | YW-Medium | YW-Large | SY-Small | SY-Medium | SY-Large |
| --- | --- | --- | --- | --- | --- | --- |
| pedestrian |  |  |  | 0.00 | 0.00 | 0.00 |
| rider |  |  |  | 0.00 | 0.00 | 0.00 |
| train |  |  |  | 0.00 | 0.00 | 0.00 |
| motorcycle |  |  |  | 0.00 | 0.00 | 0.00 |
| bicycle |  |  |  | 0.00 | 0.00 | 0.00 |
| traffic light |  |  |  | 0.00 | 0.00 | 0.00 |
| traffic sign |  |  |  | 0.00 | 0.00 | 0.00 |

> *Note: YW = YOLO-World (Scheme 2), SY = Standard YOLO (Scheme 4).*

**Tabel 4: Class-Agnostic Evaluation & Misclassification Analysis (Standard YOLO - Scheme 4)**
Tabel ini digunakan agar komparasi tetap *apple-to-apple*. Alih-alih hanya memberi nilai 0.00, tabel ini membedah *mengapa* SY gagal. Apakah SY mendeteksi objek namun salah melabelinya menjadi mobil/truk (*Misclassification*), atau apakah SY benar-benar buta terhadap objek tersebut (*Miss Rate* / *False Negative*).

| Standard YOLO Scale | Class-Agnostic Recall | Misclassification to Known Classes | Miss Rate (False Negative Rate) |
| --- | --- | --- | --- |
| Small |  |  |  |
| Medium |  |  |  |
| Large |  |  |  |

**Tabel 5: Computational Efficiency and Inference Latency**
Tabel ini menyoroti *trade-off* antara arsitektur *open-vocabulary* dan *closed-set*. FPS dan VRAM telah dihapus sesuai permintaan, menyisakan parameter dasar untuk mengevaluasi kelayakan implementasi *real-time*.

| Model & Training Config | Scale | Parameters (M) | Inference Time (ms) |
| --- | --- | --- | --- |
| YOLO-World | s <br>

<br> m <br>

<br> l |  |  |
| Standard YOLO | s <br>

<br> m <br>

<br> l |  |  |

**Tabel 6: Pure Pretrained YOLO-World Baseline (No Fine-Tuning)**
Tabel ini menambahkan baseline YOLO-World Small, Medium, dan Large yang dijalankan murni dari pretrained weights Ultralytics tanpa proses training atau fine-tuning pada BDD10K. Evaluasi memakai prompt kelas BDD10K yang sama agar performanya dapat dibandingkan sebagai baseline *zero-shot/open-vocabulary* murni.

| Baseline | Scale | Evaluation Target | $mAP_{50}$ | $mAP_{50-95}$ | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO-World Pretrained (No Fine-Tuning) | Small | All Classes |  |  |  |  |  |
| YOLO-World Pretrained (No Fine-Tuning) | Small | Unknown Classes |  |  |  |  |  |
| YOLO-World Pretrained (No Fine-Tuning) | Medium | All Classes |  |  |  |  |  |
| YOLO-World Pretrained (No Fine-Tuning) | Medium | Unknown Classes |  |  |  |  |  |
| YOLO-World Pretrained (No Fine-Tuning) | Large | All Classes |  |  |  |  |  |
| YOLO-World Pretrained (No Fine-Tuning) | Large | Unknown Classes |  |  |  |  |  |

**Tabel 7: Per-Class $mAP_{50}$ for Pure Pretrained YOLO-World on Unknown Classes**
Tabel ini memecah performa *zero-shot* pretrained YOLO-World per kelas unknown agar efek scaling Small, Medium, dan Large dapat diamati tanpa pengaruh fine-tuning.

| Unknown Classes | Pretrained YW-Small | Pretrained YW-Medium | Pretrained YW-Large |
| --- | --- | --- | --- |
| pedestrian |  |  |  |
| rider |  |  |  |
| train |  |  |  |
| motorcycle |  |  |  |
| bicycle |  |  |  |
| traffic light |  |  |  |
| traffic sign |  |  |  |

---

### 2. Panduan Visualisasi Grafik dan Kualitatif

**Grafik 1: Grouped Bar Chart untuk Performa Zero-Shot per Kelas**

* **Deskripsi:** *Grouped Bar Chart* dengan sumbu X berisi tujuh kelas *Unknown* dan sumbu Y berisi persentase $mAP_{50}$. Setiap kelas di sumbu X memiliki tiga bar bersebelahan yang mewakili skala Small, Medium, dan Large dari YOLO-World (Scheme 2).
* **Tujuan:** Memberikan impresi visual instan mengenai efek *scaling* parameter model terhadap performa *zero-shot* pada spesifik kelas (misal: apakah model ukuran Large mendeteksi *pedestrian* lebih baik daripada model Small).

**Grafik 2: Box Plot untuk Distribusi Confidence Score**

* **Deskripsi:** *Box Plot* yang mendistribusikan *Confidence Score* (0.0 - 1.0) dari prediksi YOLO-World (Scheme 2) saat mendeteksi *Unknown Classes*, dibandingkan dengan *Confidence Score* prediksi keliru (*Misclassification/False Positive*) dari Standard YOLO (Scheme 4) pada area *Bounding Box* yang sama.
* **Tujuan:** Menunjukkan bahwa saat Standard YOLO memaksakan deteksi (misal: menebak *pedestrian* sebagai *car*), *confidence score*-nya sangat rendah/marjinal, dibandingkan dengan YOLO-World yang memiliki keyakinan tinggi berkat *vision-language features*.

**Visualisasi Kualitatif: Bounding Box Detection Comparison Grid**

* **Deskripsi:** Sebuah *image grid* yang menampilkan 3-4 skenario jalan raya (misalnya kondisi *crowded* dengan *pedestrian* dan *traffic light*).
* **Kolom 1 (Ground Truth):** Gambar dengan *Bounding Box* kebenaran asli (anotasi dataset).
* **Kolom 2 (YOLO-World - Scheme 2):** Hasil prediksi sukses melokalisasi kelas *unknown* menggunakan *text prompts*.
* **Kolom 3 (Standard YOLO - Scheme 4):** Memperlihatkan *missed detection* (tidak ada kotak sama sekali) atau *misclassification* (label salah).


* **Tujuan:** Memberikan justifikasi kualitatif visual kepada *reviewer* IEEE mengenai urgensi keselamatan *safety hazard* dari model *closed-set* di ekosistem *autonomous driving*.
