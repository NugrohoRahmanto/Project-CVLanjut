# Konteks Agen LLM: Riset Evaluasi Klasifikasi Terbuka (Open-Vocabulary) YOLO-World vs Standard YOLO pada Dataset BDD 10K

Dokumen ini berfungsi sebagai panduan konteks utama bagi Agen LLM untuk memahami latar belakang, tujuan, konfigurasi dataset, arsitektur model, dan skema evaluasi dalam proyek riset ini.

---

## 1. Latar Belakang & Urgensi Riset
Sistem deteksi objek konvensional pada kendaraan otonom (*autonomous driving*) umumnya berbasis pada *closed-set object detection* (seperti Standard YOLO). Model ini hanya mampu mengenali objek yang telah didaftarkan dan dilatih secara eksplisit pada fase *training*. Di lingkungan jalan raya yang dinamis, keterbatasan ini menimbulkan risiko fatal (masalah *safety*) ketika kendaraan menemui objek asing (*unknown objects*) yang tidak ada dalam data latihan.

**YOLO-World** hadir sebagai solusi berbasis *Open-Vocabulary Object Detection* yang menggabungkan fitur visual dengan embedding teks (bahasa alami). Riset ini bertujuan untuk menguji secara empiris sejauh mana YOLO-World mampu melakukan deteksi *zero-shot* terhadap kelas-kelas yang tidak dipelajari (*unknown/unseen*) dibandingkan dengan YOLO standar, guna meningkatkan faktor keselamatan pada *autonomous driving*.

---

## 2. Tujuan Riset
1. **Komparasi Performa:** Membandingkan secara komprehensif kemampuan deteksi objek antara varian arsitektur YOLO-World dan Standard YOLO.
2. **Evaluasi Deteksi Objek Tak Dikenal:** Mengukur keandalan model dalam mendeteksi kelas *unknown* menggunakan skenario *zero-shot* (untuk YOLO-World) dan melihat limitasi pada Standard YOLO.
3. **Standardisasi Output Riset IEEE:** Memastikan metodologi pengujian, pelaporan hasil eksperimen, dan kalkulasi metrik evaluasi yang digunakan selaras dengan standar publikasi penelitian IEEE. Fokus utama adalah reproduksibilitas metodologi dan konsistensi metrik pembanding (bukan pada standar penulisan kode).

---

## 3. Konfigurasi Dataset (BDD 10K)
Dataset yang digunakan adalah subset **BDD 10K**. 

*Catatan Penting: Kelas "known" dan "unknown" bersifat dinamis dan dapat diubah sewaktu-waktu melalui pengaturan config atau flag pada program.* Namun, konfigurasi kelas di bawah ini wajib dijadikan sebagai **acuan dan default config** agar Agen LLM memahami skenario pemisahan dasar yang digunakan.

Secara *default*, terdapat **10 kelas** yang dibagi menjadi dua kategori:

### A. Kelas yang Dilatih (Known Classes) - Default Config
* `car`
* `bus`
* `truck`

### B. Kelas yang Tidak Dilatih (Unknown/Unseen Classes) - Default Config
* `pedestrian`
* `rider`
* `train`
* `motorcycle`
* `bicycle`
* `traffic light`
* `traffic sign`

> **Catatan untuk Agen LLM:** Saat melatih model dengan konfigurasi *Known Class*, kelas-kelas *Unknown* di atas harus diabaikan dari proses optimasi bobot (misal: dianotasikan ulang sebagai background atau difilter), namun tetap dipertahankan pada data uji (*test/validation set*) sebagai *ground truth unknown* untuk menguji kemampuan deteksi *zero-shot*.

---

## 4. Skema Model & Eksperimen
Riset ini melibatkan **4 skema model utama**, di mana masing-masing skema akan diuji menggunakan 3 varian skala ukuran model YOLO, yaitu: **Small (s)**, **Medium (m)**, dan **Large (l)**. Total terdapat **12 varian eksperimen model**.

Berikut adalah matriks konfigurasi eksperimen (dengan mengacu pada *default config*):

| ID Skema | Nama Model | Konfigurasi Latihan | Deskripsi Eksperimen | Varian Skala |
| :--- | :--- | :--- | :--- | :---: |
| **Skema 1** | YOLO-World (All Class) | All Class (*default*: 10 kelas) | Dilatih menggunakan seluruh kelas terdaftar untuk mengukur performa batas atas (*upper bound*). | s, m, l |
| **Skema 2** | YOLO-World (Known Class) | Known Class (*default*: 3 kelas) | Dilatih hanya pada himpunan kelas *known*. Kelas *unknown* lainnya diuji secara *zero-shot* menggunakan teks prompt. | s, m, l |
| **Skema 3** | Standard YOLO (All Class) | All Class (*default*: 10 kelas) | Model standar *closed-set* yang dilatih dengan seluruh kelas sebagai pembanding *baseline*. | s, m, l |
| **Skema 4** | Standard YOLO (Known Class) | Known Class (*default*: 3 kelas) | Model standar *closed-set* yang hanya dilatih pada kelas *known* untuk membuktikan kegagalan deteksi pada kelas di luar latihan. | s, m, l |

---

## 5. Metodologi Evaluasi & Perhitungan Metrik
Performa dari ke-12 eksperimen model di atas wajib dievaluasi secara komprehensif menggunakan metrik: **mAP50, mAP50-95, Precision, Recall, dan F1-Score rata-rata**. Evaluasi dibagi menjadi dua bagian utama untuk melihat kemampuan model secara utuh dan spesifik pada objek yang tidak dilatih:

### Bagian I: Evaluasi terhadap Semua Ground Truth (All Class)
* **Tujuan:** Mengukur performa generalisasi model secara keseluruhan terhadap semua objek yang ada di *ground truth* (baik kelas *known* maupun *unknown*).
* **Mekanisme:** Seluruh hasil *inference* dibandingkan dengan label *ground truth* dari semua kelas yang terdaftar pada sistem (contoh *default*: 10 kelas).
* **Metrik Evaluasi:** $mAP_{50}$, $mAP_{50-95}$, *Precision*, *Recall*, dan *F1-Score* rata-rata yang dihitung untuk seluruh kelas.

### Bagian II: Evaluasi terhadap Ground Truth Kelas Tak Dikenal (Unknown Class)
* **Tujuan:** Mengukur secara spesifik kemampuan *Zero-Shot / Open-Vocabulary* dari model dalam mengenali objek yang sengaja disembunyikan/tidak dipelajari selama proses *training*.
* **Mekanisme:** Hasil prediksi model difilter dan hanya dievaluasi terhadap *ground truth* yang mendefinisikan himpunan kelas *unknown* (contoh *default*: 7 kelas tidak dikenal).
* **Metrik Evaluasi:** $mAP_{50}$, $mAP_{50-95}$, *Precision*, *Recall*, dan *F1-Score* rata-rata yang secara spesifik dihitung hanya untuk kelas-kelas *unknown*.