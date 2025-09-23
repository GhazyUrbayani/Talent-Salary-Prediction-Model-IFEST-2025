# 🏆 IFest 2025: Salary Prediction for Tech Professionals

![Banner](https://img.shields.io/badge/Project-Salary%20Prediction-blueviolet) ![Status](https://img.shields.io/badge/Status-Completed-success)

Proyek ini adalah solusi untuk **Data Analytics Competition (DAC) IFest 2025**, yang diselenggarakan oleh Himpunan Mahasiswa Teknik Informatika Universitas Padjadjaran (HIMATIF UNPAD). Kami membangun model *machine learning* untuk memprediksi rentang gaji kandidat berdasarkan data pada CV.

### 💻 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=black">
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-147F9B?style=for-the-badge&logo=xgboost&logoColor=white">
  <img src="https://img.shields.io/badge/LightGBM-4169E1?style=for-the-badge&logo=lightgbm&logoColor=white">
  <img src="https://img.shields.io/badge/CatBoost-FF6600?style=for-the-badge&logo=catboost&logoColor=white">
</p>

---

### 🎯 Tujuan Proyek

Tujuan utama dari proyek ini adalah untuk memprediksi **rentang gaji yang wajar** bagi para profesional di bidang IT. Kami menggunakan model *multi-target regression* untuk memprediksi dua variabel target: `Estimated Bottom` dan `Estimated Up`, yang mewakili batas bawah dan batas atas dari ekspektasi gaji.

---

### 📂 Struktur Direktori

Berikut adalah struktur folderisasi yang digunakan dalam proyek ini:

- .
- ├── Draft/
- │   ├── draft Hakeem.ipynb
- │   ├── draft Si_data_tuh.ipynb
- │   ├── Hypertuned.ipynb
- │   ├── Hypertuned ++.ipynb
- │   └── (File kode draft lainnya...)
- ├── the Best.ipynb
- ├── train.csv
- ├── test.csv
- ├── submission.csv
- └── README.md

Folder **`Draft`** berisi semua eksperimen, ide, dan iterasi awal. Kode terbaik dan paling efisien, yang memberikan hasil akhir untuk kompetisi, berada di `the Best.ipynb`.

---

Model ini bertujuan untuk:
* Membantu perusahaan dalam memberikan penawaran gaji yang kompetitif.
* Memberikan gambaran yang adil bagi para kandidat tentang nilai pasar mereka.

### 📊 Dataset dan Metrik

#### Deskripsi Dataset
Dataset yang digunakan berasal dari kompetisi IFest 2025. Data ini mencakup berbagai informasi dari CV kandidat.

| Kolom | Deskripsi |
| :--- | :--- |
| `ID` | ID unik kandidat |
| `Current Position` | Jabatan terakhir kandidat |
| `Candidate Level` | Tingkat senioritas (Junior, Mid-Level, dll.) |
| `Domisili` | Lokasi tempat tinggal kandidat |
| `Education 1/2/3` | Riwayat pendidikan formal |
| `Tech Stack` | Daftar teknologi yang dikuasai |
| `Certification` | Daftar sertifikasi |
| `Estimate Bottom` | Batas bawah gaji yang diharapkan (Target) |
| `Estimate Up` | Batas atas gaji yang diharapkan (Target) |

#### Metrik Evaluasi
Metrik yang digunakan untuk mengevaluasi performa model adalah **Mean Absolute Error (MAE)**. Semakin rendah nilai MAE, semakin baik model dalam memprediksi gaji.

$$MAE = \frac{1}{n}\sum_{i=1}^{n}\left | y_i - \hat{y}_i \right |$$

Skor MAE akhir adalah rata-rata dari MAE untuk `Estimated Botom` dan `Estimated Up`.

---

### 🚀 Metodologi

Kami mengikuti alur kerja *machine learning* yang sistematis untuk mendapatkan hasil yang optimal.

#### 1. Data Cleaning & Feature Engineering
* **Penanganan Kolom Teks**: Kami mengubah kolom `Tech Stack` dan `Certification` yang berupa teks menjadi fitur numerik dengan menghitung jumlah item di dalamnya (`Tech Stack_count` dan `Certification_count`).
* **Klasifikasi Pendidikan**: Kami membuat fungsi khusus (`classify_university_prestige`) untuk mengklasifikasikan universitas berdasarkan tingkat prestise mereka (Tier 1-6). Fitur baru seperti `university_prestige_score`, `is_top_university`, dan lainnya diekstrak dari kolom `Education 1` dan `Education 2`.
* **Kategorisasi Jabatan**: Kolom `Candidate Level` diubah menjadi fitur numerik yang merepresentasikan senioritas (misalnya, Junior = 1, Mid-Level = 2).
* **Penanganan Missing Values**: Kami menggunakan **KNN Imputer** untuk mengisi nilai yang hilang, yang lebih canggih daripada imputasi median/modus sederhana.

#### 2. Pra-pemrosesan Data
* Kami menggunakan `ColumnTransformer` dan `Pipeline` untuk mengotomatisasi seluruh alur pra-pemrosesan. Ini memastikan bahwa setiap model menerima data yang konsisten dan bersih.
* Kolom kategorikal di-*encode* menjadi numerik terlebih dahulu menggunakan `OrdinalEncoder`, baru kemudian nilai yang hilang diisi.

#### 3. Pemodelan Regresi Multitarget
* Kami melatih dua model `HistGradientBoostingRegressor` yang terpisah:
    1. Satu model dilatih untuk memprediksi `Estimate Bottom`.
    2. Satu model lainnya dilatih untuk memprediksi `Estimate Up`.
* Pendekatan ini adalah yang paling efektif untuk masalah regresi multitarget, karena setiap target bisa memiliki pola data yang berbeda.

#### 4. Validasi dan Evaluasi
* Model dievaluasi menggunakan **K-Fold Cross-Validation** dengan metrik MAE untuk mendapatkan estimasi performa yang stabil.
* Kami juga memastikan bahwa prediksi `Estimated Up` tidak lebih rendah dari `Estimated Bottom` sebagai langkah validasi akhir.

---

### 📦 Cara Menjalankan Kode

1.  **Unduh Dataset**: Pastikan file `train.csv` dan `test.csv` berada di direktori yang sama dengan skrip.
2.  **Instalasi Library**: Jalankan perintah berikut untuk menginstal semua library yang dibutuhkan:
    ```bash
    pip install pandas numpy scikit-learn lightgbm xgboost catboost
    ```
3.  **Jalankan Skrip**: Jalankan file `predict_salary.py` untuk memulai pelatihan model dan menghasilkan file submission.
    ```bash
    python predict_salary.py
    ```

### 📄 Output

Skrip akan menghasilkan file `submission.csv` yang berisi prediksi akhir dalam format yang sesuai dengan aturan kompetisi.

| ID | Estimated Botom | Estimated Up |
| :--- | :--- | :--- |
| SKU-WENU549 | 8,281,402.65 | 8,281,402.65 |
| SKU-NUNU771 | 8,220,314.84 | 8,220,314.84 |
| ... | ... | ... |

---
*Kode dan analisis ini merupakan hasil dari proses iterasi dan perbaikan, yang menunjukkan pentingnya ketelitian dalam setiap langkah analisis data. Kami berharap ini bermanfaat.*
