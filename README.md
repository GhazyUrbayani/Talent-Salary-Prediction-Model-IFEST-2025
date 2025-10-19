# 🏆 IFest 2025: Salary Prediction for Tech Professionals

![Banner](https://img.shields.io/badge/Project-Salary%20Prediction-blueviolet) ![Status](https://img.shields.io/badge/Status-Completed-success)

Proyek ini adalah solusi untuk **Data Analytics Competition (DAC) IFest 2025**, yang diselenggarakan oleh Himpunan Mahasiswa Teknik Informatika Universitas Padjadjaran (HIMATIF UNPAD). Kami membangun model *machine learning* untuk memprediksi rentang gaji kandidat berdasarkan data pada CV.

### 💻 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-147F9B?style=for-the-badge&logo=xgboost&logoColor=white">
  <img src="https://img.shields.io/badge/LightGBM-4169E1?style=for-the-badge&logo=lightgbm&logoColor=white">
  <img src="https://img.shields.io/badge/CatBoost-FF6600?style=for-the-badge&logo=catboost&logoColor=white">
  <img src="https://img.shields.io/badge/GradientBoosting-F39C12?style=for-the-badge&logo=scikit-learn&logoColor=white">
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
- ├── Draft/
- │   ├── train.csv
- │   └── test.csv
- ├── submission_final.csv
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

---

#### Metrik Evaluasi
Metrik yang digunakan untuk mengevaluasi performa model adalah **Mean Absolute Error (MAE)**. Semakin rendah nilai MAE, semakin baik model dalam memprediksi gaji.

$$MAE = \frac{1}{n}\sum_{i=1}^{n}\left | y_i - \hat{y}_i \right |$$

Skor MAE akhir adalah rata-rata dari MAE untuk `Estimated Botom` dan `Estimated Up`.

---

### 📈 Evolusi Proyek: Dari Draft ke Kode Terbaik

Proyek ini adalah hasil dari proses iterasi dan perbaikan yang sistematis. Berikut ringkasan perbedaan antara kode-kode yang diuji:

| Nama Kode | Metode Kunci yang Digunakan | Kenapa Kode Terbaik? |
| :--- | :--- | :--- |
| `draft Hakeem.ipynb` | **Basic EDA & Imputation.** Mengidentifikasi korelasi awal dengan **Pearson** dan **Cramer's V**. Menguji model regresi sederhana. | Kode ini rentan terhadap *error* karena *preprocessing* yang tidak fleksibel. Modelnya terlalu sederhana untuk data ini. |
| `draft Si_data_tuh.ipynb` | **Initial Pipelines.** Mulai menerapkan *preprocessing* dengan `Pipeline` dan menguji model `LightGBM`. | Kode ini masih memiliki *bug* pada alur *preprocessing* dan `early_stopping`, menyebabkan model gagal. |
| `Hypertuned.ipynb` | **Hyperparameter Tuning.** Mencoba *tuning* manual dan otomatis. Menggunakan `BaggingRegressor` dan `HistGradientBoostingRegressor` sebagai model utama. | Kode ini masih mengalami masalah dengan sintaks `early_stopping` di *pipeline* `sklearn`, yang menyebabkan `TypeError`. |
| `Hypertuned ++.ipynb` | **Advanced Tuning & Stacking.** Kode ini memiliki alur *preprocessing* yang lebih kuat dan mencoba strategi *stacking*. | Kode ini adalah pengembangan dari `Hypertuned.ipynb`, tetapi masih memiliki beberapa isu dan belum menghasilkan performa terbaik. |
| `the Best.ipynb` | **Final, Polished Solution.** Menggabungkan semua perbaikan: alur *preprocessing* yang sangat tangguh, **TF-IDF** untuk fitur teks, ekstraksi fitur **tahun kelulusan**, dan model yang diuji secara sistematis. | Kode ini adalah hasil terbaik karena memuat semua perbaikan, menghasilkan MAE yang paling rendah dan stabil, serta memberikan visualisasi yang jelas. |

---

### 🚀 Metodologi Terbaik ✅ ("the Best.ipynb")

Notebook **`the Best.ipynb`** adalah puncak dari semua eksperimen yang dilakukan. Berikut adalah metodologi yang membuatnya menjadi yang terbaik:

#### 1. Pra-pemrosesan Data
* **Data Cleansing**: Kolom `'ID'`, `'Name'`, `'Expected Benefit...'`, dan `'Target Position'` dihapus karena tidak relevan atau redundan.
* **Feature Engineering**:
    * **Teks**: Kami mengubah kolom `Tech Stack` dan `Certification` menjadi fitur numerik yang lebih informatif dengan menghitung jumlah item di dalamnya (`Tech Stack_count` dan `Certification_count`).
    * **Pendidikan**: Kami mengklasifikasikan universitas ke dalam 6 tingkatan prestise dan juga mengekstrak `years_since_graduation`. Kami membuat fungsi khusus (`classify_university_prestige`) untuk mengklasifikasikan universitas berdasarkan tingkat prestise mereka (Tier 1-6). Fitur baru seperti `university_prestige_score`, `is_top_university`, dan lainnya diekstrak dari kolom `Education 1` dan `Education 2`.
    * **Kategorisasi Jabatan**: Kolom `Candidate Level` diubah menjadi fitur numerik yang merepresentasikan senioritas (misalnya, Junior = 1, Mid-Level = 2).
* **Preprocessing Pipeline**: Kami menggunakan `ColumnTransformer` yang menggabungkan `KNNImputer` untuk data numerik dan `OrdinalEncoder` untuk data kategorikal, memastikan data siap untuk model.
* **Penanganan Missing Values**: Kami menggunakan **KNN Imputer** untuk mengisi nilai yang hilang, yang lebih canggih daripada imputasi median/modus sederhana.
* Kami menggunakan `ColumnTransformer` dan `Pipeline` untuk mengotomatisasi seluruh alur pra-pemrosesan. Ini memastikan bahwa setiap model menerima data yang konsisten dan bersih.
* Kolom kategorikal di-*encode* menjadi numerik terlebih dahulu menggunakan `OrdinalEncoder`, baru kemudian nilai yang hilang diisi.

#### 2. Pemodelan Regresi Multitarget & Tuning
* Kami 'Training' banyak Model atau disebut 'Brute Force' beberapa Model.
* **Model Terbaik**: Berdasarkan hasil *cross-validation* yang stabil, model **CatBoost_Optimized** terpilih sebagai model juara.
* **Metode**: Kami melatih dua model terpisah untuk `Estimate Bottom` dan `Estimate Up` untuk akurasi maksimal.
* Kami melatih dua model `HistGradientBoostingRegressor` yang terpisah:
    1. Satu model dilatih untuk memprediksi `Estimate Bottom`.
    2. Satu model lainnya dilatih untuk memprediksi `Estimate Up`.
* Pendekatan ini adalah yang paling efektif untuk masalah regresi multitarget, karena setiap target bisa memiliki pola data yang berbeda.

#### 3. Hasil Akhir (Verifikasi Lokal)
Berikut adalah hasil dari `the Best.ipynb` yang dijalankan secara lokal:

🏆 **CHAMPION MODEL:** `CatBoost_Optimized`
* **Cross-Validation MAE**: 1.364.166,42 ± 133.128,81
* **Test MAE (Avg)**: 1.265.764,56
* **$R^2$ Score (Avg)**: 0.788
* **Bottom Metrics**: MAE=1.013.448, $R^2$=0.786
* **Up Metrics**: MAE=1.518.081, $R^2$=0.790
* **MAPE (Avg)**: 13.9%

Fitur-fitur yang paling berpengaruh terhadap prediksi gaji adalah `Candidate Level`, `Position Function`, dan `Total Working Experience`. Ini sejalan dengan teori pasar kerja.

#### 4. Validasi, Evaluasi, Post-Processing, dan Submission
* Model dievaluasi menggunakan **K-Fold Cross-Validation** dengan metrik MAE untuk mendapatkan estimasi performa yang stabil.
* **Validasi Prediksi**: Semua prediksi dipastikan non-negatif dan `Estimated Up` selalu lebih besar atau sama dengan `Estimated Bottom`.
* **Verifikasi Final**: Model terbaik ini menghasilkan file `submission_final.csv` yang akan digunakan untuk pengiriman ke Kaggle, dengan MAE yang terbukti rendah di *test set*.

---

### 📦 Cara Menjalankan Kode

1.  **Unduh Dataset**: Pastikan file `train.csv` dan `test.csv` berada di direktori yang sama dengan notebook.
2.  **Instalasi Library**: Jalankan perintah berikut untuk menginstal semua library yang dibutuhkan:
    ```bash
    pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
    ```
3.  **Jalankan Notebook**: Buka `the Best.ipynb` di Jupyter Notebook atau Google Colab, lalu jalankan semua sel secara berurutan.

### 📄 Output

Skrip akan menghasilkan file `submission.csv` yang berisi prediksi akhir dalam format yang sesuai dengan aturan kompetisi.

| ID | Estimated Botom | Estimated Up |
| :--- | :--- | :--- |
| SKU-WENU549 | 8,281,402.65 | 8,281,402.65 |
| SKU-NUNU771 | 8,220,314.84 | 8,220,314.84 |
| ... | ... | ... |

---
*Kode dan analisis ini merupakan hasil dari proses iterasi dan perbaikan, yang menunjukkan pentingnya ketelitian dalam setiap langkah analisis data. Kami berharap ini bermanfaat.*
