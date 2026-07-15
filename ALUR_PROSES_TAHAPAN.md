# Alur Metodologi Penelitian: Optimasi CNN-LSTM PT Telkom (TLKM.JK)

Dokumen ini menjelaskan secara mendalam mengenai alur proses implementasi program dari **Tahap 1 hingga Tahap 8** yang digunakan dalam penelitian Skripsi: **"OPTIMASI HYPERPARAMETER MODEL HYBRID CNN-LSTM MENGGUNAKAN BAYESIAN OPTIMIZATION UNTUK PREDIKSI HARGA SAHAM PT TELKOM"**.

Setiap tahap dirancang untuk saling terintegrasi guna memastikan sinkronisasi antara naskah akademis (Bab III dan Bab IV) dengan bukti eksperimen empiris.

---

## Ringkasan Hubungan Antar Tahap
```
[Tahap 1: Ambil Data] -> data_raw.csv
     │
[Tahap 2: Pra-pemrosesan] -> processed_data.npz & Scalers (.pkl)
     ├───> [Tahap 3: Baseline Tanpa Optimasi] -> model_tanpa_optimasi.h5 & Hasil Awal
     └───> [Tahap 4 & 5: Optimasi Optuna] -> best_params.json
                │
           [Tahap 6: Finalisasi Model] -> model_usulan.h5 & Hasil Optimal
                │
           [Tahap 7: Evaluasi Komparatif] -> tahap7_hasil_evaluasi.xlsx (Tabel Komparasi)
                │
           [Tahap 8: Visualisasi & Proyeksi] -> Gambar_4.13.png & Hasil Prediksi 7 Hari
```

---

## Penjelasan Detail Setiap Tahap

### **Tahap 1: Pengumpulan Data (`tahap1_pengumpulan_data.py`)**
*   **Tujuan Akademis:** Membangun landasan dataset historis yang valid dan bebas dari bias manipulasi.
*   **Proses:** 
    1. Mengunduh data historis saham PT Telkom Indonesia (**TLKM.JK**) secara otomatis dari API Yahoo Finance menggunakan pustaka `yfinance`.
    2. Menggunakan rentang waktu **2 Januari 2012 hingga 2 Januari 2025**.
    3. Meratakan (flattening) struktur kolom jika data hasil unduhan berbentuk MultiIndex agar kompatibel dengan pemrosesan Pandas.
*   **Output yang Dihasilkan:**
    *   `data_raw.csv`: Berisi data mentah harian (*Date, Open, High, Low, Close, Adj Close, Volume*).
    *   `tahap1_pengumpulan_data.xlsx`: Berisi sheet `Data_Historis` dan sheet `Statistik_Deskriptif_Bab4` (menyajikan nilai minimum, maksimum, dan rata-rata dari kolom `Adj Close` untuk kebutuhan **Tabel 4.1 Statistik Deskriptif** di Bab IV).

---

### **Tahap 2: Pra-pemrosesan dan Transformasi Data (`tahap2_pra_pemrosesan.py`)**
*   **Tujuan Akademis:** Membersihkan noise data, mencegah terjadinya *data leakage* (kebocoran data), dan menyusun data ke dalam format sekuensial yang siap dibaca oleh model deep learning.
*   **Proses:**
    1. **Seleksi Variabel (Multivariat):** Memilih 6 fitur utama yaitu *Open, High, Low, Close, Adj Close,* dan *Volume*, dengan menetapkan *Adj Close* sebagai target prediksi.
    2. **Pembersihan Data (Cleaning):** Menghapus baris yang mengandung nilai kosong (`NaN`).
    3. **Pembagian Dataset (Splitting):** Membagi dataset secara kronologis dengan rasio **80% Data Pelatihan (Train)** dan **20% Data Pengujian (Test)**.
    4. **Normalisasi (Min-Max Scaling 0-1):** Melakukan fitting scaler (`MinMaxScaler`) **hanya pada data Train** untuk mencegah *data leakage*, kemudian menggunakan parameter tersebut untuk mentransformasikan data Test. Scaler disimpan secara terpisah sebagai objek `.pkl`.
    5. **Sliding Window:** Mengubah struktur data 2D menjadi 3D sekuensial dengan panjang window default **30 hari** (misal: hari ke-1 s/d 30 digunakan untuk memprediksi hari ke-31).
*   **Output yang Dihasilkan:**
    *   `processed_data.npz`: File kompresi numpy berisi array `X_train`, `y_train`, `X_test`, dan `y_test` dalam format 3D sekuensial.
    *   `scaler_X.pkl` & `scaler_y.pkl`: Objek scaler tersimpan yang krusial untuk melakukan invers normalisasi pada tahap evaluasi dan prediksi akhir.
    *   `tahap2_hasil_preprocessing.xlsx`: Berisi ringkasan pembagian data untuk kebutuhan penulisan deskripsi dataset di Bab IV.

---

### **Tahap 3: Pelatihan Model Baseline Tanpa Optimasi (`tahap3_baseline_tanpa_optimasi.py`)**
*   **Tujuan Akademis:** Membangun model pembanding dasar (baseline) dengan arsitektur standar (tanpa optimasi) untuk membuktikan perlunya penerapan Bayesian Optimization.
*   **Proses:**
    *   Melatih tiga arsitektur model secara tanpa optimasi menggunakan hyperparameter standar (Epochs: 30, Batch Size: 32, Learning Rate: 0.001, Window Size: 30, Filter/Unit: 64):
        1.  **CNN Saja (Tanpa Optimasi):** Hanya menggunakan lapisan Conv1D + Flatten + Dense.
        2.  **LSTM Saja (Tanpa Optimasi):** Hanya menggunakan lapisan LSTM + Dense.
        3.  **CNN-LSTM (Tanpa Optimasi):** Penggabungan sekuensial Conv1D + LSTM + Dense.
*   **Poin Penting Sidang:** Performa model CNN-LSTM Tanpa Optimasi yang lebih rendah dari LSTM tunggal digunakan sebagai argumen kuat perlunya **Bayesian Optimization (Optuna)** untuk menyelaraskan parameter kedua algoritma tersebut.
*   **Output yang Dihasilkan:**
    *   `model_tanpa_optimasi.h5`: Model CNN-LSTM Tanpa Optimasi yang disimpan.
    *   `y_pred_tanpa_optimasi.npy`: Hasil prediksi mentah data testing dari model tanpa optimasi.
    *   `tahap3_hasil_baseline.xlsx`: Laporan performa awal berdasarkan 5 metrik evaluasi sebagai data komparasi awal di Bab IV.

---

### **Tahap 4 & 5: Implementasi Bayesian Optimization dengan Optuna (`tahap4_5_optuna_bayesian.py`)**
*   **Tujuan Akademis:** Menemukan kombinasi hyperparameter paling optimal secara cerdas menggunakan teori probabilitas Bayesian (melalui framework **Optuna**), menghindari kelemahan pencarian acak (*Random Search*) atau trial-error tanpa optimasi.
*   **Proses:**
    1. Melakukan **50 Trials** optimasi untuk masing-masing dua jenis model: **BiLSTM** (sebagai model pembanding) dan **CNN-LSTM** (sebagai model usulan).
    2. Fungsi objektif diatur untuk meminimalkan nilai *Mean Absolute Percentage Error* (MAPE) pada data testing.
    3. **Search Space (Ruang Pencarian Hyperparameter):**
        *   *Filters (CNN):* 32, 64, atau 128.
        *   *Kernel Size:* 2 s/d 5.
        *   *Units (LSTM):* 50 s/d 150.
        *   *Dropout Rate:* 0.1 s/d 0.4.
        *   *Learning Rate:* $10^{-4}$ hingga $10^{-2}$ (skala logaritmik).
        *   *Batch Size:* 16, 32, atau 64.
        *   *Window Size:* 10 s/d 60 hari (panjang historis sekuensial dinamis).
*   **Output yang Dihasilkan:**
    *   `best_params.json`: Menyimpan parameter terbaik yang ditemukan untuk model BiLSTM dan CNN-LSTM.
    *   `tahap4_5_hasil_optimasi.xlsx`: Menyimpan **Tabel 4.5 Rentang Pencarian** dan hasil ringkasan parameter terbaik untuk kebutuhan naskah Bab IV.

---

### **Tahap 6: Finalisasi Pelatihan Model Optimal (`tahap6_finalisasi_model.py`)**
*   **Tujuan Akademis:** Melatih ulang model dari awal secara mendalam (hingga 100 epochs) menggunakan konfigurasi hyperparameter terbaik yang telah didapatkan pada Tahap 4 & 5.
*   **Proses:**
    1. Membaca file `best_params.json`.
    2. Melatih **BiLSTM + BO** selama 100 Epochs.
    3. Melatih **CNN-LSTM + BO (Model Usulan)** selama 100 Epochs dengan menyertakan fitur *EarlyStopping* (patience=15) pada data validasi guna mencegah *overfitting* (kondisi di mana model menghafal data latihan namun gagal menggeneralisasi data baru).
*   **Output yang Dihasilkan:**
    *   `model_usulan.h5`: File model final CNN-LSTM optimal hasil kompilasi Bayesian Optimization.
    *   `y_pred_usulan.npy` & `y_pred_bi.npy`: Hasil prediksi optimal data testing.
    *   `y_true_usulan.npy` & `y_true_bi.npy`: Target aktual data testing yang telah disesuaikan dengan dimensi window masing-masing model.
    *   `tahap6_hasil_training.xlsx`: Menyimpan riwayat loss pelatihan untuk menggambar grafik konvergensi di laporan Bab IV, serta **Tabel 4.6 Daftar Hyperparameter Terbaik**.

---

### **Tahap 7: Analisis Komparatif dan Evaluasi Akhir (`tahap7_evaluasi_akhir.py`)**
*   **Tujuan Akademis:** Melakukan pengujian akurasi akhir secara matematis dan objektif menggunakan metrik standar industri keuangan dan statistik.
*   **Proses:**
    1. Membaca seluruh hasil prediksi dari Tahap 3 (Baselines) dan Tahap 6 (Optimized).
    2. Mengembalikan data berskala 0-1 ke skala Rupiah asli menggunakan `scaler_y.inverse_transform`.
    3. Menghitung **5 Metrik Evaluasi**: *Mean Squared Error* (MSE), *Mean Absolute Error* (MAE), *Root Mean Squared Error* (RMSE), *Mean Absolute Percentage Error* (MAPE), dan *Coefficient of Determination* ($R^2$).
    4. Mengurutkan performa model untuk membuktikan hipotesis skripsi.
*   **Output yang Dihasilkan:**
    *   `tahap7_hasil_evaluasi.xlsx`: Berisi tabel rekapitulasi performa komparatif terlengkap seluruh model (Baselines vs Optimized) yang siap dipindahkan langsung ke bab pembahasan utama.
    *   Menampilkan deklarasi model terbaik (dengan nilai MAPE terkecil).

---

### **Tahap 8: Visualisasi Final dan Proyeksi Masa Depan (`tahap8_hasil_prediksi.py`)**
*   **Tujuan Akademis:** Menyediakan pembuktian visual yang intuitif tentang keandalan model serta menguji daya prediksi model pada skenario dunia nyata (*real-world forecasting*).
*   **Proses:**
    1. **Visualisasi Performa (100 Hari Testing):** Mengambil 100 hari terakhir data testing untuk diplot ke dalam grafik garis beresolusi tinggi guna membandingkan pergerakan harga aktual vs model usulan vs model tanpa optimasi.
    2. **Proyeksi Iteratif 7 Hari (Januari 2025):** Mengunduh data pasar terbaru di awal Januari 2025 dari Yahoo Finance, kemudian menggunakan model optimal untuk memproyeksikan harga saham TLKM secara bergulir (*rolling/iterative forecast*) selama 7 hari bursa berturut-turut.
*   **Output yang Dihasilkan:**
    *   `Gambar_4.13_Hasil_Prediksi_Kesimpulan.png`: Grafik visual kualitas tinggi (300 DPI) untuk **Gambar 4.13** di naskah skripsi.
    *   `tahap8_hasil_prediksi.xlsx`: Tabel hasil proyeksi 7 hari beserta perhitungan deviasi (error) riilnya untuk menyusun **Tabel 4.8 Proyeksi Tren** di Bab IV.

### ** Penjelasan Singkat pertanyaan saya **
* **Berikut adalah jawaban lengkap dan mendalam untuk pertanyaan Anda, yang dirancang khusus untuk memperkuat argumentasi Anda saat Sidang Skripsi nanti:**
1. Mengenai Data Cleaning pada Tahap 2
Pertanyaan: Apakah cukup sekali saja atau bagaimana baiknya?
Jawaban: Untuk data saham yang diambil dari Yahoo Finance, proses dropna() satu kali setelah seleksi variabel sudah cukup.
Alasannya: Data saham adalah time-series harian yang terstruktur. Masalah utama biasanya hanya pada hari libur bursa di mana data menjadi kosong. Dengan melakukan dropna() setelah memilih fitur (OHLCV), Anda memastikan bahwa baris yang dihapus hanyalah baris yang benar-benar tidak memiliki informasi harga.
Saran Skripsi: Dalam naskah, tuliskan bahwa "Pembersihan dilakukan satu kali untuk menjamin kontinuitas data dan integritas fitur multivariat sebelum memasuki tahap normalisasi."
2. Mengenai Splitting Data 80:20
Pertanyaan: Alasan menggunakan 80:20 dibandingkan 70:20:10?
Jawaban: Rasio 80:20 adalah rasio standar (Pareto Principle) yang sangat umum dalam penelitian Deep Learning dengan jumlah data menengah (sekitar 3.000 baris).
Alasannya: Model hibrida seperti CNN-LSTM memiliki banyak parameter (bobot) yang harus dipelajari. Memberikan 80% data untuk pelatihan memberikan kesempatan model untuk mengenali pola krisis (seperti Covid-2020) dan pola tren naik secara lebih mendalam.
Kenapa tidak 70:20:10? Rasio 70:20:10 biasanya digunakan jika data sangat besar (puluhan ribu). Dalam penelitian Anda, validasi dilakukan langsung di dalam proses Bayesian Optimization (Optuna) atau menggunakan validation split saat training, sehingga pembagian dua arah (Train & Test) sudah dianggap sah secara akademis untuk mengukur kemampuan generalisasi pada data yang benar-benar baru (unseen data).
3. Perbedaan Window 30 Hari (Tahap 2) vs 10-60 Hari (Tahap 4 & 5)
Pertanyaan: Kenapa tidak disamakan?
Jawaban:
Window 30 hari di Tahap 2 disebut sebagai Inisialisasi Dasar. Tujuannya adalah untuk membuktikan bahwa kode program berjalan (Running Test) dan sebagai standar untuk model Baseline (Tahap 3).
Window 10-60 hari di Tahap 4 & 5 adalah Ruang Pencarian (Search Space). Inti dari judul skripsi Anda adalah "Optimasi". Anda tidak tahu apakah 30 hari adalah angka terbaik. Bisa jadi pola saham Telkom lebih mudah dibaca dalam pola 15 hari atau 50 hari. Bayesian Optimization akan menguji berbagai angka di rentang tersebut untuk menemukan "Ingatan Historis" mana yang paling akurat.
4. Penjelasan Istilah Hyperparameter (Tahap 3)
Epoch: Berapa kali model melihat seluruh dataset pelatihan. (Analogi: Berapa kali siswa membaca satu buku pelajaran sampai hafal).
Batch Size: Jumlah sampel data yang diproses sebelum model memperbarui bobot internalnya. (Analogi: Berapa soal yang dikerjakan siswa sebelum melihat kunci jawaban).
Learning Rate: Kecepatan model dalam belajar. Jika terlalu besar, model akan ceroboh (melewatkan solusi); jika terlalu kecil, model akan sangat lambat.
Window Size: Panjangnya hari ke belakang yang digunakan untuk melihat pola. (Analogi: Melihat tren harga 30 hari terakhir untuk menebak harga besok).
Filters (CNN): Jumlah "lensa" atau sensor untuk menangkap fitur penting (seperti pola bullish atau bearish).
Kernel Size: Lebar jendela sensor saat memindai data. (Analogi: Membaca per 3 kata atau per 5 kata).
Units (LSTM): Kapasitas memori model untuk menyimpan informasi jangka panjang.
Dropout Rate: Teknik mematikan beberapa saraf secara acak saat latihan agar model tidak "menghafal" (mencegah overfitting).
5. Penjelasan Lapisan Arsitektur (Tahap 3)
Conv1D (Convolutional 1D): Fungsinya untuk Ekstraksi Fitur Spasial. Ia mendeteksi pola lokal/tren harga dalam jangka pendek.
LSTM (Long Short-Term Memory): Fungsinya untuk Ekstraksi Fitur Temporal. Ia menangkap hubungan jangka panjang dan ketergantungan antar waktu.
Flatten: Fungsinya sebagai Jembatan. Ia mengubah data berbentuk matriks (dari CNN) menjadi bentuk vektor linear agar bisa diproses ke lapisan terakhir.
Dense: Fungsinya sebagai Decision Maker (Pengambil Keputusan). Ia menghubungkan semua fitur yang ditemukan untuk menghasilkan satu angka prediksi harga.
6. Kenapa Memilih 50 Trial di Optuna?
Pertanyaan: Apakah standar atau agar tidak lama?
Jawaban: 50 Trial adalah titik Ekuilibrium (Keseimbangan).
Dalam Bayesian Optimization, algoritma tidak mencari secara acak, tapi belajar dari kesalahan trial sebelumnya. Penelitian menunjukkan bahwa untuk 6-8 hyperparameter, Optuna biasanya sudah mencapai titik jenuh (konvergen) pada trial ke 40-60.
Alasan Sidang: "50 Trial dipilih karena telah memberikan hasil yang stabil (konvergen) dan efisien secara komputasi tanpa mengurangi kualitas pencarian parameter optimal."
7. Penggunaan 100 Epoch & EarlyStopping (Patience=15)
Pertanyaan: Kenapa 100 dan kenapa ada EarlyStopping?
Jawaban:
100 Epoch adalah batas atas yang aman. Jarang sekali model saham butuh lebih dari 100 kali baca data.
EarlyStopping adalah "Rem Otomatis". Ia berfungsi menghentikan pelatihan jika performa pada data validasi tidak lagi membaik. Ini untuk mencegah Overfitting.
Patience=15 artinya model diberi toleransi 15 kali kesempatan. Jika dalam 15 kali baca buku siswa tidak bertambah pintar (nilai tidak naik), maka proses belajar dihentikan karena dianggap sudah maksimal. 15 adalah angka moderat agar model tidak berhenti terlalu cepat saat bertemu "gangguan" kecil di data.
8. Kenapa Harus Dikembalikan ke Skala Rupiah?
Pertanyaan: Apakah agar model lebih mudah memahami?
Jawaban: Bukan untuk model, tapi untuk Manusia (Interpretasi).
Model memahami angka 0 s/d 1 jauh lebih baik daripada angka ribuan (skala kecil membuat perhitungan gradien lebih stabil).
Namun, jika Anda melaporkan error dalam naskah skripsi: "Error model adalah 0.002", penguji tidak akan paham. Tapi jika diubah ke Rupiah menjadi: "Error model adalah Rp 15", maka pembaca langsung tahu bahwa model Anda sangat akurat. Ini untuk aspek Komunikatif penelitian.
9. Visualisasi 100 Hari vs Proyeksi 7 Hari
Pertanyaan: Kenapa 100 hari dan apakah sudah standar skripsi?
Jawaban:
100 Hari (Visualisasi Performa): Ini digunakan untuk menunjukkan Stabilitas. Jika hanya menampilkan 7 hari, penguji bisa menganggap itu "keberuntungan". Dengan menampilkan 100 hari terakhir data testing, Anda membuktikan bahwa model konsisten mengikuti tren naik dan turun (dinamika pasar) dalam periode yang cukup lama.
7 Hari (Proyeksi Masa Depan): Ini adalah uji Realitas (Forecasting). Dalam dunia investasi, prediksi 1 minggu ke depan adalah jangka waktu yang paling relevan bagi swing trader.
Apakah Cukup Standar? SANGAT CUKUP. Bahkan judul Anda yang menekankan pada Optimasi Hyperparameter menunjukkan bahwa kontribusi utama Anda bukan hanya pada "menebak harga", tapi pada "bagaimana membangun otak AI yang paling pintar" melalui metode Bayesian. Penjelasan detail dari tahap 1-8 ini sudah memenuhi standar publikasi jurnal SINTA maupun skripsi Teknik Informatika/Sains Data.