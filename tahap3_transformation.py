import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

def transform_data():
    """
    Fungsi Transformasi Data Lengkap (Fase KDD: Data Transformation)
    Mengubah data bersih menjadi format sekuensial 3D (Sliding Window) & Normalisasi Fitur.
    Menghasilkan: processed_data.npz, scaler_X.pkl, scaler_y.pkl, tahap3_hasil_baseline.xlsx
    """
    print("\n" + "="*70)
    print(" [TAHAP 3: TRANSFORMASI DATA MULTIVARIAT SEKUENSIAL] ")
    print(" [FASE KDD: DATA TRANSFORMATION (TRANSFORMASI DATA)] ")
    print("="*70)
    
    print("\n>>> PENJELASAN METODOLOGI (UNTUK BAB IV):")
    print("    - Pada tahap ini, data bersih ditransformasikan ke skala yang seragam menggunakan Min-Max Scaling [0, 1].")
    print("    - Normalisasi dilakukan terpisah untuk mencegah kebocoran data (Data Leakage): fit hanya pada data Latih.")
    print("    - Data kemudian ditransformasikan ke format 3D [samples, window_size, features] menggunakan Sliding Window.")
    print("    - Panjang Sliding Window default ditetapkan sebesar 30 hari untuk representasi temporal.")
    
    if not os.path.exists("tahap2_hasil_preprocessing.xlsx"):
        print("Error: File tahap2_hasil_preprocessing.xlsx tidak ditemukan! Jalankan Tahap 2 terlebih dahulu.")
        return
        
    # 1. Memuat data hasil preprocessing
    train_df = pd.read_excel("tahap2_hasil_preprocessing.xlsx", sheet_name="Train_Cleaned")
    test_df = pd.read_excel("tahap2_hasil_preprocessing.xlsx", sheet_name="Test_Cleaned")
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # 2. NORMALISASI (Min-Max Scaling)
    print("\n>>> 1. Memulai Normalisasi Data menggunakan Min-Max Scaling [0 s/d 1]...")
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Fit & Transform hanya pada data latih
    train_X_scaled = scaler_X.fit_transform(train_df[feature_cols].values)
    train_y_scaled = scaler_y.fit_transform(train_df[['Adj Close']].values)
    
    # Transform data uji menggunakan parameter dari data latih
    test_X_scaled = scaler_X.transform(test_df[feature_cols].values)
    test_y_scaled = scaler_y.transform(test_df[['Adj Close']].values)
    
    # Simpan objek scaler untuk inverse transform di tahap evaluasi
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("       * Objek scaler berhasil disimpan: scaler_X.pkl & scaler_y.pkl")

    # 3. SLIDING WINDOW TRANSFORMATION (Window Size = 30)
    print("\n>>> 2. Mengubah Data 2D menjadi Sekuensial 3D (Sliding Window)...")
    window_size = 30
    
    def create_dataset(X_data, y_data, window):
        X, y = [], []
        for i in range(len(X_data) - window):
            X.append(X_data[i:(i + window), :])
            y.append(y_data[i + window, 0])
        return np.array(X), np.array(y)
        
    X_train, y_train = create_dataset(train_X_scaled, train_y_scaled, window_size)
    X_test, y_test = create_dataset(test_X_scaled, test_y_scaled, window_size)
    
    # Mengambil nilai target asli (Rupiah) yang sejajar dengan hasil windowing
    y_test_original = test_df['Adj Close'].values[window_size:]
    
    # 4. PENYIMPANAN DATA AKHIR (.NPZ)
    print("\n>>> 3. Menyimpan hasil transformasi ke processed_data.npz...")
    np.savez("processed_data.npz", 
             X_train=X_train, y_train=y_train,   
             X_test=X_test, y_test=y_test,       
             train_X_scaled=train_X_scaled,      
             train_y_scaled=train_y_scaled,      
             test_X_scaled=test_X_scaled,        
             test_y_scaled=test_y_scaled)
    
    print(f"       * Dimensi Awal Train Set: {train_df.shape}  | Hasil Transformasi 3D (X_train): {X_train.shape}")
    print(f"       * Dimensi Awal Test Set : {test_df.shape}   | Hasil Transformasi 3D (X_test) : {X_test.shape}")
    print("       * File processed_data.npz berhasil diperbarui.")

    # 5. EKSPOR SUMMARY REKAP KE EXCEL UNTUK TABEL BAB IV
    print("\n>>> 4. Mengekspor laporan rekapitulasi data transformasi ke Excel...")
    df_transform = pd.DataFrame([
        {
            'Dataset': 'Data Pelatihan (Train)', 
            'Baris Asli': len(train_df), 
            'Window Size': window_size, 
            'Dimensi 3D Input (Samples, Window, Features)': str(X_train.shape), 
            'Dimensi Target Output': str(y_train.shape)
        },
        {
            'Dataset': 'Data Pengujian (Test)', 
            'Baris Asli': len(test_df), 
            'Window Size': window_size, 
            'Dimensi 3D Input (Samples, Window, Features)': str(X_test.shape), 
            'Dimensi Target Output': str(y_test.shape)
        }
    ])
    
    # Tetap disimpan sebagai tahap3_hasil_baseline.xlsx agar tidak memecah konfigurasi dashboard app.py
    df_transform.to_excel("tahap3_hasil_baseline.xlsx", index=False)
    
    print("\n" + "="*110)
    print("REKAPITULASI HASIL DATA TRANSFORMATION (TABEL KDD TAHAP 3)")
    print("="*110)
    print(df_transform.to_string(index=False))
    print("="*110)
    print(">>> Transformasi selesai. Rekapitulasi disimpan di: tahap3_hasil_baseline.xlsx")
    print("="*70 + "\n")

if __name__ == "__main__":
    transform_data()
