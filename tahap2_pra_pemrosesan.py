import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data():
    """
    Fungsi Pra-pemrosesan Lengkap: Seleksi -> Cleaning -> Splitting -> Scaling -> Windowing
    Menghasilkan: processed_data.npz, scaler_X.pkl, scaler_y.pkl
    """
    print("\n" + "="*70)
    print(" [TAHAP 2: PRA-PEMROSESAN DAN TRANSFORMASI DATA] ")
    print(" [FASE KDD: DATA PREPROCESSING, CLEANING, & TRANSFORMATION] ")
    print("="*70)
    
    # 1. Membaca data mentah
    print("1. Membaca data mentah (data_raw.csv)...")
    df_raw = pd.read_csv("data_raw.csv", index_col=0)
    
    # 2. SELEKSI VARIABEL
    print("2. Menyeleksi variabel multivariat (OHLCV + Adj Close)...")
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_selected = df_raw[feature_cols].copy()
    
    # 3. PEMBERSIHAN DATA
    print("3. Proses Pembersihan Data (Data Cleaning):")
    print("   - Melakukan verifikasi kelengkapan data bursa harian (mengecek nilai kosong/NaN).")
    print("   - Jumlah percobaan pemeriksaan: 1 kali pemeriksaan menyeluruh pada dataset.")
    initial_len = len(df_selected)
    df_selected.dropna(inplace=True)
    cleaned_len = len(df_selected)
    removed_rows = initial_len - cleaned_len
    print(f"   - Hasil Percobaan: Ditemukan {removed_rows} baris bernilai kosong (NaN) dari total {initial_len} baris data.")
    print(f"   - Data dinyatakan 100% BERSIH setelah 1 kali pembersihan. Baris tersisa: {cleaned_len} baris.")
    df_selected['Target'] = df_selected['Adj Close']
    
    # 4. PEMBAGIAN DATASET (80% Train, 20% Test)
    print("4. Pembagian Dataset secara Kronologis (Data Splitting):")
    print("   - Rasio pembagian: 80% Data Pelatihan (Train) dan 20% Data Pengujian (Test) untuk menjaga sekuensial waktu.")
    split_idx = int(len(df_selected) * 0.8)
    train_df = df_selected.iloc[:split_idx]
    test_df = df_selected.iloc[split_idx:]
    print(f"   - Jumlah data pelatihan (sebelum windowing) : {len(train_df)} baris")
    print(f"   - Jumlah data pengujian (sebelum windowing)  : {len(test_df)} baris")
 
    # 5. NORMALISASI (Min-Max Scaling)
    print("5. Transformasi Data melalui Normalisasi (Min-Max Scaling 0 s/d 1):")
    print("   - Catatan Akademis: Normalisasi di-fit HANYA menggunakan data Train untuk mencegah Data Leakage.")
    print("   - Parameter scaler kemudian digunakan untuk mentransformasikan data Test.")
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
 
    train_X_scaled = scaler_X.fit_transform(train_df[feature_cols].values)
    train_y_scaled = scaler_y.fit_transform(train_df[['Target']].values)
    
    test_X_scaled = scaler_X.transform(test_df[feature_cols].values)
    test_y_scaled = scaler_y.transform(test_df[['Target']].values)
 
    # Simpan Scaler (PENTING untuk Tahap 8 nanti)
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("   >>> Berhasil menyimpan file scaler: scaler_X.pkl dan scaler_y.pkl")
 
    # 6. TRANSFORMASI SLIDING WINDOW
    print("6. Transformasi Data Sekuensial menggunakan Sliding Window:")
    print("   - Panjang Window (Ingatan Historis): 30 hari default untuk model baseline.")
    def create_dataset(X_data, y_data, window=30):
        X, y = [], []
        for i in range(len(X_data) - window):
            X.append(X_data[i:(i + window), :])
            y.append(y_data[i + window, 0])
        return np.array(X), np.array(y)
 
    X_train, y_train = create_dataset(train_X_scaled, train_y_scaled, window=30)
    X_test, y_test = create_dataset(test_X_scaled, test_y_scaled, window=30)
    print(f"   - Struktur akhir input pelatihan (X_train Shape) : {X_train.shape}")
    print(f"   - Struktur akhir input pengujian (X_test Shape)   : {X_test.shape}")
 
    # 7. PENYIMPANAN DATA AKHIR (.NPZ)
    print("7. Menyimpan dataset akhir yang ter-preprocessing (processed_data.npz)...")
    np.savez("processed_data.npz", 
             X_train=X_train, y_train=y_train,   
             X_test=X_test, y_test=y_test,       
             train_X_scaled=train_X_scaled,      
             train_y_scaled=train_y_scaled,      
             test_X_scaled=test_X_scaled,        
             test_y_scaled=test_y_scaled)
    print("   >>> File processed_data.npz berhasil dibuat.")
 
    # 8. EKSPOR STATISTIK UNTUK BAB IV
    print("8. Mengekspor laporan preprocessing ke format Excel...")
    df_stats = pd.DataFrame({
        'Tahap Preprocessing': ['1. Baris Data Mentah', '2. Setelah Pembersihan NaN', '3. Alokasi Data Pelatihan (80%)', '4. Alokasi Data Pengujian (20%)', '5. Dimensi Array X_train (3D)', '6. Dimensi Array X_test (3D)'],
        'Hasil / Nilai': [len(df_raw), len(df_selected), len(train_df), len(test_df), str(X_train.shape), str(X_test.shape)]
    })
 
    with pd.ExcelWriter("tahap2_hasil_preprocessing.xlsx") as writer:
        df_selected.to_excel(writer, sheet_name='Data_Cleaned')
        df_stats.to_excel(writer, sheet_name='Statistik_Dataset_Bab4', index=False)
    
    print("\n" + "="*80)
    print("TAMPILAN REKAPITULASI UNTUK TABEL 4.2 DAN 4.3 PREPROCESSING")
    print("="*80)
    print(df_stats.to_string(index=False))
    print("="*80)
    print(">>> Preprocessing selesai. Tabel pendukung Bab IV disimpan di: tahap2_hasil_preprocessing.xlsx")
    print("="*70 + "\n")

if __name__ == "__main__":
    preprocess_data()