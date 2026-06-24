import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data():
    """
    Fungsi Pra-pemrosesan Lengkap: Seleksi -> Cleaning -> Splitting -> Scaling -> Windowing
    Menghasilkan: processed_data.npz, scaler_X.pkl, scaler_y.pkl
    """
    print("\n" + "="*60)
    print("[TAHAP 2: PRA-PEMROSESAN DAN TRANSFORMASI DATA]")
    print("="*60)
    
    # 1. Membaca data mentah
    print("1. Membaca data mentah (data_raw.csv)...")
    df_raw = pd.read_csv("data_raw.csv", index_col=0)
    
    # 2. SELEKSI VARIABEL
    print("2. Seleksi variabel multivariat (O, H, L, C, AC, V)...")
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_selected = df_raw[feature_cols].copy()
    
    # 3. PEMBERSIHAN DATA
    print("3. Menangani nilai kosong (Cleaning)...")
    df_selected.dropna(inplace=True)
    df_selected['Target'] = df_selected['Adj Close']
    
    # 4. PEMBAGIAN DATASET (80% Train, 20% Test)
    print("4. Melakukan splitting data (80% Train, 20% Test)...")
    split_idx = int(len(df_selected) * 0.8)
    train_df = df_selected.iloc[:split_idx]
    test_df = df_selected.iloc[split_idx:]

    # 5. NORMALISASI (Min-Max Scaling)
    # --- DI SINI FILE .PKL DIHASILKAN ---
    print("5. Normalisasi data dengan Min-Max Scaling (0-1)...")
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Fit hanya pada data TRAIN untuk menghindari Data Leakage
    train_X_scaled = scaler_X.fit_transform(train_df[feature_cols].values)
    train_y_scaled = scaler_y.fit_transform(train_df[['Target']].values)
    
    # Transform data TEST menggunakan parameter dari data TRAIN
    test_X_scaled = scaler_X.transform(test_df[feature_cols].values)
    test_y_scaled = scaler_y.transform(test_df[['Target']].values)

    # Simpan Scaler (PENTING untuk Tahap 8 nanti)
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("   >>> Berhasil menyimpan: scaler_X.pkl dan scaler_y.pkl")

    # 6. TRANSFORMASI SLIDING WINDOW
    print("6. Transformasi data ke format sliding window (Window: 30)...")
    def create_dataset(X_data, y_data, window=30):
        X, y = [], []
        for i in range(len(X_data) - window):
            X.append(X_data[i:(i + window), :])
            y.append(y_data[i + window, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_X_scaled, train_y_scaled, window=30)
    X_test, y_test = create_dataset(test_X_scaled, test_y_scaled, window=30)

    # 7. PENYIMPANAN DATA AKHIR (.NPZ)
    print("7. Menyimpan hasil akhir ke file processed_data.npz...")
    np.savez("processed_data.npz", 
             X_train=X_train, y_train=y_train,   # Data 3D (Window 30)
             X_test=X_test, y_test=y_test,       # Data 3D (Window 30)
             train_X_scaled=train_X_scaled,      # Data 2D (PENTING UNTUK TAHAP 6)
             train_y_scaled=train_y_scaled,      # Data 2D
             test_X_scaled=test_X_scaled,        # Data 2D
             test_y_scaled=test_y_scaled)        # Data 2D
    print("   >>> Berhasil menyimpan: processed_data.npz dengan semua key.")

    # 8. EKSPOR STATISTIK UNTUK BAB IV
    print("8. Mengekspor statistik ke Excel...")
    df_stats = pd.DataFrame({
        'Keterangan': ['Baris Data Awal', 'Setelah Cleaning', 'Data Train (80%)', 'Data Test (20%)', 'X_train Shape', 'X_test Shape'],
        'Nilai': [len(df_raw), len(df_selected), len(train_df), len(test_df), str(X_train.shape), str(X_test.shape)]
    })

    with pd.ExcelWriter("tahap2_hasil_preprocessing.xlsx") as writer:
        df_selected.to_excel(writer, sheet_name='Data_Cleaned')
        df_stats.to_excel(writer, sheet_name='Statistik_Dataset_Bab4', index=False)
    
    print("\n" + "="*60)
    print(" [TAHAP 2 SELESAI] ")
    print(" Semua file pendukung penelitian telah dibuat.")
    print("="*60 + "\n")

if __name__ == "__main__":
    preprocess_data()