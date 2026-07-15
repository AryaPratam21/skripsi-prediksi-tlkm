import pandas as pd
import numpy as np

def preprocess_data():
    """
    Fungsi Preprocessing Lengkap (Fase KDD: Seleksi & Cleaning & Splitting)
    Menghasilkan data bersih siap transformasi: tahap2_hasil_preprocessing.xlsx
    """
    print("\n" + "="*70)
    print(" [TAHAP 2: PRA-PEMROSESAN DATA (CLEANING & SPLITTING)] ")
    print(" [FASE KDD: PREPROCESSING (PRA-PEMROSESAN DATA)] ")
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
    initial_len = len(df_selected)
    df_selected.dropna(inplace=True)
    cleaned_len = len(df_selected)
    removed_rows = initial_len - cleaned_len
    print(f"   - Hasil Pemeriksaan: Ditemukan {removed_rows} baris bernilai kosong (NaN) dari total {initial_len} baris data.")
    print(f"   - Data dinyatakan 100% BERSIH setelah pembersihan. Baris tersisa: {cleaned_len} baris.")
    
    # 4. PEMBAGIAN DATASET (80% Train, 20% Test)
    print("4. Pembagian Dataset secara Kronologis (Data Splitting):")
    print("   - Rasio pembagian: 80% Data Pelatihan (Train) dan 20% Data Pengujian (Test) untuk menjaga sekuensial waktu.")
    split_idx = int(len(df_selected) * 0.8)
    train_df = df_selected.iloc[:split_idx].copy()
    test_df = df_selected.iloc[split_idx:].copy()
    print(f"   - Jumlah data pelatihan (sebelum transformasi) : {len(train_df)} baris")
    print(f"   - Jumlah data pengujian (sebelum transformasi)  : {len(test_df)} baris")
 
    # 5. EKSPOR DATA BERSIH KE EXCEL
    print("5. Mengekspor hasil pembersihan dan splitting ke Excel...")
    df_stats = pd.DataFrame({
        'Tahap Preprocessing': ['1. Baris Data Mentah', '2. Setelah Pembersihan NaN', '3. Alokasi Data Pelatihan (80%)', '4. Alokasi Data Pengujian (20%)'],
        'Hasil / Nilai': [len(df_raw), len(df_selected), len(train_df), len(test_df)]
    })
 
    with pd.ExcelWriter("tahap2_hasil_preprocessing.xlsx") as writer:
        train_df.to_excel(writer, sheet_name='Train_Cleaned', index=True, index_label="Date")
        test_df.to_excel(writer, sheet_name='Test_Cleaned', index=True, index_label="Date")
        df_stats.to_excel(writer, sheet_name='Statistik_Dataset_Bab4', index=False)
    
    print("\n" + "="*80)
    print("TAMPILAN REKAPITULASI UNTUK TABEL PREPROCESSING")
    print("="*80)
    print(df_stats.to_string(index=False))
    print("="*80)
    print(">>> Preprocessing selesai. Data pelatihan & pengujian bersih disimpan di: tahap2_hasil_preprocessing.xlsx")
    print("="*70 + "\n")

if __name__ == "__main__":
    preprocess_data()