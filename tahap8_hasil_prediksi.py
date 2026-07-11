import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import yfinance as yf
import os
import json
from tensorflow.keras.models import load_model

# Konfigurasi Lingkungan agar log bersih
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_prediction_final():
    print("\n" + "="*70)
    print(" [TAHAP 8: VISUALISASI FINAL SESUAI LAPORAN BAB 4] ")
    print(" [FASE KDD: KNOWLEDGE REPRESENTATION (PENYAJIAN PENGETAHUAN)] ")
    print("="*70)
    
    print("\n>>> PENJELASAN VISUALISASI DAN PROYEKSI (UNTUK BAB IV):")
    print("    1. Visualisasi Keseluruhan (Gambar 4.13): memplot pergerakan harga aktual (Adj Close) bursa")
    print("       melawan prediksi model usulan (CNN-LSTM + BO) dan model baseline tanpa optimasi selama 100 hari terakhir data testing.")
    print("    2. Proyeksi 7 Hari Kerja (Tabel 4.8): pengujian simulasi trading real-world sekuensial.")
    print("       - Prosedur Proyeksi Iteratif: Prediksi hari ke-t digunakan sebagai bagian dari data input historis")
    print("         untuk memprediksi harga hari ke-t+1 secara berantai/bergulir (rolling forecast).")
    print("       - Proyeksi 7 hari ini HANYA menampilkan model usulan terbaik Anda sesuai permintaan.")
    
    # --- 1. LOAD ASSET (Model & Scaler) ---
    if not os.path.exists('scaler_X.pkl') or not os.path.exists('scaler_y.pkl'):
        print("Error: File scaler (.pkl) tidak ditemukan! Jalankan Tahap 2.")
        return

    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    
    # Load Model Usulan (Hasil Tahap 6)
    if not os.path.exists("model_usulan.h5"):
        print("Error: model_usulan.h5 tidak ditemukan!")
        return
    m_usulan = load_model("model_usulan.h5", compile=False)
    
    # Load Model Tanpa Optimasi (Hasil Tahap 3)
    if not os.path.exists("model_tanpa_optimasi.h5"):
        print("Error: model_tanpa_optimasi.h5 tidak ditemukan!")
        return
    m_tanpa_optimasi = load_model("model_tanpa_optimasi.h5", compile=False)
    print("\n>>> 1. Memuat model usulan (.h5) dan model pembanding baseline...")
    print("       * Model Usulan & Model Tanpa Optimasi Berhasil Dimuat.")

    # Load Hyperparameters untuk Window Size
    data = np.load("processed_data.npz")
    with open('best_params.json', 'r') as f:
        best = json.load(f)
    
    ws_usulan = best['cnn_lstm']['window_size']
    ws_tanpa_optimasi = 30 # Standar baseline tanpa optimasi

    # --- 2. GENERATE GAMBAR 4.13 (PERFORMA KESELURUHAN - 100 HARI TESTING) ---
    print("\n>>> 2. Membuat Gambar 4.13 (Grafik Perbandingan Performa Keseluruhan)...")
    ts_X_raw = data['test_X_scaled']
    ts_y_raw = data['test_y_scaled']

    def get_preds_all(model, ws, X_raw):
        X_tmp = []
        for i in range(len(X_raw) - ws):
            X_tmp.append(X_raw[i:(i + ws), :])
        preds = model.predict(np.array(X_tmp), verbose=0)
        return scaler_y.inverse_transform(preds).flatten()

    y_pred_usulan_all = get_preds_all(m_usulan, ws_usulan, ts_X_raw)
    y_pred_tanpa_optimasi_all = get_preds_all(m_tanpa_optimasi, ws_tanpa_optimasi, ts_X_raw)
    
    # Sinkronisasi panjang data aktual
    min_len = min(len(y_pred_usulan_all), len(y_pred_tanpa_optimasi_all))
    y_act_all = scaler_y.inverse_transform(ts_y_raw[-min_len:].reshape(-1,1)).flatten()
    
    plt.figure(figsize=(15, 8))
    plt.plot(y_act_all[-100:], color='black', label='Harga Aktual (Adj Close)', linewidth=2.5)
    plt.plot(y_pred_usulan_all[-min_len:][-100:], color='red', linestyle='--', label='CNN-LSTM + BO (USULAN)', linewidth=2)
    plt.plot(y_pred_tanpa_optimasi_all[-min_len:][-100:], color='green', linestyle=':', label='CNN-LSTM (Tanpa Optimasi)', linewidth=2)
    
    plt.title('Gambar 4.13: Hasil Prediksi Dan Kesimpulan Akhir (Performa Testing)', fontsize=14)
    plt.xlabel('Hari (Data Testing)'); plt.ylabel('Harga Saham (Rp)')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.savefig("Gambar_4.13_Hasil_Prediksi_Kesimpulan.png", dpi=300)
    print("       * Gambar 4.13 Berhasil Disimpan di folder proyek: Gambar_4.13_Hasil_Prediksi_Kesimpulan.png")

    # --- 3. MEKANISME PROYEKSI 7 HARI (UNTUK TABEL 4.8) ---
    print("\n>>> 3. Memulai Proses Proyeksi Iteratif 7 Hari Kerja (Januari 2025)...")
    print("       * Mengunduh data pasar terbaru Telkom (TLKM.JK) dari Yahoo Finance...")
    df_full = yf.download("TLKM.JK", start="2024-09-01", end="2025-01-20", progress=False, auto_adjust=False)
    if isinstance(df_full.columns, pd.MultiIndex): df_full.columns = df_full.columns.get_level_values(0)
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_target = df_full.loc["2025-01-02":"2025-01-10"].copy()
    dates = df_target.index.strftime('%Y-%m-%d')
    actual_7 = df_target['Adj Close'].values

    # Simulasi Langkah demi Langkah Proyeksi Iteratif
    print("\n      --- Langkah demi Langkah Proyeksi Iteratif (CNN-LSTM + BO Usulan) ---")
    history = df_full[df_full.index < "2025-01-02"].tail(ws_usulan)[feature_cols].values
    curr_win = scaler_X.transform(history).reshape(1, ws_usulan, -1)
    
    preds_usulan = []
    mape_list = []
    
    for i in range(len(dates)):
        date_str = dates[i]
        act_val = actual_7[i]
        
        # Prediksi harga hari ke-t
        p = m_usulan.predict(curr_win, verbose=0)
        p_rupiah = float(scaler_y.inverse_transform(p.reshape(-1, 1))[0, 0])
        preds_usulan.append(p_rupiah)
        
        # Hitung Deviasi & Persentase Error
        dev = abs(act_val - p_rupiah)
        err_pct = (dev / act_val) * 100
        mape_list.append(err_pct)
        
        print(f"        * Hari ke-{i+1} ({date_str}):")
        print(f"          - Menggunakan input window: {curr_win.shape}")
        print(f"          - Hasil Prediksi = Rp {p_rupiah:,.2f} | Harga Asli = Rp {act_val:,.2f} | Selisih = Rp {dev:,.2f} ({err_pct:.4f}%)")
        
        # Umpan Balik (Rolling/Shift Window): Masukkan data aktual hari ke-t untuk melangkah ke hari t+1
        print(f"          - Menggeser window dan memasukkan data hari ini ke dalam sejarah input untuk hari berikutnya...")
        new_row = scaler_X.transform(df_target.iloc[[i]][feature_cols].values)
        curr_win = np.append(curr_win[:, 1:, :], new_row.reshape(1, 1, -1), axis=1)

    y_pred_7_usulan = np.array(preds_usulan)
    avg_mape_7 = np.mean(mape_list)

    # --- 4. SIMPAN EXCEL TABEL 4.8 ---
    df_48 = pd.DataFrame({
        'Tanggal': dates,
        'Harga Asli (Rp)': actual_7,
        'Harga Prediksi Usulan (Rp)': y_pred_7_usulan
    })
    
    file_excel = "tahap8_hasil_prediksi.xlsx"
    df_48.to_excel("tahap8_hasil_prediksi.xlsx", sheet_name='Tabel 4.8', index=False)
    
    print(f"\n>>> File Excel Tabel 4.8 Berhasil Dibuat: {file_excel}")
    print("\n" + "="*80)
    print("[TABEL REKAPITULASI PROYEKSI 7 HARI (TABEL 4.8 BAB IV)]")
    print("="*80)
    print(df_48.to_string(index=False))
    print("="*80)
    print(f"Rata-rata Kesalahan Proyeksi (MAPE 7 Hari): {avg_mape_7:.4f}%")
    print("="*80)
    print(">>> Tahap 8 Selesai.")

if __name__ == "__main__":
    run_prediction_final()