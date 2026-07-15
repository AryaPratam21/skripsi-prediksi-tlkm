import numpy as np
import pandas as pd
import joblib
import os
import json
import yfinance as yf
from tensorflow.keras.models import load_model

def run_price_prediction_pure():
    """
    Fungsi Tahap 6: Prediksi Harga Saham 7 Hari ke Depan (Fase KDD: Knowledge Representation)
    Murni melakukan rolling forecast bergulir untuk memproyeksikan harga saham TLKM
    selama 7 hari kerja ke depan berdasarkan model usulan terbaik.
    Menghasilkan: tahap6_hasil_prediksi_komparasi.xlsx (Tabel 4.8)
    """
    print("\n" + "="*95)
    print(" [TAHAP 6: PREDIKSI HARGA SAHAM 7 HARI KE DEPAN (PROYEKSI)] ")
    print(" [FASE KDD: KNOWLEDGE REPRESENTATION (PENYAJIAN PENGETAHUAN)] ")
    print("="*95)
    
    # 1. Verifikasi Aset
    if not os.path.exists("model_usulan.h5"):
        print("Error: model_usulan.h5 tidak ditemukan! Silakan jalankan Tahap 5 terlebih dahulu.")
        return
        
    if not os.path.exists("scaler_X.pkl") or not os.path.exists("scaler_y.pkl"):
        print("Error: Berkas scaler (.pkl) tidak ditemukan!")
        return

    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)
        
    ws_usulan = best_params['cnn_lstm']['window_size']
    m_usulan = load_model("model_usulan.h5", compile=False)
    
    # 2. Simulasi Proyeksi Iteratif 7 Hari Kerja ke Depan
    print("\n>>> Memulai Proses Proyeksi Iteratif 7 Hari Kerja (Januari 2025)...")
    print("       * Mengunduh data real-world terbaru Telkom (TLKM.JK) dari Yahoo Finance...")
    df_full = yf.download("TLKM.JK", start="2024-09-01", end="2025-01-20", progress=False, auto_adjust=False)
    if isinstance(df_full.columns, pd.MultiIndex): 
        df_full.columns = df_full.columns.get_level_values(0)
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_target = df_full.loc["2025-01-02":"2025-01-10"].copy()
    proj_dates = df_target.index.strftime('%Y-%m-%d')
    actual_7 = df_target['Adj Close'].values

    # Mengambil histori harga sebelum periode proyeksi sepanjang window size
    history = df_full[df_full.index < "2025-01-02"].tail(ws_usulan)[feature_cols].values
    curr_win = scaler_X.transform(history).reshape(1, ws_usulan, -1)
    
    preds_proj = []
    proj_err_pct = []
    
    print("\n      --- Proses Geser Window Bergulir (Rolling Forecast) ---")
    for i in range(len(proj_dates)):
        date_str = proj_dates[i]
        act_val = actual_7[i]
        
        # Prediksi harga hari ke-t
        p = m_usulan.predict(curr_win, verbose=0)
        p_rupiah = float(scaler_y.inverse_transform(p.reshape(-1, 1))[0, 0])
        preds_proj.append(p_rupiah)
        
        dev = abs(act_val - p_rupiah)
        err = (dev / act_val) * 100
        proj_err_pct.append(err)
        
        print(f"        * Hari ke-{i+1} ({date_str}): Prediksi = Rp {p_rupiah:,.2f} | Aktual = Rp {act_val:,.2f} | Error = {err:.4f}%")
        
        # Geser window (masukkan data aktual hari ini untuk prediksi besok)
        new_row = scaler_X.transform(df_target.iloc[[i]][feature_cols].values)
        curr_win = np.append(curr_win[:, 1:, :], new_row.reshape(1, 1, -1), axis=1)

    # 3. Menyusun Laporan Proyeksi
    df_proj = pd.DataFrame({
        'Tanggal': proj_dates,
        'Harga Asli (Rp)': actual_7,
        'Harga Prediksi Usulan (Rp)': np.array(preds_proj),
        'Selisih (Rp)': np.abs(actual_7 - np.array(preds_proj)),
        'Error (%)': np.array(proj_err_pct)
    })
    
    avg_mape_7 = np.mean(proj_err_pct)

    # 4. Menyimpan Hasil ke Excel
    print("\n>>> Mengekspor laporan proyeksi ke Excel...")
    df_proj.to_excel("tahap6_hasil_prediksi_komparasi.xlsx", sheet_name='Proyeksi_7_Hari_Jan2025', index=False)
    print("       * Berkas berhasil disimpan: tahap6_hasil_prediksi_komparasi.xlsx")
    
    print("\n" + "="*80)
    print("[TABEL REKAPITULASI PROYEKSI 7 HARI (TABEL 4.8 BAB IV)]")
    print("="*80)
    print(df_proj.to_string(index=False))
    print("="*80)
    print(f"Rata-rata Kesalahan Proyeksi (MAPE 7 Hari): {avg_mape_7:.4f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_price_prediction_pure()
