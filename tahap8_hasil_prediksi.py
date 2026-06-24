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
    print("\n" + "="*65)
    print("[TAHAP 8: VISUALISASI FINAL SESUAI LAPORAN BAB 4]")
    print("="*65)
    
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
    
    # Load Model Manual (Hasil Tahap 3)
    if not os.path.exists("model_manual.h5"):
        print("Error: model_manual.h5 tidak ditemukan!")
        return
    m_manual = load_model("model_manual.h5", compile=False)
    print(">>> Model Usulan & Model Manual Berhasil Dimuat.")

    # Load Hyperparameters untuk Window Size
    data = np.load("processed_data.npz")
    with open('best_params.json', 'r') as f:
        best = json.load(f)
    
    ws_usulan = best['cnn_lstm']['window_size']
    ws_manual = 30 # Standar baseline manual

    # --- 2. GENERATE GAMBAR 4.13 (PERFORMA KESELURUHAN - 100 HARI TESTING) ---
    # Bagian ini tetap dipertahankan sesuai narasi utama Bab 4
    print("\n>>> Membuat Gambar 4.13 (Grafik Perbandingan Performa Keseluruhan)...")
    ts_X_raw = data['test_X_scaled']
    ts_y_raw = data['test_y_scaled']

    def get_preds_all(model, ws, X_raw):
        X_tmp = []
        for i in range(len(X_raw) - ws):
            X_tmp.append(X_raw[i:(i + ws), :])
        preds = model.predict(np.array(X_tmp), verbose=0)
        return scaler_y.inverse_transform(preds).flatten()

    y_pred_usulan_all = get_preds_all(m_usulan, ws_usulan, ts_X_raw)
    y_pred_manual_all = get_preds_all(m_manual, ws_manual, ts_X_raw)
    
    # Sinkronisasi panjang data aktual
    min_len = min(len(y_pred_usulan_all), len(y_pred_manual_all))
    y_act_all = scaler_y.inverse_transform(ts_y_raw[-min_len:].reshape(-1,1)).flatten()
    
    plt.figure(figsize=(15, 8))
    plt.plot(y_act_all[-100:], color='black', label='Harga Aktual (Adj Close)', linewidth=2.5)
    plt.plot(y_pred_usulan_all[-min_len:][-100:], color='red', linestyle='--', label='CNN-LSTM + BO (USULAN)', linewidth=2)
    plt.plot(y_pred_manual_all[-min_len:][-100:], color='green', linestyle=':', label='CNN-LSTM (Manual)', linewidth=2)
    
    plt.title('Gambar 4.13: Hasil Prediksi Dan Kesimpulan Akhir (Performa Testing)', fontsize=14)
    plt.xlabel('Hari (Data Testing)'); plt.ylabel('Harga Saham (Rp)')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.savefig("Gambar_4.13_Hasil_Prediksi_Kesimpulan.png", dpi=300)
    print("    - Gambar 4.13 Berhasil Disimpan.")

    # --- 3. MEKANISME PROYEKSI 7 HARI (UNTUK TABEL 4.8) ---
    print("\n>>> Memproses Data Proyeksi 7 Hari (Januari 2025)...")
    df_full = yf.download("TLKM.JK", start="2024-09-01", end="2025-01-20", progress=False, auto_adjust=False)
    if isinstance(df_full.columns, pd.MultiIndex): df_full.columns = df_full.columns.get_level_values(0)
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_target = df_full.loc["2025-01-02":"2025-01-10"].copy()
    dates = df_target.index.strftime('%Y-%m-%d')
    actual_7 = df_target['Adj Close'].values

    def iterative_forecast(model, ws):
        history = df_full[df_full.index < "2025-01-02"].tail(ws)[feature_cols].values
        curr_win = scaler_X.transform(history).reshape(1, ws, -1)
        preds = []
        for i in range(len(dates)):
            p = model.predict(curr_win, verbose=0)
            preds.append(p[0, 0])
            new_row = scaler_X.transform(df_target.iloc[[i]][feature_cols].values)
            curr_win = np.append(curr_win[:, 1:, :], new_row.reshape(1, 1, -1), axis=1)
        return scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    y_pred_7_usulan = iterative_forecast(m_usulan, ws_usulan)
    y_pred_7_manual = iterative_forecast(m_manual, ws_manual)

    # --- 4. SIMPAN EXCEL TABEL 4.8 ---
    # Bagian ini tetap dipertahankan karena narasi Anda menyebutkan ekspor data ke Excel
    df_48 = pd.DataFrame({
        'Tanggal': dates,
        'Harga Asli (Rp)': actual_7,
        'Harga Prediksi Manual (Rp)': y_pred_7_manual,
        'Harga Prediksi Usulan (Rp)': y_pred_7_usulan
    })
    
    file_excel = "tahap8_hasil_prediksi.xlsx"
    df_48.to_excel("tahap8_hasil_prediksi.xlsx", sheet_name='Tabel 4.8', index=False)
    
    print(f"\n>>> File Excel Berhasil Dibuat: {file_excel}")
    print("="*65)
    print("[PREVIEW TABEL 4.8]")
    print(df_48.to_string(index=False))
    print("="*65)
    print(">>> Tahap 8 Selesai.")

if __name__ == "__main__":
    run_prediction_final()