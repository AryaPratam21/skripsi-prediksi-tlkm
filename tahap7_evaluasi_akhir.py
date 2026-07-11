import numpy as np
import pandas as pd
import joblib
import os
from utils import evaluate_metrics

def run_evaluation():
    """
    Fungsi untuk menghitung akurasi akhir model dan membandingkannya
    dengan model-model pembanding (Rekapitulasi Akhir).
    Menampilkan detail proses perhitungan matematis secara transparan.
    """
    print("\n" + "="*90)
    print(" [TAHAP 7: ANALISIS KOMPARATIF DAN EVALUASI AKHIR] ")
    print(" [FASE KDD: EVALUATION & INTERPRETATION (EVALUASI AKHIR)] ")
    print("="*90)
    
    print("\n>>> PENJELASAN METODOLOGI EVALUASI (UNTUK BAB IV):")
    print("    1. Membaca data aktual dan data prediksi dalam bentuk ternormalisasi [0-1].")
    print("    2. Mengembalikan data tersebut ke skala asli (Rupiah) melalui Inverse MinMaxScaler.")
    print("    3. Menghitung 5 metrik evaluasi utama menggunakan formula statistik standar:")
    print("       * Mean Squared Error (MSE) : Rata-rata selisih kuadrat.")
    print("       * Mean Absolute Error (MAE) : Rata-rata selisih absolut (dalam Rupiah).")
    print("       * Root Mean Squared Error (RMSE) : Akar kuadrat dari MSE (dalam Rupiah).")
    print("       * Mean Absolute Percentage Error (MAPE) : Persentase deviasi rata-rata.")
    print("       * R-squared (R2) : Koefisien determinasi (kecocokan pola tren bursa).")

    # 1. Memuat data actual dasar
    data = np.load("processed_data.npz")
    y_test_default = data['y_test']
    scaler_y = joblib.load('scaler_y.pkl')
    
    if not os.path.exists("tahap3_hasil_baseline.xlsx"):
        print("Error: File tahap3_hasil_baseline.xlsx tidak ditemukan! Jalankan Tahap 3 terlebih dahulu.")
        return
    
    print("\n>>> 1. Memuat baseline model tanpa optimasi dari tahap3_hasil_baseline.xlsx...")
    df_baseline = pd.read_excel("tahap3_hasil_baseline.xlsx")
    for idx, row in df_baseline.iterrows():
        print(f"       * Terbaca baseline: {row['Model']} | MAPE: {row['MAPE']:.4f}% | RMSE: {row['RMSE']:.2f}")
    
    all_results = df_baseline.to_dict('records')
    
    if not os.path.exists('best_params.json'):
        print("Error: best_params.json tidak ditemukan!")
        return
        
    import json
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)
    
    preds_config = {
        'BiLSTM + BO (Pembanding)': {
            'param_key': 'bilstm', 
            'pred_file': 'y_pred_bi.npy', 
            'true_file': 'y_true_bi.npy'
        },
        'CNN-LSTM + BO (USULAN)': {
            'param_key': 'cnn_lstm', 
            'pred_file': 'y_pred_usulan.npy', 
            'true_file': 'y_true_usulan.npy'
        }
    }
    
    print("\n>>> 2. Memulai kalkulasi metrik evaluasi model teroptimasi...")
    for name, config in preds_config.items():
        if not os.path.exists(config['pred_file']):
            print(f"       * Peringatan: File prediksi {config['pred_file']} untuk {name} tidak ditemukan. Dilewati.")
            continue
            
        print(f"\n      --- Mengolah Performa Model: {name} ---")
        y_p = np.load(config['pred_file'])
        y_t = np.load(config['true_file']) if os.path.exists(config['true_file']) else y_test_default
        
        print(f"          - Memuat array prediksi ({len(y_p)} data) dan array aktual ({len(y_t)} data).")
        
        # Lakukan Inverse Scaling
        print(f"          - Melakukan Inverse Min-Max Scaling ke Rupiah (Rp)...")
        y_t_inv = scaler_y.inverse_transform(y_t.reshape(-1, 1))
        y_p_inv = scaler_y.inverse_transform(y_p.reshape(-1, 1))
        
        # Tampilkan sampel data sebelum & sesudah inverse
        print(f"          - Sampel 3 data pertama setelah dikembalikan ke Rupiah:")
        for i in range(min(3, len(y_t_inv))):
            act_val = float(y_t_inv[i, 0])
            pred_val = float(y_p_inv[i, 0])
            dev_val = abs(act_val - pred_val)
            print(f"            * Hari ke-{i+1}: Aktual = Rp {act_val:,.2f} | Prediksi = Rp {pred_val:,.2f} | Deviasi = Rp {dev_val:,.2f}")
            
        # Hitung Metrik
        print(f"          - Menghitung nilai metrik statistik...")
        res = evaluate_metrics(y_t_inv, y_p_inv)
        
        # Hitung R2 manual untuk log penjelasan akademis
        ss_res = np.sum((y_t_inv - y_p_inv) ** 2)
        ss_tot = np.sum((y_t_inv - np.mean(y_t_inv)) ** 2)
        r2_calculated = 1 - (ss_res / ss_tot)
        
        print(f"            * Hasil MSE  (Mean Squared Error)       : {res['MSE']:.4f}")
        print(f"            * Hasil MAE  (Mean Absolute Error)      : Rp {res['MAE']:.2f}")
        print(f"            * Hasil RMSE (Root Mean Squared Error) : Rp {res['RMSE']:.2f}")
        print(f"            * Hasil MAPE (Mean Absolute Percentage Error) : {res['MAPE']:.4f}%")
        print(f"            * Hasil R2   (Coefficient of Determination)   : {res['R2']:.4f} (Akurasi tren: {res['R2']*100:.2f}%)")
        
        params = best_params.get(config['param_key'], {})
        all_results.append({
            'Model': name,
            'Filters': params.get('filters', '-'),
            'Kernel': params.get('kernel_size', '-'),
            'Units': params.get('units', '-'),
            'Dropout': params.get('dropout', '-'),
            'Window': params.get('window_size', '-'),
            'LR': f"{params.get('lr', 0):.4f}",
            'Batch': params.get('batch_size', '-'),
            **res
        })
        
    # 5. Menyusun dan Menampilkan Laporan Akhir
    df_final = pd.DataFrame(all_results)
    
    print("\n" + "="*120)
    print("REKAPITULASI PERFORMA SELURUH MODEL (BASELINES VS OPTIMIZED) - TABEL 4.7 BAB IV")
    print("="*120)
    print(df_final.to_string(index=False, justify='center'))
    print("="*120)
    
    # Simpan ke Excel
    df_final.to_excel("tahap7_hasil_evaluasi.xlsx", index=False)
    print(f"\n>>> Laporan lengkap berhasil dibuat dan disimpan: tahap7_hasil_evaluasi.xlsx")
    
    # Kesimpulan Akhir
    best_model = df_final.loc[df_final['MAPE'].idxmin()]
    print(f"\n>>> ANALISIS KESIMPULAN SIDANG SKRIPSI:")
    print(f"    - Model terbaik yang direkomendasikan adalah: {best_model['Model']}")
    print(f"    - Nilai error terkecil (MAPE) berhasil dicapai sebesar: {best_model['MAPE']:.4f}%")
    print(f"    - Keandalan model dalam membaca pergerakan tren harga TLKM (R2) adalah: {best_model['R2']:.4f} ({best_model['R2']*100:.2f}%)")
    print("="*90 + "\n")

if __name__ == "__main__":
    run_evaluation()
