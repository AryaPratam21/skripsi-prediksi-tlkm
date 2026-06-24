import numpy as np
import pandas as pd
import joblib
import os
from utils import evaluate_metrics

def run_evaluation():
    """
    Fungsi untuk menghitung akurasi akhir model dan membandingkannya
    dengan model-model pembanding (Rekapitulasi Akhir).
    """
    print("\n" + "="*80)
    print("[TAHAP 7: ANALISIS KOMPARATIF DAN EVALUASI AKHIR]")
    print("="*80)
    
    # 1. Memuat data
    data = np.load("processed_data.npz")
    y_test_default = data['y_test']
    scaler_y = joblib.load('scaler_y.pkl')
    
    if not os.path.exists("tahap3_hasil_baseline.xlsx"):
        print("Error: Jalankan Tahap 3 terlebih dahulu.")
        return
    df_baseline = pd.read_excel("tahap3_hasil_baseline.xlsx")
    
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
    
    all_results = df_baseline.to_dict('records')
    
    # 4. Iterasi Metrik
    print(">>> Mengkalkulasi metrik untuk model teroptimasi...")
    for name, config in preds_config.items():
        if not os.path.exists(config['pred_file']):
            continue
            
        y_p = np.load(config['pred_file'])
        y_t = np.load(config['true_file']) if os.path.exists(config['true_file']) else y_test_default
            
        y_t_inv = scaler_y.inverse_transform(y_t.reshape(-1, 1))
        y_p_inv = scaler_y.inverse_transform(y_p.reshape(-1, 1))
        
        res = evaluate_metrics(y_t_inv, y_p_inv)
        params = best_params.get(config['param_key'], {})
        
        # Mapping parameter agar sesuai kolom tabel (Sinkron dengan Tahap 4-5)
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
    
    # Urutkan agar model usulan ada di paling bawah/paling menonjol
    print("\n" + "="*100)
    print("REKAPITULASI PERFORMA SELURUH MODEL (BASELINES VS OPTIMIZED)")
    print("="*100)
    print(df_final.to_string(index=False, justify='center'))
    print("="*100)
    
    # Simpan ke Excel
    df_final.to_excel("tahap7_hasil_evaluasi.xlsx", index=False)
    print(f"\n>>> Laporan lengkap berhasil dibuat: tahap7_hasil_evaluasi.xlsx")
    
    # Kesimpulan Singkat
    best_model = df_final.loc[df_final['MAPE'].idxmin()]
    print(f">>> MODEL TERBAIK: {best_model['Model']} dengan MAPE: {best_model['MAPE']:.4f}%")

if __name__ == "__main__":
    run_evaluation()
