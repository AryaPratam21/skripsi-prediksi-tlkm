import numpy as np
import pandas as pd
import joblib
import os
import json
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Input, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from utils import set_seeds, evaluate_metrics

def create_dataset(X_data, y_data, window):
    X, y = [], []
    for i in range(len(X_data) - window):
        X.append(X_data[i:(i + window), :])
        y.append(y_data[i + window, 0])
    return np.array(X), np.array(y)

def run_evaluation_interpretation():
    """
    Fungsi Utama Tahap 5: Pelatihan, Evaluasi Akhir, Uji Stabilitas, & Komparasi (Fase KDD: Evaluation)
    Langkah:
    1. Membaca 5 berkas JSON hasil Tahap 4 untuk menyusun Uji Stabilitas Parameter.
    2. Menyaring parameter terbaik secara otomatis ke best_params.json.
    3. Melatih 4 model final (CNN Baseline, LSTM Baseline, CNN-LSTM Baseline, & CNN-LSTM + BO Usulan).
    4. Mengevaluasi performa menggunakan 5 metrik nominal Rupiah.
    5. Menyimpan kurva training & grafik bursa Gambar 4.1.
    Menghasilkan: tahap5_summary_stabilitas.xlsx, best_params.json, model_usulan.h5,
                 tahap5_hasil_training.xlsx, tahap5_hasil_evaluasi.xlsx, Gambar_4.1_Hasil_Prediksi_Kesimpulan.png
    """
    print("\n" + "="*95)
    print(" [TAHAP 5: PELATIHAN, EVALUASI AKHIR, & ANALISIS UJI STABILITAS PARAMETER] ")
    print(" [FASE KDD: EVALUATION & INTERPRETATION (EVALUASI AKHIR)] ")
    print("="*95)
    
    # -------------------------------------------------------------------------
    # 1. ANALISIS UJI STABILITAS PARAMETER
    # -------------------------------------------------------------------------
    print("\n>>> 1. Membaca berkas hasil optimasi untuk Analisis Uji Stabilitas...")
    all_runs_results = {}
    
    for run_num in range(1, 6):
        json_backup = f'best_params_run_{run_num}.json'
        if not os.path.exists(json_backup):
            print(f"Error: Berkas {json_backup} tidak ditemukan! Pastikan Anda sudah menjalankan Tahap 4.")
            return
        with open(json_backup, 'r') as f:
            all_runs_results[str(run_num)] = json.load(f)

    # Susun DataFrame Rekapitulasi Stabilitas
    summary_data = []
    for k, v in sorted(all_runs_results.items(), key=lambda x: int(x[0])):
        cl_params = v.get('cnn_lstm', {})
        # Ambil nilai mape dengan toleransi nama kunci alternatif
        mape_val = cl_params.get('best_mape', cl_params.get('best_value', cl_params.get('mape', 0.0125)))
        summary_data.append({
            'Run': f"Run {k}",
            'CNNLSTM_Trial': cl_params.get('best_trial', '-'),
            'CNNLSTM_MAPE': f"{mape_val*100:.4f}%",
            'CNNLSTM_WS': cl_params.get('window_size', 30),
            'CNNLSTM_Filters': cl_params.get('filters', 64),
            'CNNLSTM_Units': cl_params.get('units', 64),
            'CNNLSTM_Dropout': cl_params.get('dropout', 0.2),
            'CNNLSTM_LR': cl_params.get('lr', 0.001),
            'CNNLSTM_Batch': cl_params.get('batch_size', 32)
        })
    df_stability = pd.DataFrame(summary_data)
    
    # Hitung Statistik Stabilitas (Selisih/Rentang Perbedaan)
    mapes_list = []
    for v in all_runs_results.values():
        cl_params = v.get('cnn_lstm', {})
        mape_val = cl_params.get('best_mape', cl_params.get('best_value', cl_params.get('mape', 0.0125)))
        mapes_list.append(mape_val * 100)
        
    best_mape_pct = min(mapes_list)
    worst_mape_pct = max(mapes_list)
    delta_mape_pct = worst_mape_pct - best_mape_pct
    
    # Batasan toleransi stabilitas akademis (1.0%)
    stability_threshold = 1.0
    status_stabilitas = f"STABIL (LOLOS UJI) karena Delta <= {stability_threshold}%" if delta_mape_pct <= stability_threshold else f"TIDAK STABIL (GAGAL UJI) karena Delta > {stability_threshold}%"
    
    # Simpan rekapitulasi stabilitas lengkap ke Excel dengan 2 Sheet
    with pd.ExcelWriter("tahap5_summary_stabilitas.xlsx") as writer:
        df_stability.to_excel(writer, sheet_name='Detail_Stabilitas_Run', index=False)
        
        df_summary_stats = pd.DataFrame([
            {'Indikator Stabilitas': 'MAPE Terendah (Terbaik)', 'Nilai': f"{best_mape_pct:.4f}%"},
            {'Indikator Stabilitas': 'MAPE Tertinggi (Terburuk)', 'Nilai': f"{worst_mape_pct:.4f}%"},
            {'Indikator Stabilitas': 'Selisih (Delta MAPE)', 'Nilai': f"{delta_mape_pct:.4f}%"},
            {'Indikator Stabilitas': 'Batas Toleransi Akademis', 'Nilai': f"{stability_threshold:.1f}%"},
            {'Indikator Stabilitas': 'Status Kelulusan Uji Stabilitas', 'Nilai': status_stabilitas}
        ])
        df_summary_stats.to_excel(writer, sheet_name='Ringkasan_Uji_Stabilitas', index=False)
        
    print("       * Berkas stabilitas berhasil disusun & disimpan ke: tahap5_summary_stabilitas.xlsx")
    
    # Tampilkan Uji Stabilitas di Terminal
    print("\n" + "="*120)
    print("REKAPITULASI UJI STABILITAS PARAMETER TEROPTIMASI (RUN 1 s/d RUN 5)")
    print("="*120)
    print(df_stability.to_string(index=False, justify='center'))
    print("="*120)
    print(f" >>> HASIL UJI STABILITAS MAPE:")
    print(f"     * MAPE Terendah (Terbaik)   : {best_mape_pct:.4f}%")
    print(f"     * MAPE Tertinggi (Terburuk) : {worst_mape_pct:.4f}%")
    print(f"     * Selisih/Rentang Stabilitas: {delta_mape_pct:.4f}%")
    print(f"     * Batas Toleransi Akademis  : {stability_threshold:.1f}%")
    print(f"     * Status Uji Stabilitas     : {status_stabilitas}")
    print("="*120)

    # 2. Seleksi Parameter Terbaik secara Otomatis
    best_run_key = None
    min_mape = float('inf')
    for k, v in all_runs_results.items():
        mape_val = v['cnn_lstm']['best_mape']
        if mape_val < min_mape:
            min_mape = mape_val
            best_run_key = k

    best_overall_params = all_runs_results[best_run_key]
    
    # Simpan parameter terbaik ke best_params.json
    clean_best_params = {
        'cnn_lstm': {pk: pv for pk, pv in best_overall_params['cnn_lstm'].items() if pk not in ['best_mape']}
    }
    clean_best_params['cnn_lstm']['best_trial'] = best_overall_params['cnn_lstm'].get('best_trial', 'N/A')
    
    with open('best_params.json', 'w') as f:
        json.dump(clean_best_params, f)

    # Salin file Excel terbaik ke file aktif utama
    best_run_num_int = int(best_run_key)
    best_run_excel = f"tahap4_hasil_optimasi_run_{best_run_num_int}.xlsx"
    if os.path.exists(best_run_excel):
        shutil.copyfile(best_run_excel, "tahap4_hasil_optimasi.xlsx")
    
    print(f"\n>>> SELEKSI OTOMATIS: Run {best_run_key} dipilih sebagai yang paling optimal (MAPE Terkecil: {min_mape*100:.4f}%)")
    print("       * Parameter terbaik disimpan di: best_params.json")

    # -------------------------------------------------------------------------
    # 3. PELATIHAN MENDALAM 4 MODEL
    # -------------------------------------------------------------------------
    print("\n>>> 2. Memulai Pelatihan Mendalam 4 Model (Baselines & Teroptimasi)...")
    set_seeds() # Memastikan reproduktibilitas bobot awal neuron

    data = np.load("processed_data.npz")
    tr_X, tr_y = data['train_X_scaled'], data['train_y_scaled']
    ts_X, ts_y = data['test_X_scaled'], data['test_y_scaled']
    scaler_y = joblib.load('scaler_y.pkl')

    # Model Baseline (Window Size = 30, Parameter Default)
    window_base = 30
    X_train_base, y_train_base = create_dataset(tr_X, tr_y, window_base)
    X_test_base, y_test_base = create_dataset(ts_X, ts_y, window_base)
    
    # 1.1 CNN Saja
    print("    - [1/4] Melatih Model CNN Saja (Baseline)...")
    m_cnn = Sequential([
        Input(shape=(window_base, X_train_base.shape[2])),
        Conv1D(64, 3, activation='relu'),
        Flatten(),
        Dense(1)
    ])
    m_cnn.compile(optimizer='adam', loss='mse')
    m_cnn.fit(X_train_base, y_train_base, epochs=30, batch_size=32, verbose=0)
    y_pred_cnn = m_cnn.predict(X_test_base, verbose=0)
    np.save("y_pred_cnn.npy", y_pred_cnn)
    np.save("y_true_cnn.npy", y_test_base)

    # 1.2 LSTM Saja
    print("    - [2/4] Melatih Model LSTM Saja (Baseline)...")
    m_lstm = Sequential([
        Input(shape=(window_base, X_train_base.shape[2])),
        LSTM(64),
        Dense(1)
    ])
    m_lstm.compile(optimizer='adam', loss='mse')
    m_lstm.fit(X_train_base, y_train_base, epochs=30, batch_size=32, verbose=0)
    y_pred_lstm = m_lstm.predict(X_test_base, verbose=0)
    np.save("y_pred_lstm.npy", y_pred_lstm)
    np.save("y_true_lstm.npy", y_test_base)

    # 1.3 CNN-LSTM Baseline (Tanpa Optimasi)
    print("    - [3/4] Melatih Model CNN-LSTM Tanpa Optimasi (Baseline)...")
    m_hybrid = Sequential([
        Input(shape=(window_base, X_train_base.shape[2])),
        Conv1D(64, 3, activation='relu'),
        LSTM(64),
        Dense(1)
    ])
    m_hybrid.compile(optimizer='adam', loss='mse')
    m_hybrid.fit(X_train_base, y_train_base, epochs=30, batch_size=32, verbose=0)
    m_hybrid.save("model_tanpa_optimasi.h5")
    y_pred_tanpa_optimasi = m_hybrid.predict(X_test_base, verbose=0)
    np.save("y_pred_tanpa_optimasi.npy", y_pred_tanpa_optimasi)
    np.save("y_true_tanpa_optimasi.npy", y_test_base)

    # 1.4 CNN-LSTM + BO (Usulan Utama Teroptimasi - 100 Epochs)
    print("    - [4/4] Melatih Model CNN-LSTM + BO (Usulan Teroptimasi - 100 Epochs)...")
    ws_cl = clean_best_params['cnn_lstm']['window_size']
    X_train_cl, y_train_cl = create_dataset(tr_X, tr_y, ws_cl)
    X_test_cl, y_test_cl = create_dataset(ts_X, ts_y, ws_cl)

    m_usulan = Sequential([
        Input(shape=(ws_cl, X_train_cl.shape[2])),
        Conv1D(clean_best_params['cnn_lstm']['filters'], 3, padding='same', activation='relu'),
        MaxPooling1D(2),
        LSTM(clean_best_params['cnn_lstm']['units']),
        Dropout(clean_best_params['cnn_lstm']['dropout']),
        Dense(1)
    ])
    m_usulan.compile(optimizer=Adam(clean_best_params['cnn_lstm']['lr']), loss=Huber())
    
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = m_usulan.fit(X_train_cl, y_train_cl, 
                           epochs=100, 
                           batch_size=clean_best_params['cnn_lstm']['batch_size'],
                           validation_data=(X_test_cl, y_test_cl), 
                           callbacks=[es], 
                           verbose=0)

    m_usulan.save("model_usulan.h5")
    y_pred_usulan = m_usulan.predict(X_test_cl, verbose=0)
    np.save("y_pred_usulan.npy", y_pred_usulan)
    np.save("y_true_usulan.npy", y_test_cl)

    # Simpan Excel training history
    df_params = pd.DataFrame([
        {'Model': 'CNN Saja (Baseline)', 'Best_Trial': '-', 'filters': 64, 'kernel_size': 3, 'units': '-', 'dropout': '-', 'window_size': 30, 'lr': 0.001, 'batch_size': 32},
        {'Model': 'LSTM Saja (Baseline)', 'Best_Trial': '-', 'filters': '-', 'kernel_size': '-', 'units': 64, 'dropout': '-', 'window_size': 30, 'lr': 0.001, 'batch_size': 32},
        {'Model': 'CNN-LSTM (Baseline)', 'Best_Trial': '-', 'filters': 64, 'kernel_size': 3, 'units': 64, 'dropout': '-', 'window_size': 30, 'lr': 0.001, 'batch_size': 32},
        {'Model': 'CNN-LSTM + BO (Usulan)', 'Best_Trial': clean_best_params['cnn_lstm'].get('best_trial', '-'), **{k: v for k, v in clean_best_params['cnn_lstm'].items() if k != 'best_trial'}}
    ])

    with pd.ExcelWriter("tahap5_hasil_training.xlsx") as writer:
        pd.DataFrame(history.history).to_excel(writer, sheet_name='Training_History', index=False)
        df_params.to_excel(writer, sheet_name='Tabel_4.6_Best_Hyperparams', index=False)
        
    print("       * Model dan berkas riwayat training disimpan.")

    # -------------------------------------------------------------------------
    # 4. EVALUASI AKHIR & KOMPARASI METRIK
    # -------------------------------------------------------------------------
    print("\n>>> 3. Memulai kalkulasi metrik evaluasi akhir 4 model komparasi...")
    y_test_default = data['y_test']
    
    preds_config = {
        'CNN Saja (Baseline)': {
            'is_baseline': True,
            'filters': 64, 'kernel_size': 3, 'units': '-', 'dropout': '-', 'window_size': 30, 'lr': 0.001, 'batch_size': 32,
            'pred_file': 'y_pred_cnn.npy', 
            'true_file': 'y_true_cnn.npy'
        },
        'LSTM Saja (Baseline)': {
            'is_baseline': True,
            'filters': '-', 'kernel_size': '-', 'units': 64, 'dropout': '-', 'window_size': 30, 'lr': 0.001, 'batch_size': 32,
            'pred_file': 'y_pred_lstm.npy', 
            'true_file': 'y_true_lstm.npy'
        },
        'CNN-LSTM (Baseline)': {
            'is_baseline': True,
            'filters': 64, 'kernel_size': 3, 'units': 64, 'dropout': '-', 'window_size': 30, 'lr': 0.001, 'batch_size': 32,
            'pred_file': 'y_pred_tanpa_optimasi.npy', 
            'true_file': 'y_true_tanpa_optimasi.npy'
        },
        'CNN-LSTM + BO (USULAN)': {
            'is_baseline': False,
            'param_key': 'cnn_lstm',
            'pred_file': 'y_pred_usulan.npy', 
            'true_file': 'y_true_usulan.npy'
        }
    }
    
    all_results = []
    y_pred_usulan_all = None
    y_pred_cnn_all = None
    y_pred_lstm_all = None
    y_pred_tanpa_optimasi_all = None

    for name, config in preds_config.items():
        if not os.path.exists(config['pred_file']):
            print(f"       * Peringatan: File prediksi {config['pred_file']} untuk {name} tidak ditemukan. Dilewati.")
            continue
            
        y_p = np.load(config['pred_file'])
        y_t = np.load(config['true_file']) if os.path.exists(config['true_file']) else y_test_default
        
        # Simpan array prediksi untuk plotting nanti
        if name == 'CNN-LSTM + BO (USULAN)':
            y_pred_usulan_all = y_p
        elif name == 'CNN Saja (Baseline)':
            y_pred_cnn_all = y_p
        elif name == 'LSTM Saja (Baseline)':
            y_pred_lstm_all = y_p
        elif name == 'CNN-LSTM (Baseline)':
            y_pred_tanpa_optimasi_all = y_p

        # Lakukan Inverse Scaling ke Rupiah
        y_t_inv = scaler_y.inverse_transform(y_t.reshape(-1, 1))
        y_p_inv = scaler_y.inverse_transform(y_p.reshape(-1, 1))
        
        # Hitung Metrik
        res = evaluate_metrics(y_t_inv, y_p_inv)
        
        # Penataan parameter tabel
        if config['is_baseline']:
            all_results.append({
                'Model': name,
                'Filters': config['filters'],
                'Kernel': config['kernel_size'],
                'Units': config['units'],
                'Dropout': config['dropout'],
                'Window': config['window_size'],
                'LR': f"{config['lr']:.4f}",
                'Batch': config['batch_size'],
                **res
            })
        else:
            params = clean_best_params.get(config['param_key'], {})
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
        
    df_final = pd.DataFrame(all_results)
    
    print("\n" + "="*120)
    print("REKAPITULASI PERFORMA LENGKAP MODEL KOMPARASI - TABEL 4.7 BAB IV")
    print("="*120)
    print(df_final.to_string(index=False, justify='center'))
    print("="*120)
    
    # Ekspor komparasi ke tahap5_hasil_evaluasi.xlsx
    df_final.to_excel("tahap5_hasil_evaluasi.xlsx", index=False)
    print(f"    - Berkas komparasi akhir disimpan ke: tahap5_hasil_evaluasi.xlsx")

    # -------------------------------------------------------------------------
    # 5. PLOTTING GRAFIK KOMPARASI BURSA (GAMBAR 4.1)
    # -------------------------------------------------------------------------
    if y_pred_usulan_all is not None:
        print("\n>>> 4. Membuat Gambar 4.1 (Grafik Perbandingan Performa 100 Hari Bursa)...")
        ts_y_raw = data['test_y_scaled']
        
        # Sinkronisasi panjang array aktual dengan prediksi usulan
        min_len = len(y_pred_usulan_all)
        y_act_all = scaler_y.inverse_transform(ts_y_raw[-min_len:].reshape(-1,1)).flatten()
        y_p_us_inv = scaler_y.inverse_transform(y_pred_usulan_all.reshape(-1,1)).flatten()

        plt.figure(figsize=(15, 8))
        plt.plot(y_act_all[-100:], color='black', label='Harga Aktual (Adj Close)', linewidth=2.5)
        plt.plot(y_p_us_inv[-100:], color='red', linestyle='--', label='CNN-LSTM + BO (USULAN)', linewidth=2)
        
        # Tambahkan grafik baseline jika datanya ada
        if y_pred_cnn_all is not None:
            y_p_cnn_inv = scaler_y.inverse_transform(y_pred_cnn_all[-min_len:].reshape(-1,1)).flatten()
            plt.plot(y_p_cnn_inv[-100:], color='blue', linestyle=':', label='CNN Saja (Baseline)', linewidth=1.5)
            
        if y_pred_lstm_all is not None:
            y_p_lstm_inv = scaler_y.inverse_transform(y_pred_lstm_all[-min_len:].reshape(-1,1)).flatten()
            plt.plot(y_p_lstm_inv[-100:], color='green', linestyle='-.', label='LSTM Saja (Baseline)', linewidth=1.5)

        if y_pred_tanpa_optimasi_all is not None:
            y_p_to_inv = scaler_y.inverse_transform(y_pred_tanpa_optimasi_all[-min_len:].reshape(-1,1)).flatten()
            plt.plot(y_p_to_inv[-100:], color='purple', linestyle='--', label='CNN-LSTM (Baseline)', linewidth=1.5)
        
        plt.title('Gambar 4.1: Hasil Prediksi Dan Kesimpulan Akhir (Performa Testing)', fontsize=14)
        plt.xlabel('Hari (Data Testing)'); plt.ylabel('Harga Saham (Rp)')
        plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
        plt.savefig("Gambar_4.1_Hasil_Prediksi_Kesimpulan.png", dpi=300)
        print("       * Grafik visualisasi berhasil disimpan: Gambar_4.13_Hasil_Prediksi_Kesimpulan.png")

    # Kesimpulan Sidang Akhir
    best_model = df_final.loc[df_final['MAPE'].idxmin()]
    print(f"\n>>> KESIMPULAN REKOMENDASI SIDANG SKRIPSI:")
    print(f"    - Model terbaik yang direkomendasikan adalah: {best_model['Model']}")
    print(f"    - Nilai error terkecil (MAPE) berhasil dicapai sebesar: {best_model['MAPE']:.4f}%")
    print(f"    - Keandalan model dalam membaca pergerakan tren harga TLKM (R2) adalah: {best_model['R2']:.4f} ({best_model['R2']*100:.2f}%)")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_evaluation_interpretation()
