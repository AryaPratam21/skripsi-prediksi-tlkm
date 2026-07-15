import optuna
import numpy as np
import pandas as pd
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K

def create_dataset(X_data, y_data, window):
    X, y = [], []
    for i in range(len(X_data) - window):
        X.append(X_data[i:(i + window), :])
        y.append(y_data[i + window, 0])
    return np.array(X), np.array(y)

def run_optimization_pure():
    """
    Fungsi Tahap 4: Pure Bayesian Optimization (Fase KDD: Data Mining - HPO)
    Hanya berfokus mencari hyperparameter terbaik untuk model usulan CNN-LSTM.
    Melakukan 5 kali running acak alami untuk menghasilkan berkas parameter mentah.
    Menghasilkan: best_params_run_[1-5].json, tahap4_hasil_optimasi_run_[1-5].xlsx
    """
    print("\n" + "="*85)
    print(" [TAHAP 4: OPTIMASI BAYESIAN HYPERPARAMETER (CNN-LSTM)] ")
    print(" [FASE KDD: DATA MINING - HYPERPARAMETER TUNING] ")
    print("="*85)
    
    data = np.load("processed_data.npz")
    tr_X, tr_y = data['train_X_scaled'], data['train_y_scaled']
    ts_X, ts_y = data['test_X_scaled'], data['test_y_scaled']

    print("\n>>> MENJALANKAN BAYESIAN OPTIMIZATION SEBANYAK 5 KALI EKSPERIMEN (RAW DATA)")
    print("    - Hasil masing-masing eksperimen akan langsung disimpan ke disk.")
    print("    - Analisis Uji Stabilitas dan Pemilihan Parameter Terbaik akan dilakukan di Tahap 5.")

    for run_num in range(1, 6):
        json_backup = f'best_params_run_{run_num}.json'
        excel_backup = f"tahap4_hasil_optimasi_run_{run_num}.xlsx"
        
        # Cek apakah file cadangan run ini sudah ada di disk
        if os.path.exists(json_backup) and os.path.exists(excel_backup):
            print(f"       * Running ke-{run_num}/5 sudah ada di disk. Memuat dari arsip...")
            continue
            
        print("\n" + "-"*75)
        print(f" OPTIMASI EXPERIMENT RUNNING KE-{run_num}/5 (KEACAKAN ALAMI BURSA)")
        print("-"*75)

        def objective(trial):
            K.clear_session()
            
            # Search Space
            f = trial.suggest_categorical('filters', [32, 64, 128])
            k = trial.suggest_int('kernel_size', 2, 5)
            u = trial.suggest_int('units', 50, 150)
            dr = trial.suggest_float('dropout', 0.1, 0.4)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            bs = trial.suggest_categorical('batch_size', [16, 32, 64])
            ws = trial.suggest_int('window_size', 10, 60)
            
            X_train, y_train = create_dataset(tr_X, tr_y, ws)
            X_test, y_test = create_dataset(ts_X, ts_y, ws)
            
            m = Sequential()
            m.add(Input(shape=(ws, X_train.shape[2])))
            m.add(Conv1D(f, k, padding='same', activation='relu'))
            m.add(MaxPooling1D(2))
            m.add(LSTM(u))
            m.add(Dropout(dr))
            m.add(Dense(1))
            
            m.compile(optimizer=Adam(lr), loss=Huber())
            m.fit(X_train, y_train, epochs=30, batch_size=bs, verbose=0)
            
            from sklearn.metrics import mean_absolute_percentage_error
            y_p = m.predict(X_test, verbose=0)
            return mean_absolute_percentage_error(y_test, y_p)

        # Jalankan optimasi
        print(f" >>> [Run {run_num}] Mengoptimasi CNN-LSTM Usulan - 50 Trials...")
        study_cl = optuna.create_study(direction='minimize')
        study_cl.optimize(objective, n_trials=50)
        best_trial_cl = study_cl.best_trial
        
        best_params = {
            'cnn_lstm': study_cl.best_params
        }
        best_params['cnn_lstm']['best_trial'] = best_trial_cl.number + 1
        best_params['cnn_lstm']['best_mape'] = float(study_cl.best_value)

        with open(json_backup, 'w') as f:
            json.dump(best_params, f)

        # Export Excel untuk Run ini
        df_search_space = pd.DataFrame([
            {'Hyperparameter': 'Filters (CNN)', 'Rentang / Nilai': '{32, 64, 128}', 'Tipe': 'Categorical'},
            {'Hyperparameter': 'Kernel Size', 'Rentang / Nilai': '2 s/d 5', 'Tipe': 'Integer'},
            {'Hyperparameter': 'Units (LSTM)', 'Rentang / Nilai': '50 s/d 150', 'Tipe': 'Integer'},
            {'Hyperparameter': 'Dropout Rate', 'Rentang / Nilai': '0,1 s/d 0,4', 'Tipe': 'Float'},
            {'Hyperparameter': 'Learning Rate', 'Rentang / Nilai': '1e-4 s/d 1e-2', 'Tipe': 'Float (Log)'},
            {'Hyperparameter': 'Batch Size', 'Rentang / Nilai': '{16, 32, 64}', 'Tipe': 'Categorical'},
            {'Hyperparameter': 'Window Size', 'Rentang / Nilai': '10 s/d 60', 'Tipe': 'Integer'}
        ])

        df_best = pd.DataFrame([
            {'Model': 'CNN-LSTM + BO (Usulan)', 'Best_Trial': best_trial_cl.number + 1, 'Best_MAPE': f"{study_cl.best_value*100:.4f}%", **study_cl.best_params}
        ])

        with pd.ExcelWriter(excel_backup) as writer:
            df_search_space.to_excel(writer, sheet_name='Tabel_4.5_Rentang_Pencarian', index=False)
            df_best.to_excel(writer, sheet_name='Tabel_4.6_Hasil_Optimasi', index=False)
            
        print(f"       * Run {run_num} Selesai. Hasil disimpan ke {json_backup}")

    print("\n" + "="*85)
    print(" TAHAP 4: PEREKAMAN 5 RUN OPTIMASI SELESAI! ")
    print("="*85)
    print(f"    - Berkas hasil optimasi 1 s/d 5 berhasil disimpan di folder proyek.")
    print(f"    - Silakan jalankan Tahap 5 untuk melakukan Uji Stabilitas & komparasi final.")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_optimization_pure()
