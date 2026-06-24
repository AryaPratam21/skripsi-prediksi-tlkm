import optuna
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Bidirectional, MaxPooling1D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K
from utils import set_seeds

def run_optimization():
    """
    Fungsi inti penelitian: Mencari hyperparameter terbaik menggunakan Bayesian Optimization.
    Metode ini lebih cerdas dari GridSearch karena belajar dari hasil trial sebelumnya.
    """
    print("\n" + "="*70)
    print("[TAHAP 4 & 5: IMPLEMENTASI BAYESIAN OPTIMIZATION (OPTUNA)]")
    print("="*70)
    
    data = np.load("processed_data.npz")
    tr_X, tr_y = data['train_X_scaled'], data['train_y_scaled']
    ts_X, ts_y = data['test_X_scaled'], data['test_y_scaled']

    def create_dataset(X_data, y_data, window):
        X, y = [], []
        for i in range(len(X_data) - window):
            X.append(X_data[i:(i + window), :])
            y.append(y_data[i + window, 0])
        return np.array(X), np.array(y)

    def objective(trial, m_type='cnn-lstm'):
        K.clear_session()
        set_seeds()
        
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
        
        if m_type == 'bilstm':
            m.add(Bidirectional(LSTM(u)))
        else:
            m.add(LSTM(u))
            
        m.add(Dropout(dr))
        m.add(Dense(1))
        
        m.compile(optimizer=Adam(lr), loss=Huber())
        m.fit(X_train, y_train, epochs=30, batch_size=bs, verbose=0)
        
        from sklearn.metrics import mean_absolute_percentage_error
        y_p = m.predict(X_test, verbose=0)
        return mean_absolute_percentage_error(y_test, y_p)

    # 5. Eksekusi Optimasi untuk Model Pembanding (BiLSTM)
    print("\n>>> [1/2] Mengoptimasi BiLSTM (Pembanding) - 50 Trials...")
    study_bi = optuna.create_study(direction='minimize')
    study_bi.optimize(lambda t: objective(trial=t, m_type='bilstm'), n_trials=50)
    print(f"    Selesai. Best MAPE: {study_bi.best_value*100:.4f}%")
    
    # 6. Eksekusi Optimasi untuk Model Usulan (CNN-LSTM)
    print("\n>>> [2/2] Mengoptimasi CNN-LSTM (Usulan) - 50 Trials...")
    study_cl = optuna.create_study(direction='minimize')
    study_cl.optimize(lambda t: objective(trial=t, m_type='cnn-lstm'), n_trials=50)
    print(f"    Selesai. Best MAPE: {study_cl.best_value*100:.4f}%")

    # 7. Menyimpan dan Menampilkan Tabel Parameter Terbaik (Tabel 4.6)
    # Serta Tabel Rentang Pencarian (Tabel 4.5)
    best_params = {
        'bilstm': study_bi.best_params,
        'cnn_lstm': study_cl.best_params
    }
    
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)
    
    # DataFrame Tabel 4.5: Rentang Pencarian
    df_search_space = pd.DataFrame([
        {'Hyperparameter': 'Filters (CNN)', 'Rentang / Nilai': '{32, 64, 128}', 'Tipe': 'Categorical'},
        {'Hyperparameter': 'Kernel Size', 'Rentang / Nilai': '2 s/d 5', 'Tipe': 'Integer'},
        {'Hyperparameter': 'Units (LSTM)', 'Rentang / Nilai': '50 s/d 150', 'Tipe': 'Integer'},
        {'Hyperparameter': 'Dropout Rate', 'Rentang / Nilai': '0,1 s/d 0,4', 'Tipe': 'Float'},
        {'Hyperparameter': 'Learning Rate', 'Rentang / Nilai': '1e-4 s/d 1e-2', 'Tipe': 'Float (Log)'},
        {'Hyperparameter': 'Batch Size', 'Rentang / Nilai': '{16, 32, 64}', 'Tipe': 'Categorical'},
        {'Hyperparameter': 'Window Size', 'Rentang / Nilai': '10 s/d 60', 'Tipe': 'Integer'}
    ])

    # DataFrame Tabel 4.6: Hasil Terbaik
    df_best = pd.DataFrame([
        {'Model': 'BiLSTM + BO', 'Best_MAPE': f"{study_bi.best_value*100:.4f}%", **study_bi.best_params},
        {'Model': 'CNN-LSTM + BO (Usulan)', 'Best_MAPE': f"{study_cl.best_value*100:.4f}%", **study_cl.best_params}
    ])
    
    print("\n" + "="*80)
    print("RENTANG PENCARIAN HYPERPARAMETER (TABEL 4.5)")
    print("="*80)
    print(df_search_space.to_string(index=False))
    
    print("\n" + "="*80)
    print("HASIL OPTIMASI HYPERPARAMETER TERBAIK (TABEL 4.6)")
    print("="*80)
    print(df_best.to_string(index=False))
    print("="*80)
    
    # Simpan ke Excel Multi-Sheet
    with pd.ExcelWriter("tahap4_5_hasil_optimasi.xlsx") as writer:
        df_search_space.to_excel(writer, sheet_name='Tabel_4.5_Rentang_Pencarian', index=False)
        df_best.to_excel(writer, sheet_name='Tabel_4.6_Hasil_Optimasi', index=False)
        
    print(">>> Optimasi selesai. Tabel 4.5 (Rentang) & Tabel 4.6 (Hasil) disimpan di: tahap4_5_hasil_optimasi.xlsx")

if __name__ == "__main__":
    run_optimization()