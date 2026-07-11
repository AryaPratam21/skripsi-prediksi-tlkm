import numpy as np
import pandas as pd
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Bidirectional, MaxPooling1D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from utils import set_seeds

def create_dataset(X_data, y_data, window):
    X, y = [], []
    for i in range(len(X_data) - window):
        X.append(X_data[i:(i + window), :])
        y.append(y_data[i + window, 0])
    return np.array(X), np.array(y)

def train_final():
    """
    Fungsi untuk melatih model final secara mendalam menggunakan 
    parameter terbaik yang ditemukan oleh Bayesian Optimization.
    """
    print("\n" + "="*70)
    print(" [TAHAP 6: FINALISASI PELATIHAN MODEL OPTIMAL] ")
    print(" [FASE KDD: DATA MINING - MODEL TRAINING] ")
    print("="*70)
    
    print("\n>>> PENJELASAN METODOLOGI (UNTUK BAB IV):")
    print("    - Pada tahap ini, model final dilatih lebih mendalam (hingga 100 Epochs).")
    print("    - Menggunakan konfigurasi hyperparameter optimal hasil optimasi Bayesian (Optuna).")
    print("    - Menggunakan callback EarlyStopping dengan toleransi (patience) 15 untuk mencegah overfitting.")
    
    set_seeds()
    
    data = np.load("processed_data.npz")
    tr_X, tr_y = data['train_X_scaled'], data['train_y_scaled']
    ts_X, ts_y = data['test_X_scaled'], data['test_y_scaled']
    
    # 1. Memuat hyperparameter terbaik
    if not os.path.exists('best_params.json'):
        print("Error: best_params.json tidak ditemukan! Jalankan Tahap 4 & 5 terlebih dahulu.")
        return
        
    with open('best_params.json', 'r') as f:
        best = json.load(f)
        
    print("\n>>> MEMUAT HYPERPARAMETER OPTIMAL DARI TAHAP 4 & 5:")
    print(f"    - BiLSTM + BO (Trial ke-{best['bilstm'].get('best_trial', 'N/A')})      : Window={best['bilstm']['window_size']}, Filters={best['bilstm']['filters']}, Units={best['bilstm']['units']}, LR={best['bilstm']['lr']:.4f}, Batch={best['bilstm']['batch_size']}")
    print(f"    - CNN-LSTM + BO (Usulan, Trial ke-{best['cnn_lstm'].get('best_trial', 'N/A')}): Window={best['cnn_lstm']['window_size']}, Filters={best['cnn_lstm']['filters']}, Units={best['cnn_lstm']['units']}, LR={best['cnn_lstm']['lr']:.4f}, Batch={best['cnn_lstm']['batch_size']}")

    # 2. Melatih Model Pembanding (BiLSTM + BO)
    print("\n>>> [1/2] Melatih Model BiLSTM + BO (100 Epochs)...")
    ws_bi = best['bilstm']['window_size']
    X_train_bi, y_train_bi = create_dataset(tr_X, tr_y, ws_bi)
    X_test_bi, y_test_bi = create_dataset(ts_X, ts_y, ws_bi)
    
    print(f"    - Dimensi Data Latih BiLSTM (X_train): {X_train_bi.shape} | (y_train): {y_train_bi.shape}")
    m_bi = Sequential([
        Input(shape=(ws_bi, X_train_bi.shape[2])),
        Conv1D(best['bilstm']['filters'], 3, padding='same', activation='relu'),
        MaxPooling1D(2),
        Bidirectional(LSTM(best['bilstm']['units'])),
        Dropout(best['bilstm']['dropout']),
        Dense(1)
    ])
    m_bi.compile(optimizer=Adam(best['bilstm']['lr']), loss=Huber())
    print("    - Mulai pelatihan Model BiLSTM...")
    # Kita latih 100 epochs, verbose=2 agar log detail per epoch terlihat di terminal
    history_bi = m_bi.fit(X_train_bi, y_train_bi, epochs=100, batch_size=best['bilstm']['batch_size'], verbose=2)
    print(f"    Selesai. Loss Akhir BiLSTM: {history_bi.history['loss'][-1]:.6f}")

    y_pred_bi = m_bi.predict(X_test_bi, verbose=0)
    np.save("y_pred_bi.npy", y_pred_bi)
    np.save("y_true_bi.npy", y_test_bi)
    print(f"    Data BiLSTM disimpan (y_pred_bi.npy dan y_true_bi.npy).")

    # 3. Melatih Model Usulan (CNN-LSTM + BO)
    print("\n>>> [2/2] Melatih Model CNN-LSTM + BO (Usulan - 100 Epochs)...")
    ws_cl = best['cnn_lstm']['window_size']
    X_train_cl, y_train_cl = create_dataset(tr_X, tr_y, ws_cl)
    X_test_cl, y_test_cl = create_dataset(ts_X, ts_y, ws_cl)

    print(f"    - Dimensi Data Latih CNN-LSTM (X_train): {X_train_cl.shape} | (y_train): {y_train_cl.shape}")
    m_usulan = Sequential([
        Input(shape=(ws_cl, X_train_cl.shape[2])),
        Conv1D(best['cnn_lstm']['filters'], 3, padding='same', activation='relu'),
        MaxPooling1D(2),
        LSTM(best['cnn_lstm']['units']),
        Dropout(best['cnn_lstm']['dropout']),
        Dense(1)
    ])
    m_usulan.compile(optimizer=Adam(best['cnn_lstm']['lr']), loss=Huber())

    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("    - Mulai pelatihan Model CNN-LSTM (dengan EarlyStopping)...")
    history = m_usulan.fit(X_train_cl, y_train_cl, 
                           epochs=100, 
                           batch_size=best['cnn_lstm']['batch_size'],
                           validation_data=(X_test_cl, y_test_cl), 
                           callbacks=[es], 
                           verbose=2) # verbose=2 agar log detail per epoch terlihat di terminal

    # 5. Export History, Model, Prediksi, dan Tabel 4.6
    # Buat DataFrame untuk Tabel 4.6
    df_params = pd.DataFrame([
        {'Model': 'BiLSTM + BO', 'Best_Trial': best['bilstm'].get('best_trial', '-'), **{k: v for k, v in best['bilstm'].items() if k != 'best_trial'}},
        {'Model': 'CNN-LSTM + BO (Usulan)', 'Best_Trial': best['cnn_lstm'].get('best_trial', '-'), **{k: v for k, v in best['cnn_lstm'].items() if k != 'best_trial'}}
    ])

    with pd.ExcelWriter("tahap6_hasil_training.xlsx") as writer:
        pd.DataFrame(history.history).to_excel(writer, sheet_name='Training_History', index=False)
        df_params.to_excel(writer, sheet_name='Tabel_4.6_Best_Hyperparams', index=False)
    
    m_usulan.save("model_usulan.h5")
    
    y_pred_usulan = m_usulan.predict(X_test_cl, verbose=0)
    np.save("y_pred_usulan.npy", y_pred_usulan)
    np.save("y_true_usulan.npy", y_test_cl)
    
    epochs_run = len(history.history['loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"\n>>> HASIL TRAINING MODEL USULAN (CNN-LSTM + BO):")
    print(f"    - Total Epoch Terlaksana  : {epochs_run} Epochs (Training berhenti otomatis)")
    print(f"    - Epoch Terbaik           : Epoch ke-{best_epoch} (Bobot dengan Val Loss terendah dikembalikan)")
    print(f"    - Loss Pelatihan (Huber)  : {history.history['loss'][-1]:.6f}")
    print(f"    - Loss Validasi (Huber)   : {history.history['val_loss'][-1]:.6f}")
    print("\n>>> Final training selesai. Model 'model_usulan.h5' siap digunakan.")

if __name__ == "__main__":
    train_final()
