import numpy as np
import pandas as pd
import json
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
    print("[TAHAP 6: FINALISASI PELATIHAN MODEL OPTIMAL]")
    print("="*70)
    set_seeds()
    
    data = np.load("processed_data.npz")
    tr_X, tr_y = data['train_X_scaled'], data['train_y_scaled']
    ts_X, ts_y = data['test_X_scaled'], data['test_y_scaled']
    
    # 1. Memuat hyperparameter terbaik
    # PENTING: Jika Anda ingin menggunakan hasil yang sudah ditulis di Bab 4,
    # pastikan best_params.json berisi nilai yang sesuai.
    with open('best_params.json', 'r') as f:
        best = json.load(f)

    # 2. Melatih Model Pembanding (BiLSTM + BO)
    print("\n>>> [1/2] Melatih Model BiLSTM + BO (100 Epochs)...")
    ws_bi = best['bilstm']['window_size']
    X_train_bi, y_train_bi = create_dataset(tr_X, tr_y, ws_bi)
    X_test_bi, y_test_bi = create_dataset(ts_X, ts_y, ws_bi)
    
    m_bi = Sequential([
        Input(shape=(ws_bi, X_train_bi.shape[2])),
        Conv1D(best['bilstm']['filters'], 3, padding='same', activation='relu'),
        MaxPooling1D(2),
        Bidirectional(LSTM(best['bilstm']['units'])),
        Dropout(best['bilstm']['dropout']),
        Dense(1)
    ])
    m_bi.compile(optimizer=Adam(best['bilstm']['lr']), loss=Huber())
    m_bi.fit(X_train_bi, y_train_bi, epochs=100, batch_size=best['bilstm']['batch_size'], verbose=1)

    y_pred_bi = m_bi.predict(X_test_bi, verbose=0)
    np.save("y_pred_bi.npy", y_pred_bi)
    np.save("y_true_bi.npy", y_test_bi)
    print(f"    Selesai. Data BiLSTM disimpan.")

    # 3. Melatih Model Usulan (CNN-LSTM + BO)
    print("\n>>> [2/2] Melatih Model CNN-LSTM + BO (Usulan - 100 Epochs)...")
    ws_cl = best['cnn_lstm']['window_size']
    X_train_cl, y_train_cl = create_dataset(tr_X, tr_y, ws_cl)
    X_test_cl, y_test_cl = create_dataset(ts_X, ts_y, ws_cl)

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

    print("    Proses training sedang berjalan (dengan EarlyStopping)...")
    history = m_usulan.fit(X_train_cl, y_train_cl, 
                           epochs=100, 
                           batch_size=best['cnn_lstm']['batch_size'],
                           validation_data=(X_test_cl, y_test_cl), 
                           callbacks=[es], 
                           verbose=1)

    
    # 5. Export History, Model, Prediksi, dan Tabel 4.6
    # Buat DataFrame untuk Tabel 4.6
    df_params = pd.DataFrame([
        {'Model': 'BiLSTM + BO', **best['bilstm']},
        {'Model': 'CNN-LSTM + BO (Usulan)', **best['cnn_lstm']}
    ])

    with pd.ExcelWriter("tahap6_hasil_training.xlsx") as writer:
        pd.DataFrame(history.history).to_excel(writer, sheet_name='Training_History', index=False)
        df_params.to_excel(writer, sheet_name='Tabel_4.6_Best_Hyperparams', index=False)
    
    m_usulan.save("model_usulan.h5")
    
    y_pred_usulan = m_usulan.predict(X_test_cl, verbose=0)
    np.save("y_pred_usulan.npy", y_pred_usulan)
    np.save("y_true_usulan.npy", y_test_cl)
    
    print(f"    Selesai. Training berhenti pada epoch ke-{len(history.history['loss'])}.")
    print(f"    Loss Akhir: {history.history['loss'][-1]:.6f} | Val Loss: {history.history['val_loss'][-1]:.6f}")
    print("\n>>> Final training selesai. Model 'model_usulan.h5' siap digunakan.")

if __name__ == "__main__":
    train_final()
