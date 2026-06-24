import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Input
from utils import set_seeds, evaluate_metrics

def train_baseline():
    """
    Fungsi untuk melatih model standar (CNN saja, LSTM saja, CNN-LSTM manual)
    sebagai benchmark/pembanding awal di Bab IV.
    """
    print("\n" + "="*60)
    print("[TAHAP 3: PELATIHAN MODEL BASELINE (MANUAL)]")
    print("="*60)
    set_seeds() # Memastikan hasil eksperimen konsisten/reproducible
    
    # 1. Load data hasil preprocessing
    data = np.load("processed_data.npz")
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    scaler_y = joblib.load('scaler_y.pkl')

    results = []
    # Parameter default untuk baseline manual
    window_size = 30
    batch_size = 32
    epochs = 30
    lr = 0.001

    # 2. Model 1: CNN Saja (Arsitektur dasar tanpa LSTM)
    # 2. Model 1: CNN Saja
    print("\n>>> [1/3] Melatih Model CNN Saja...")
    f, k = 64, 3
    m_cnn = Sequential([
        Input(shape=(window_size, X_train.shape[2])),
        Conv1D(f, k, activation='relu'),
        Flatten(), 
        Dense(1)
    ])
    m_cnn.compile(optimizer='adam', loss='mse')
    # Tambahkan validation_split sedikit untuk melihat performa saat training
    m_cnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    y_pred_cnn = m_cnn.predict(X_test, verbose=0)
    res_cnn = evaluate_metrics(scaler_y.inverse_transform(y_test.reshape(-1,1)), 
                               scaler_y.inverse_transform(y_pred_cnn))
    results.append({
        'Model': 'CNN Saja (Manual)',
        'Filters': f, 'Kernel': k, 'Units': '-', 'Dropout': '-', 'Window': window_size, 'LR': lr, 'Batch': batch_size,
        **res_cnn
    })
    print(f"    Selesai. MAPE: {res_cnn['MAPE']:.4f}% | RMSE: {res_cnn['RMSE']:.2f}")

    # 3. Model 2: LSTM Saja
    print("\n>>> [2/3] Melatih Model LSTM Saja...")
    u = 64
    m_lstm = Sequential([
        Input(shape=(window_size, X_train.shape[2])),
        LSTM(u),
        Dense(1)
    ])
    m_lstm.compile(optimizer='adam', loss='mse')
    m_lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred_lstm = m_lstm.predict(X_test, verbose=0)
    res_lstm = evaluate_metrics(scaler_y.inverse_transform(y_test.reshape(-1,1)), 
                                scaler_y.inverse_transform(y_pred_lstm))
    results.append({
        'Model': 'LSTM Saja (Manual)',
        'Filters': '-', 'Kernel': '-', 'Units': u, 'Dropout': '-', 'Window': window_size, 'LR': lr, 'Batch': batch_size,
        **res_lstm
    })
    print(f"    Selesai. MAPE: {res_lstm['MAPE']:.4f}% | RMSE: {res_lstm['RMSE']:.2f}")

    # 4. Model 3: CNN-LSTM Manual
    print("\n>>> [3/3] Melatih Model CNN-LSTM Manual...")
    f, k, u = 64, 3, 64
    m_hybrid = Sequential([
        Input(shape=(window_size, X_train.shape[2])),
        Conv1D(f, k, activation='relu'),
        LSTM(u),
        Dense(1)
    ])
    m_hybrid.compile(optimizer='adam', loss='mse')
    m_hybrid.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    m_hybrid.save("model_manual.h5")
    print(">>> Model CNN-LSTM Manual berhasil disimpan ke model_manual.h5")
    y_pred_manual = m_hybrid.predict(X_test, verbose=0)
    res_hybrid = evaluate_metrics(scaler_y.inverse_transform(y_test.reshape(-1,1)), 
                                  scaler_y.inverse_transform(y_pred_manual))
    results.append({
        'Model': 'CNN-LSTM (Manual)', 
        'Filters': f, 'Kernel': k, 'Units': u, 'Dropout': '-', 'Window': window_size, 'LR': lr, 'Batch': batch_size,
        **res_hybrid
    })
    print(f"    Selesai. MAPE: {res_hybrid['MAPE']:.4f}% | RMSE: {res_hybrid['RMSE']:.2f}")
    
    # 5. Menampilkan Tabel Rekapitulasi Baseline
    np.save("y_pred_manual.npy", y_pred_manual)
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("HASIL EVALUASI BASELINE MODEL (DATA TESTING)")
    print("="*80)
    print(df_results.to_string(index=False, justify='center'))
    print("="*80)
    
    # Simpan ke Excel untuk Tabel Bab IV
    df_results.to_excel("tahap3_hasil_baseline.xlsx", index=False)
    print(">>> Baseline selesai. Tabel komparasi disimpan di: tahap3_hasil_baseline.xlsx")

if __name__ == "__main__":
    train_baseline()
