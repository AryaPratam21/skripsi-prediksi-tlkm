import numpy as np
import tensorflow as tf
import random
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def set_seeds(seed=42):
    """Memastikan hasil eksperimen konsisten setiap kali dijalankan (Sesuai Bab III)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Memaksa TensorFlow menggunakan satu thread agar hasil identik
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def evaluate_metrics(y_true, y_pred):
    """Menghitung 5 metrik evaluasi sesuai Kerangka Berpikir: MSE, MAE, RMSE, MAPE, dan R^2."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # MAPE dalam persen
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': round(mse, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4),
        'R2': round(r2, 4)
    }
