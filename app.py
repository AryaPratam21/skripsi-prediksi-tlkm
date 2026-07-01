import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Skripsi - Optimasi CNN-LSTM TLKM",
    page_icon="📈",
    layout="wide"
)

# Judul Utama
st.title("📈 Dashboard Riset Prediksi Saham Telkom")
st.markdown("Dashboard ini mengambil data **Final** dari setiap tahap penelitian (Output: Excel)")

# --- Sidebar Navigasi ---
st.sidebar.title("🔍 Menu Navigasi")
menu = st.sidebar.radio("Pilih Tahap:", [
    "Tahap 1: Data Acquisition",
    "Tahap 2: Preprocessing",
    "Tahap 3: Baseline Performance",
    "Tahap 4 & 5: Bayesian Results",
    "Tahap 6: Training History",
    "Tahap 7 & 8: Hasil & Proyeksi"
])

# Pemetaan File Excel Final
excel_files = {
    "t1": "tahap1_pengumpulan_data.xlsx",
    "t2": "tahap2_hasil_preprocessing.xlsx",
    "t3": "tahap3_hasil_baseline.xlsx",
    "t4_5": "tahap4_5_hasil_optimasi.xlsx",
    "t6": "tahap6_hasil_training.xlsx",
    "t7": "tahap7_hasil_evaluasi.xlsx",
    "t8": "tahap8_hasil_prediksi.xlsx"
}

# Fungsi Membaca Data
def load_excel(file_path):
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    return None

# ------------------------------------------------------------------------------
# KONTEN DASHBOARD
# ------------------------------------------------------------------------------

if menu == "Tahap 1: Data Acquisition":
    st.header("📂 Data Mentah TLKM.JK")
    df = load_excel(excel_files["t1"])
    df_stats = None
    if os.path.exists(excel_files["t1"]):
        df_stats = pd.read_excel(excel_files["t1"], sheet_name='Statistik_Deskriptif_Bab4')

    if df is not None:
        df['Date'] = pd.to_datetime(df['Date'])
        st.subheader("Sampel Data Historis (10 Baris Terakhir)")
        st.dataframe(df.tail(10), width="stretch") # Ganti use_container_width
        
        if df_stats is not None:
            st.subheader("Tabel 4.1: Statistik Deskriptif (Adj Close)")
            st.table(df_stats.astype(str)) # Konversi ke string untuk hindari Arrow Error

elif menu == "Tahap 2: Preprocessing":
    st.header("⚙️ Hasil Pra-pemrosesan Data")
    df = load_excel(excel_files["t2"])
    if df is not None:
        st.write("Data Multivariat dengan seleksi fitur (OHLCV + Adj Close):")
        st.dataframe(df.tail(10), width="stretch")
    else:
        st.error(f"File {excel_files['t2']} tidak ditemukan.")

elif menu == "Tahap 3: Baseline Performance":
    st.header("📉 Performa Model Tunggal (Baseline)")
    df = load_excel(excel_files["t3"])
    if df is not None:
        # PENTING: Gunakan astype(str) agar tanda '-' tidak menyebabkan error Arrow
        st.table(df.astype(str))
    else:
        st.error(f"File {excel_files['t3']} tidak ditemukan.")

elif menu == "Tahap 4 & 5: Bayesian Results":
    st.header("🧠 Hyperparameter Hasil Optimasi")
    df = load_excel(excel_files["t4_5"])
    if df is not None:
        st.table(df.astype(str)) # Gunakan astype(str)
    else:
        st.error(f"File {excel_files['t4_5']} tidak ditemukan.")

elif menu == "Tahap 6: Training History":
    st.header("🚀 History Pelatihan Model Usulan")
    df_history = None
    if os.path.exists(excel_files["t6"]):
        try:
            df_history = pd.read_excel(excel_files["t6"], sheet_name='Training_History')
        except:
            df_history = pd.read_excel(excel_files["t6"], sheet_name=0)
    
    if df_history is not None:
        st.subheader("Kurva Loss (Pelatihan)")
        fig, ax = plt.subplots(figsize=(10, 4))
        epochs = range(1, len(df_history) + 1)
        loss_col = 'loss' if 'loss' in df_history.columns else df_history.columns[1]
        ax.plot(epochs, df_history[loss_col], label='Loss (Train)')
        if 'val_loss' in df_history.columns:
            ax.plot(epochs, df_history['val_loss'], label='Loss (Validation)', linestyle='--')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.error(f"File {excel_files['t6']} tidak ditemukan.")

elif menu == "Tahap 7 & 8: Hasil & Proyeksi":
    st.header("🏆 Evaluasi Akhir & Proyeksi 7 Hari")
    
    # 1. Laporan Akhir (Tabel 4.7)
    df_eval = load_excel(excel_files["t7"])
    if df_eval is not None:
        st.subheader("Tabel 4.7: Komparasi Performa Seluruh Model")
        st.dataframe(df_eval.astype(str), width="stretch")
        
    # 2. Prediksi 7 Hari (Tabel 4.8)
    file_t8 = excel_files["t8"]
    if os.path.exists(file_t8):
        try:
            # Deteksi Nama Sheet Otomatis (Tetap dipertahankan agar anti-error)
            xl = pd.ExcelFile(file_t8)
            daftar_sheet = xl.sheet_names
            sheet_target = next((s for s in daftar_sheet if 'Tabel' in s), daftar_sheet[0])
            
            # Baca data
            df_future = pd.read_excel(file_t8, sheet_name=sheet_target)
            
            if df_future is not None:
                # Bersihkan kolom
                df_future.columns = df_future.columns.str.strip()
                df_future['Tanggal'] = pd.to_datetime(df_future['Tanggal'])
                
                # TAMPILKAN JUDUL TABEL (Lebih Akademis daripada notifikasi st.success)
                st.subheader(f"Tabel 4.8: Proyeksi 7 Hari ke Depan (Data dari {sheet_target})")
                st.dataframe(df_future, width="stretch")
                
                # Visualisasi
                st.subheader("Grafik Proyeksi Harga Januari 2025")
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Deteksi Nama Kolom
                col_asli = [c for c in df_future.columns if 'Asli' in c][0]
                col_pred = [c for c in df_future.columns if 'Prediksi' in c][0]
                
                ax.plot(df_future['Tanggal'], df_future[col_asli], marker='o', label='Harga Asli', color='black')
                ax.plot(df_future['Tanggal'], df_future[col_pred], marker='D', label='Prediksi CNN-LSTM + BO', linestyle='--', color='red')
            
                ax.set_ylabel("Harga (Rp)"); ax.set_xlabel("Tanggal"); ax.legend(); ax.grid(True, alpha=0.3)
                plt.xticks(rotation=15)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal memproses data proyeksi: {e}")
    else:
        st.error(f"File {file_t8} tidak ditemukan.")

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Penelitian Prediksi Harga Saham PT TELKOM")
