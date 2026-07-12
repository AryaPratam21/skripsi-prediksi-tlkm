import faulthandler
faulthandler.enable()
import streamlit as st
import pandas as pd
import numpy as np
import os
import json

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Skripsi - Optimasi CNN-LSTM TLKM",
    page_icon="📈",
    layout="wide"
)

# Fungsi Utilitas untuk mengubah DataFrame menjadi tabel Markdown (Bypass PyArrow)
def df_to_markdown(df):
    headers = " | ".join(map(str, df.columns))
    separator = " | ".join(["---"] * len(df.columns))
    rows = []
    for _, row in df.iterrows():
        row_str = []
        for val in row.values:
            val_cleaned = str(val).replace('|', '\\|').replace('\n', ' ')
            row_str.append(val_cleaned)
        rows.append(" | ".join(row_str))
    return f"| {headers} |\n| {separator} |\n" + "\n".join([f"| {r} |" for r in rows])

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
        st.markdown(df_to_markdown(df.tail(10))) # Menggunakan Markdown untuk bypass PyArrow / Segfault
        
        if df_stats is not None:
            st.subheader("Tabel 4.1: Statistik Deskriptif (Adj Close)")
            st.table(df_stats.astype(str)) # Konversi ke string untuk hindari Arrow Error

elif menu == "Tahap 2: Preprocessing":
    st.header("⚙️ Hasil Pra-pemrosesan Data")
    df = load_excel(excel_files["t2"])
    if df is not None:
        st.write("Data Multivariat dengan seleksi fitur (OHLCV + Adj Close):")
        st.markdown(df_to_markdown(df.tail(10)))
    else:
        st.error(f"File {excel_files['t2']} tidak ditemukan.")

elif menu == "Tahap 3: Baseline Performance":
    st.header("📉 Performa Model Tunggal (Baseline)")
    df = load_excel(excel_files["t3"])
    if df is not None:
        st.markdown(df_to_markdown(df.astype(str)))
    else:
        st.error(f"File {excel_files['t3']} tidak ditemukan.")

elif menu == "Tahap 4 & 5: Bayesian Results":
    st.header("🧠 Hyperparameter Hasil Optimasi (Tabel 4.6)")
    df = None
    if os.path.exists(excel_files["t4_5"]):
        try:
            df = pd.read_excel(excel_files["t4_5"], sheet_name='Tabel_4.6_Hasil_Optimasi')
        except:
            df = pd.read_excel(excel_files["t4_5"], sheet_name=1)
    if df is not None:
        st.markdown(df_to_markdown(df.astype(str)))
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
        st.subheader("Kurva Loss (Pelatihan) - Interaktif")
        loss_col = 'loss' if 'loss' in df_history.columns else df_history.columns[1]
        chart_cols = [loss_col]
        if 'val_loss' in df_history.columns:
            chart_cols.append('val_loss')
        
        # Buat DataFrame khusus untuk line chart
        df_chart = df_history[chart_cols].copy()
        df_chart.index = range(1, len(df_chart) + 1) # Set index sebagai Epoch (dimulai dari 1)
        st.line_chart(df_chart)
    else:
        st.error(f"File {excel_files['t6']} tidak ditemukan.")

elif menu == "Tahap 7 & 8: Hasil & Proyeksi":
    st.header("🏆 Evaluasi Akhir & Proyeksi 7 Hari")
    
    # 1. Laporan Akhir (Tabel 4.7)
    df_eval = load_excel(excel_files["t7"])
    if df_eval is not None:
        st.subheader("Tabel 4.7: Komparasi Performa Seluruh Model")
        st.markdown(df_to_markdown(df_eval.astype(str)))
        
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
                st.markdown(df_to_markdown(df_future))
                # Visualisasi
                st.subheader("Grafik Proyeksi Harga Januari 2025 (Interaktif)")
                
                # Deteksi Nama Kolom
                col_asli = [c for c in df_future.columns if 'Asli' in c][0]
                col_usulan = [c for c in df_future.columns if 'Usulan' in c][0]
                
                # Buat DataFrame khusus untuk line chart dengan index tanggal
                df_chart_future = df_future.set_index('Tanggal')[[col_asli, col_usulan]].copy()
                df_chart_future.columns = ['Harga Asli', 'Prediksi CNN-LSTM + BO (Usulan)']
                st.line_chart(df_chart_future)

        except Exception as e:
            st.error(f"Gagal memproses data proyeksi: {e}")
    else:
        st.error(f"File {file_t8} tidak ditemukan.")

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Penelitian Prediksi Harga Saham PT TELKOM")