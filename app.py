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

# Fungsi menggambar chart loss menggunakan Chart.js (Bypass PyArrow)
def draw_chartjs_loss(df):
    epochs = list(range(1, len(df) + 1))
    loss_col = 'loss' if 'loss' in df.columns else df.columns[1]
    loss_train = list(df[loss_col].values)
    loss_val = list(df['val_loss'].values) if 'val_loss' in df.columns else []
    
    datasets = [
        {
            "label": 'Loss (Train)',
            "data": loss_train,
            "borderColor": '#36a2eb',
            "backgroundColor": 'rgba(54, 162, 235, 0.1)',
            "fill": True,
            "borderWidth": 2,
            "tension": 0.1
        }
    ]
    if loss_val:
        datasets.append({
            "label": 'Loss (Validation)',
            "data": loss_val,
            "borderColor": '#ff6384',
            "backgroundColor": 'rgba(255, 99, 132, 0.1)',
            "fill": True,
            "borderWidth": 2,
            "borderDash": [5, 5],
            "tension": 0.1
        })

    html_code = f"""
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body style="margin:0;padding:0;background-color:transparent;">
        <canvas id="lossChart" style="width:100%;height:350px;"></canvas>
        <script>
            const ctx = document.getElementById('lossChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {epochs},
                    datasets: {json.dumps(datasets)}
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch', color: '#888' }}, grid: {{ color: 'rgba(200,200,200,0.1)' }} }},
                        y: {{ title: {{ display: true, text: 'Loss', color: '#888' }}, grid: {{ color: 'rgba(200,200,200,0.1)' }} }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#888' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=380)

# Fungsi menggambar chart prediksi menggunakan Chart.js (Bypass PyArrow)
def draw_chartjs_prediction(df_future):
    col_asli = [c for c in df_future.columns if 'Asli' in c][0]
    col_usulan = [c for c in df_future.columns if 'Usulan' in c][0]
    
    dates_list = list(df_future['Tanggal'].dt.strftime('%Y-%m-%d').values)
    val_asli = list(df_future[col_asli].values)
    val_usulan = list(df_future[col_usulan].values)
    
    datasets = [
        {
            "label": 'Harga Asli',
            "data": val_asli,
            "borderColor": '#000000',
            "backgroundColor": '#000000',
            "pointRadius": 5,
            "fill": False,
            "borderWidth": 2.5
        },
        {
            "label": 'Prediksi CNN-LSTM + BO (Usulan)',
            "data": val_usulan,
            "borderColor": '#ff0000',
            "backgroundColor": '#ff0000',
            "pointStyle": 'rectRot',
            "pointRadius": 6,
            "fill": False,
            "borderWidth": 2,
            "borderDash": [5, 5]
        }
    ]

    html_code = f"""
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body style="margin:0;padding:0;background-color:transparent;">
        <canvas id="predChart" style="width:100%;height:350px;"></canvas>
        <script>
            const ctx = document.getElementById('predChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(dates_list)},
                    datasets: {json.dumps(datasets)}
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Tanggal', color: '#888' }}, grid: {{ color: 'rgba(200,200,200,0.1)' }} }},
                        y: {{ title: {{ display: true, text: 'Harga Saham (Rp)', color: '#888' }}, grid: {{ color: 'rgba(200,200,200,0.1)' }} }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#888' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=380)

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
            st.markdown(df_to_markdown(df_stats.astype(str))) # Ganti st.table dengan st.markdown untuk bypass PyArrow

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
        draw_chartjs_loss(df_history)
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
                draw_chartjs_prediction(df_future)

        except Exception as e:
            st.error(f"Gagal memproses data proyeksi: {e}")
    else:
        st.error(f"File {file_t8} tidak ditemukan.")

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Penelitian Prediksi Harga Saham PT TELKOM")
