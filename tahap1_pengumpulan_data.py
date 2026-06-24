import yfinance as yf
import pandas as pd
import warnings

# Menghilangkan peringatan sistem agar terminal bersih
warnings.filterwarnings("ignore")

def collect_data():
    """
    Fungsi untuk mengambil data mentah dari Yahoo Finance.
    Sesuai dengan Bab III Metodologi Penelitian.
    """
    print("\n" + "="*60)
    print(" [TAHAP 1: IDENTIFIKASI OBJEK & PENGAMBILALIHAN DATASET] ")
    print("="*60)
    
    # 1. Menentukan simbol saham (TLKM.JK) dan rentang waktu (2012-2025)
    symbol = "TLKM.JK"
    df = yf.download(symbol, start="2012-01-02", end="2025-01-02", progress=False, auto_adjust=False)
    
    # 2. Menangani format kolom jika data berbentuk MultiIndex
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    
    # 3. Penyimpanan Data untuk Sistem (CSV)
    df.to_csv("data_raw.csv")
    
    # --- PERBAIKAN TAMPILAN UNTUK TERMINAL ---
    # Jika ada MultiIndex 'Price', kita ratakan agar tidak membingungkan
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # 4. Penyimpanan Khusus untuk Laporan Excel (Multi-Sheet)
    df_excel = df.copy()
    df_excel.index = df_excel.index.strftime('%Y-%m-%d')
    
    # Hitung Statistik Deskriptif untuk Tabel 4.1
    stats = df['Adj Close'].describe()
    df_stats = pd.DataFrame({
        'Statistik': ['Minimum', 'Maximum', 'Rata-rata (Mean)'],
        'Nilai (Adj Close)': [stats['min'], stats['max'], stats['mean']]
    })

    with pd.ExcelWriter("tahap1_pengumpulan_data.xlsx") as writer:
        df_excel.to_excel(writer, sheet_name='Data_Historis', index_label="Date")
        df_stats.to_excel(writer, sheet_name='Statistik_Deskriptif_Bab4', index=False)
    
    print(f">>> Berhasil mengunduh {len(df)} baris data {symbol}.")
    print("\n[STATISTIK DESKRIPTIF ADJ CLOSE]")
    print(df_stats.to_string(index=False))
    
    print("\n>>> File Sistem  : data_raw.csv")
    print(">>> File Laporan : tahap1_pengumpulan_data.xlsx")
    print("="*60 + "\n")

if __name__ == "__main__":
    collect_data()