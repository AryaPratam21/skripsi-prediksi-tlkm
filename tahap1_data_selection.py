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
    print("\n" + "="*70)
    print(" [TAHAP 1: IDENTIFIKASI OBJEK & PENGAMBILALIHAN DATASET] ")
    print(" [FASE KDD: DATA SELECTION (PEMILIHAN DATA)] ")
    print("="*70)
    
    # 1. Menentukan simbol saham (TLKM.JK) dan rentang waktu (2012-2025)
    symbol = "TLKM.JK"
    print(f">>> Menentukan simbol saham target: {symbol}")
    print(">>> Mengunduh data historis harian dari API Yahoo Finance...")
    print("    Rentang waktu penelitian: 2 Januari 2012 hingga 2 Januari 2025")
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
        'Statistik': ['Minimum', 'Maximum', 'Rata-rata (Mean)', 'Total Baris Data'],
        'Nilai (Adj Close)': [stats['min'], stats['max'], stats['mean'], float(len(df))]
    })

    with pd.ExcelWriter("tahap1_pengumpulan_data.xlsx") as writer:
        df_excel.to_excel(writer, sheet_name='Data_Historis', index_label="Date")
        df_stats.to_excel(writer, sheet_name='Statistik_Deskriptif_Bab4', index=False)
    
    print(f"\n>>> HASIL DATA SELECTION:")
    print(f"    - Berhasil mengunduh total {len(df)} baris data saham {symbol}.")
    print("    - Kolom yang dipilih untuk multivariat: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']")
    print("    - Target prediksi ditetapkan pada kolom: 'Adj Close'")
    
    print("\n[TAMPILAN DATA UNTUK TABEL 4.1: STATISTIK DESKRIPTIF (ADJ CLOSE)]")
    print(df_stats.to_string(index=False))
    
    print("\n>>> File Data Mentah Terbentuk: data_raw.csv")
    print(">>> File Excel Laporan Bab IV  : tahap1_pengumpulan_data.xlsx (Sheet: Statistik_Deskriptif_Bab4)")
    print("="*70 + "\n")

if __name__ == "__main__":
    collect_data()