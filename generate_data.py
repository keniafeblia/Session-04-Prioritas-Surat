"""
generate_data.py
Membuat dataset simulasi surat masuk kantor (300 surat) lengkap dengan label prioritas.
Sengaja disisipkan beberapa nilai kosong (missing value) agar tahap pembersihan data bisa dipraktikkan.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 300

jenis = rng.choice(["Undangan", "Permohonan", "Pemberitahuan", "Laporan", "Edaran"], N)
asal = rng.choice(["Internal", "Mitra", "Instansi Pemerintah", "Masyarakat"], N)
hari_tenggat = rng.integers(1, 31, N)             # sisa hari sampai batas waktu
jumlah_lampiran = rng.integers(0, 6, N)
kata_segera = rng.choice([0, 1], N, p=[0.7, 0.3])  # 1 = perihal memuat kata "segera/penting"
perlu_ttd = rng.choice([0, 1], N, p=[0.6, 0.4])    # 1 = butuh tanda tangan pimpinan

# Aturan pemberian skor (seolah-olah dari keputusan staf TU) + sedikit noise
skor = (
    (hari_tenggat <= 3) * 3 + ((hari_tenggat > 3) & (hari_tenggat <= 7)) * 1.5
    + kata_segera * 2 + perlu_ttd * 1.5
    + (asal == "Instansi Pemerintah") * 1.5 + (jenis == "Undangan") * 1
    + rng.normal(0, 0.8, N)
)
prioritas = np.where(skor >= 5, "Tinggi", np.where(skor >= 2.5, "Sedang", "Rendah"))

df = pd.DataFrame({
    "id_surat": [f"SM-{i:03d}" for i in range(1, N + 1)],
    "jenis_surat": jenis,
    "asal_surat": asal,
    "hari_tenggat": hari_tenggat.astype(float),
    "jumlah_lampiran": jumlah_lampiran.astype(float),
    "kata_segera": kata_segera,
    "perlu_ttd_pimpinan": perlu_ttd,
    "prioritas": prioritas,
})

# Sisipkan missing value (~5%) pada dua kolom
for kolom in ["hari_tenggat", "jumlah_lampiran"]:
    idx = rng.choice(N, 15, replace=False)
    df.loc[idx, kolom] = np.nan

df.to_csv("data/surat_masuk.csv", index=False)
print("Dataset tersimpan di data/surat_masuk.csv")
print(df["prioritas"].value_counts())
