"""
main.py
Studi Kasus: Klasifikasi Prioritas Surat Masuk Kantor dengan Decision Tree
Alur mengikuti materi Sesi 4: kumpulkan data -> bersihkan -> pilih fitur ->
bagi data 80/20 -> latih -> evaluasi -> cek overfitting -> prediksi data baru.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# 1. Pengumpulan data
df = pd.read_csv("data/surat_masuk.csv")
print("=== 1. DATA ===")
print(f"Jumlah surat: {len(df)}")

# 2. Pembersihan data: isi missing value dengan median
print("\n=== 2. PEMBERSIHAN DATA ===")
print("Missing value sebelum:\n", df.isna().sum()[df.isna().sum() > 0])
for kolom in ["hari_tenggat", "jumlah_lampiran"]:
    df[kolom] = df[kolom].fillna(df[kolom].median())
print("Missing value sesudah:", int(df.isna().sum().sum()))

# 3. Pemilihan fitur (id_surat tidak dipakai karena tidak bermakna)
fitur = ["jenis_surat", "asal_surat", "hari_tenggat", "jumlah_lampiran",
         "kata_segera", "perlu_ttd_pimpinan"]
X = pd.get_dummies(df[fitur], columns=["jenis_surat", "asal_surat"])
y = df["prioritas"]

# 4. Pembagian data 80% latih : 20% uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n=== 3. PEMBAGIAN DATA ===\nLatih: {len(X_train)} | Uji: {len(X_test)}")

# 5. Cek overfitting: bandingkan pohon tanpa batas vs pohon dibatasi
print("\n=== 4. CEK OVERFITTING ===")
for depth in [None, 4]:
    m = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    print(f"max_depth={str(depth):4} -> akurasi latih {m.score(X_train, y_train):.2f}"
          f" | akurasi uji {m.score(X_test, y_test):.2f}")

# 6. Model akhir (max_depth=4 agar tidak overfitting dan mudah dibaca)
model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== 5. EVALUASI MODEL AKHIR ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, digits=2))

label = ["Tinggi", "Sedang", "Rendah"]
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=label),
                       display_labels=label).plot(cmap="Blues")
plt.title("Confusion Matrix - Prioritas Surat")
plt.tight_layout(); plt.savefig("output/confusion_matrix.png", dpi=120); plt.close()

plt.figure(figsize=(18, 8))
plot_tree(model, feature_names=X.columns, class_names=model.classes_,
          filled=True, rounded=True, fontsize=8)
plt.tight_layout(); plt.savefig("output/pohon_keputusan.png", dpi=110); plt.close()

penting = pd.Series(model.feature_importances_, index=X.columns).sort_values()
penting[penting > 0].plot.barh(color="#2b6cb0", figsize=(7, 4))
plt.title("Fitur Paling Berpengaruh"); plt.tight_layout()
plt.savefig("output/fitur_penting.png", dpi=120); plt.close()

# 7. Prediksi surat baru
print("=== 6. PREDIKSI SURAT BARU ===")
baru = pd.DataFrame([
    {"jenis_surat": "Undangan", "asal_surat": "Instansi Pemerintah", "hari_tenggat": 2,
     "jumlah_lampiran": 1, "kata_segera": 1, "perlu_ttd_pimpinan": 1},
    {"jenis_surat": "Laporan", "asal_surat": "Internal", "hari_tenggat": 20,
     "jumlah_lampiran": 3, "kata_segera": 0, "perlu_ttd_pimpinan": 0},
])
X_baru = pd.get_dummies(baru, columns=["jenis_surat", "asal_surat"]).reindex(
    columns=X.columns, fill_value=0)
for (_, s), p in zip(baru.iterrows(), model.predict(X_baru)):
    print(f"- {s.jenis_surat} dari {s.asal_surat}, tenggat {s.hari_tenggat} hari -> {p}")

print("\nGrafik tersimpan di folder output/")
