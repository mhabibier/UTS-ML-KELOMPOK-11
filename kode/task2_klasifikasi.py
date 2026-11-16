"""Task 2 - Klasifikasi: deteksi transaksi fraud.

Tahap ini (PR 1) berisi audit data:
  1. kualitas data (nilai hilang, ID ganda, nilai tidak wajar, kategori tidak konsisten),
  2. keterlambatan label (tingkat fraud per minggu),
  3. uji kebocoran fitur (dampak membuang kolom yang dicurigai).

Cara menjalankan dari akar repo:
    python kode/task2_klasifikasi.py
    python kode/task2_klasifikasi.py --berkas klasifikasi_train_sampel.csv.gz

Semua tabel dan gambar ditulis ke kode/hasil/.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # menggambar ke berkas, tanpa jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HASIL_DIR = Path(__file__).resolve().parent / "hasil"

TARGET = "is_fraud"
KOLOM_ID = ["id_transaksi", "id_pengguna", "id_merchant"]
KOLOM_KATEGORI = ["kategori_merchant", "kanal", "tipe_perangkat", "email_domain"]
KOLOM_BINER = [
    "hari_libur", "perangkat_baru", "negara_ip_beda", "kartu_tidak_hadir",
    "flag_review_manual", TARGET,
]
# Kolom yang dicurigai bocor berdasarkan kamus data (harus dibuktikan dengan data)
KANDIDAT_BOCOR = [
    "flag_review_manual",
    "jumlah_akun_per_perangkat",
    "refund_merchant_30h",
]


# --------------------------------------------------------------------------
# Muat data
# --------------------------------------------------------------------------
def muat_data(nama_berkas):
    """Membaca data latih dan mengubah kolom waktu menjadi tipe tanggal."""
    jalur = DATA_DIR / nama_berkas
    if not jalur.exists():
        raise FileNotFoundError(f"Berkas tidak ditemukan: {jalur}")
    df = pd.read_csv(jalur)
    df["waktu"] = pd.to_datetime(df["waktu"], errors="coerce")
    return df


# --------------------------------------------------------------------------
# 1. Audit kualitas data
# --------------------------------------------------------------------------
def audit_kualitas(df):
    """Mencetak dan menyimpan bukti masalah data: jumlah baris dan contohnya."""
    temuan = []

    def catat(masalah, jumlah, tindakan):
        temuan.append({"masalah": masalah, "jumlah_baris": int(jumlah), "tindakan": tindakan})

    print("=== 1. AUDIT KUALITAS DATA ===")
    print(f"Jumlah baris  : {len(df):,}")
    print(f"Jumlah kolom  : {df.shape[1]}")
    print(f"Rentang waktu : {df['waktu'].min()} s.d. {df['waktu'].max()}")
    print(f"Tingkat fraud : {df[TARGET].mean():.4%} ({int(df[TARGET].sum()):,} transaksi)")

    catat("waktu tidak bisa dibaca (NaT)", df["waktu"].isna().sum(),
          "periksa format tanggal; baris dibuang bila jumlahnya kecil")

    # nilai hilang per kolom
    hilang = df.isna().sum()
    hilang = hilang[hilang > 0].sort_values(ascending=False)
    print(f"\nKolom dengan nilai hilang: {len(hilang)}")
    if len(hilang):
        tabel_hilang = pd.DataFrame({"jumlah": hilang, "persen": hilang / len(df) * 100})
        print(tabel_hilang.round(3).head(15).to_string())
        tabel_hilang.to_csv(HASIL_DIR / "t2_audit_nilai_hilang.csv")
        catat("kolom dengan nilai hilang", len(hilang), "imputasi atau biarkan NaN untuk LightGBM")

    # duplikat
    n_dup_id = df.duplicated("id_transaksi").sum()
    n_dup_baris = df.duplicated().sum()
    print(f"\nid_transaksi ganda : {n_dup_id}")
    print(f"Baris identik      : {n_dup_baris}")
    catat("id_transaksi ganda", n_dup_id, "buang duplikat, pertahankan baris pertama")

    # nilai tidak wajar
    n_nominal = (df["nominal"] <= 0).sum()
    print(f"nominal <= 0       : {n_nominal}")
    catat("nominal <= 0", n_nominal, "periksa contoh; buang bila bukan transaksi sah")
    kuantil = df["nominal"].quantile([0.5, 0.9, 0.99, 0.999, 1.0])
    print("Kuantil nominal    :", {k: round(v) for k, v in kuantil.items()})

    n_jam = (df["waktu"].dt.hour != df["jam"]).sum()
    print(f"jam tidak cocok dengan waktu : {n_jam}")
    catat("kolom jam tidak cocok dengan jam di waktu", n_jam, "percayai waktu, hitung ulang jam")

    if "persen_saldo" in df:
        n_ps = ((df["persen_saldo"] < 0) | (df["persen_saldo"] > 1.0001)).sum()
        print(f"persen_saldo di luar [0, 1]  : {n_ps}")
        catat("persen_saldo di luar 0-1", n_ps, "periksa; mungkin saldo kecil atau pengisian ulang")

    for kol in KOLOM_BINER:
        if kol in df:
            n_bin = (~df[kol].dropna().isin([0, 1])).sum()
            if n_bin:
                print(f"{kol} bukan 0/1 : {n_bin}")
                catat(f"{kol} bukan 0/1", n_bin, "periksa dan seragamkan")

    # konsistensi penulisan kategori
    print("\nKonsistensi penulisan kategori (asli -> setelah strip + huruf kecil):")
    for kol in KOLOM_KATEGORI:
        if kol in df:
            asli = df[kol].nunique()
            bersih = df[kol].astype(str).str.strip().str.lower().nunique()
            print(f"  {kol:20s}: {asli} -> {bersih}")
            if asli != bersih:
                catat(f"penulisan {kol} tidak konsisten", asli - bersih,
                      "seragamkan dengan strip dan huruf kecil")

    pd.DataFrame(temuan).to_csv(HASIL_DIR / "t2_audit_ringkasan.csv", index=False)


def audit_tambahan(df):
    """Pemeriksaan lanjutan: pola nilai hilang dan persen_saldo di luar 0-1."""
    print("\n=== 1b. PEMERIKSAAN LANJUTAN ===")

    kol = [k for k in ["menit_sejak_trx_terakhir", "rasio_nominal_rata2"] if k in df]
    if kol:
        hilang = df[kol[0]].isna()
        sama = all((df[k].isna() == hilang).all() for k in kol)
        urut = df.sort_values("waktu")
        pertama = (~urut.duplicated("id_pengguna")).reindex(df.index)
        print(f"Kolom riwayat hilang di baris yang sama      : {sama}")
        print(f"Jumlah pengguna unik                         : {df['id_pengguna'].nunique():,}")
        print(f"Transaksi pertama pengguna dalam data        : {int(pertama.sum()):,}")
        print(f"Baris hilang                                 : {int(hilang.sum()):,}")
        print(f"Baris hilang yang merupakan transaksi pertama: {(hilang & pertama).sum() / max(hilang.sum(), 1):.2%}")
        tabel = df.groupby(hilang)[TARGET].agg(transaksi="size", tingkat="mean")
        tabel.index.name = "riwayat_hilang"
        print("Tingkat fraud menurut ada/tidaknya riwayat:")
        print(tabel.to_string(float_format=lambda x: f"{x:.4f}"))
        tabel.to_csv(HASIL_DIR / "t2_pola_nilai_hilang.csv")

    if "persen_saldo" in df and "kanal" in df:
        luar = (df["persen_saldo"] < 0) | (df["persen_saldo"] > 1.0001)
        tabel = (df.assign(di_luar=luar).groupby("kanal")["di_luar"]
                 .agg(transaksi="size", di_luar="sum", persen="mean"))
        print("\npersen_saldo di luar [0, 1] menurut kanal:")
        print(tabel.to_string(float_format=lambda x: f"{x:.4f}"))
        tabel.to_csv(HASIL_DIR / "t2_persen_saldo_per_kanal.csv")



# --------------------------------------------------------------------------
# 2. Keterlambatan label: tingkat fraud per minggu
# --------------------------------------------------------------------------
def fraud_per_minggu(df):
    """Menghitung dan menggambar tingkat fraud per minggu."""
    print("\n=== 2. TINGKAT FRAUD PER MINGGU ===")
    minggu = df["waktu"].dt.to_period("W").dt.start_time
    tabel = df.groupby(minggu)[TARGET].agg(transaksi="size", fraud="sum", tingkat="mean")
    tabel.index.name = "minggu_mulai"

    # Pembanding: tingkat fraud dasar = median minggu-minggu lama (di luar 6 minggu terakhir).
    # Minggu pertama dan terakhir bisa belum penuh, jadi tidak ikut menentukan dasar.
    tengah = tabel.iloc[1:-1]
    if len(tengah) > 10:
        dasar = tengah["tingkat"].iloc[:-6].median()
        tabel["rasio_terhadap_dasar"] = tabel["tingkat"] / dasar
    else:
        dasar = np.nan
        tabel["rasio_terhadap_dasar"] = np.nan
    tabel.to_csv(HASIL_DIR / "t2_fraud_per_minggu.csv")

    print(tabel.drop(columns="rasio_terhadap_dasar").tail(10).to_string(float_format=lambda x: f"{x:.4f}"))

    if not np.isnan(dasar):
        print(f"\nTingkat fraud dasar (median minggu lama): {dasar:.4%}")
        print("Rasio 6 minggu terakhir terhadap dasar (1,00 = setara minggu lama):")
        print(tabel["rasio_terhadap_dasar"].tail(6).round(2).to_string())
        belum = tabel.tail(6)[tabel.tail(6)["rasio_terhadap_dasar"] < 0.75]
        if len(belum):
            print("\n-> Minggu dengan rasio < 0,75 (label diduga belum matang):",
                  ", ".join(str(t.date()) for t in belum.index))
            print("   Periksa grafiknya; minggu terakhir bisa belum penuh bila data tidak "
                  "berakhir di hari Minggu.")
        else:
            print("\n-> Tidak ada minggu terakhir yang jauh di bawah dasar; tetap periksa grafiknya.")

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(tabel.index, tabel["transaksi"], width=5, color="#cfd8dc", label="Jumlah transaksi")
    ax1.set_ylabel("Jumlah transaksi")
    ax2 = ax1.twinx()
    ax2.plot(tabel.index, tabel["tingkat"] * 100, color="#c62828", marker="o", ms=3,
             label="Tingkat fraud (%)")
    ax2.set_ylabel("Tingkat fraud (%)")
    ax1.set_title("Tingkat fraud per minggu pada data latih")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(HASIL_DIR / "t2_fraud_per_minggu.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Uji kebocoran fitur
# --------------------------------------------------------------------------
def fitur_model(df):
    """Daftar fitur untuk model audit: numerik + kategori berkardinalitas rendah."""
    buang = set(KOLOM_ID + ["waktu", TARGET])
    return [k for k in df.columns if k not in buang]


def siapkan_matriks(df, fitur):
    x = df[fitur].copy()
    for kol in KOLOM_KATEGORI:
        if kol in x:
            x[kol] = x[kol].astype(str).str.strip().str.lower().astype("category")
    return x


def pisah_waktu(df, buang_akhir_hari=14, panjang_valid_hari=28, jeda_hari=7):
    """Pemisahan sementara untuk uji kebocoran.

    Minggu paling akhir dibuang (labelnya paling belum matang), lalu ada jeda antara
    data latih dan validasi. Nilai ini hanya awal; panjang jeda final ditentukan di PR 2
    berdasarkan grafik keterlambatan label.
    """
    akhir = df["waktu"].max() - pd.Timedelta(days=buang_akhir_hari)
    mulai_valid = akhir - pd.Timedelta(days=panjang_valid_hari)
    batas_latih = mulai_valid - pd.Timedelta(days=jeda_hari)
    latih = df[df["waktu"] < batas_latih]
    valid = df[(df["waktu"] >= mulai_valid) & (df["waktu"] < akhir)]
    print(f"Latih  : {len(latih):,} baris, fraud {latih[TARGET].mean():.4%}  (sebelum {batas_latih.date()})")
    print(f"Valid  : {len(valid):,} baris, fraud {valid[TARGET].mean():.4%}  ({mulai_valid.date()} s.d. {akhir.date()})")
    if latih.empty or valid.empty or valid[TARGET].sum() == 0:
        raise ValueError("Pemisahan waktu menghasilkan data kosong atau tanpa fraud; ubah parameter.")
    return latih, valid


def skor_model(latih, valid, fitur):
    """Melatih LightGBM dan mengembalikan PR-AUC pada data validasi."""
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(siapkan_matriks(latih, fitur), latih[TARGET])
    prob = model.predict_proba(siapkan_matriks(valid, fitur))[:, 1]
    return average_precision_score(valid[TARGET], prob)


def uji_kebocoran(df):
    print("\n=== 3. UJI KEBOCORAN FITUR ===")

    # (a) Hubungan langsung kolom curiga dengan label
    if "flag_review_manual" in df:
        print("\nTingkat fraud menurut flag_review_manual:")
        tabel = df.groupby("flag_review_manual")[TARGET].agg(transaksi="size", tingkat="mean")
        print(tabel.to_string(float_format=lambda x: f"{x:.4f}"))
        tabel.to_csv(HASIL_DIR / "t2_flag_review_vs_fraud.csv")

    # (b) Saring tiap kolom numerik: AUC satu fitur terhadap label.
    # Nilai yang terlalu tinggi layak dicurigai.
    print("\nKolom numerik dengan AUC tunggal tertinggi (0,5 = tidak informatif):")
    baris = []
    for kol in df.select_dtypes("number").columns:
        if kol in KOLOM_ID or kol == TARGET:
            continue
        sub = df[[kol, TARGET]].dropna()
        if sub[kol].nunique() < 2:
            continue
        auc = roc_auc_score(sub[TARGET], sub[kol])
        baris.append({"kolom": kol, "auc_tunggal": max(auc, 1 - auc)})
    saring = pd.DataFrame(baris).sort_values("auc_tunggal", ascending=False)
    saring.to_csv(HASIL_DIR / "t2_auc_tunggal.csv", index=False)
    print(saring.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # (c) Dampak membuang tiap kolom curiga pada PR-AUC validasi berbasis waktu
    print("\nDampak membuang kolom curiga (PR-AUC validasi berbasis waktu):")
    latih, valid = pisah_waktu(df)
    semua = fitur_model(df)
    hasil = [{"fitur_dibuang": "(tidak ada)", "pr_auc": skor_model(latih, valid, semua)}]
    for kol in KANDIDAT_BOCOR:
        if kol in semua:
            hasil.append({"fitur_dibuang": kol,
                          "pr_auc": skor_model(latih, valid, [k for k in semua if k != kol])})
    ada = [k for k in KANDIDAT_BOCOR if k in semua]
    if len(ada) > 1:
        hasil.append({"fitur_dibuang": " + ".join(ada),
                      "pr_auc": skor_model(latih, valid, [k for k in semua if k not in ada])})
    tabel = pd.DataFrame(hasil)
    tabel["pr_auc_acak"] = valid[TARGET].mean()  # PR-AUC model tanpa kemampuan = tingkat fraud
    tabel.to_csv(HASIL_DIR / "t2_dampak_kebocoran.csv", index=False)
    print(tabel.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBaca hasilnya: bila membuang satu kolom menurunkan PR-AUC sangat tajam, "
          "kolom itu patut dicurigai sebagai fitur bocor. Periksa juga maknanya di kamus data.")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Audit data Task 2 (klasifikasi fraud)")
    parser.add_argument("--berkas", default="klasifikasi_train.csv.gz",
                        help="nama berkas di folder data/")
    args = parser.parse_args()

    HASIL_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    df = muat_data(args.berkas)
    audit_kualitas(df)
    audit_tambahan(df)
    fraud_per_minggu(df)
    uji_kebocoran(df)
    print(f"\nTabel dan gambar tersimpan di: {HASIL_DIR}")


if __name__ == "__main__":
    main()
