"""Task 2 - fungsi bersama: konstanta, muat data, daftar fitur, dan pembagian waktu.

Modul ini dipakai oleh kode/task2_klasifikasi.py dan diuji di kode/tests/test_t2_data.py.
Bila anggota A membuat kode/common.py, konstanta umum (SEED, ROOT, DATA_DIR) bisa dipindah ke sana.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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

# Fitur bocor, dibuang dari semua model:
# - flag_review_manual (PR 1): tinjauan manual terjadi SETELAH transaksi.
# - umur_merchant_hari (PR 2): negatif di 955 transaksi (tingkat fraud 20%) karena tanggal daftar
#   merchant tercatat setelah transaksi; sumbangannya ke PR-AUC hampir seluruhnya dari baris itu.
FITUR_BOCOR = ["flag_review_manual", "umur_merchant_hari"]

# Belum terbukti bocor; diuji ulang dengan beberapa seed di tahap validasi.
# volume_merchant_30h: nilainya konstan per merchant (satu potret, bukan hitungan 30 hari terakhir).
KOLOM_DIPANTAU = ["jumlah_akun_per_perangkat", "refund_merchant_30h", "volume_merchant_30h"]

# Label dari tanggal ini ke atas belum matang (lihat t2_fraud_per_minggu.png):
# transaksi mulai 11 Agustus 2025 tidak dipakai untuk latih maupun validasi.
BATAS_LABEL_MATANG = pd.Timestamp("2025-08-11")


# --------------------------------------------------------------------------
# Muat dan siapkan data
# --------------------------------------------------------------------------
def muat_data(nama_berkas):
    """Membaca berkas di data/ dan mengubah kolom waktu menjadi tipe tanggal (tanpa pembersihan lain)."""
    jalur = DATA_DIR / nama_berkas
    if not jalur.exists():
        raise FileNotFoundError(f"Berkas tidak ditemukan: {jalur}")
    df = pd.read_csv(jalur)
    df["waktu"] = pd.to_datetime(df["waktu"], errors="coerce")
    return df


def bersihkan_kategori(df):
    """Menyeragamkan penulisan kolom kategori (strip + huruf kecil) dan mengubahnya ke tipe category.

    Dilakukan sekali pada seluruh data supaya data latih dan validasi memakai daftar kategori yang sama.
    """
    df = df.copy()
    for kol in KOLOM_KATEGORI:
        if kol in df:
            df[kol] = df[kol].astype(str).str.strip().str.lower().astype("category")
    return df


def data_matang(df, batas=BATAS_LABEL_MATANG):
    """Hanya transaksi sebelum batas label matang."""
    return df[df["waktu"] < batas]


def fitur_model(df, buang=()):
    """Daftar kolom fitur: semua kolom kecuali ID, waktu, label, fitur bocor, dan kolom di `buang`."""
    dikecualikan = set(KOLOM_ID) | {"waktu", TARGET} | set(FITUR_BOCOR) | set(buang)
    return [k for k in df.columns if k not in dikecualikan]


def tambah_kolom(df, fitur, tambahan):
    """Daftar fitur + kolom `tambahan`, dengan urutan kolom tetap mengikuti urutan di df.

    Dipakai untuk uji bukti kebocoran (sengaja memasukkan kembali kolom bocor). Urutan dijaga
    karena urutan kolom ikut menentukan hasil acak LightGBM (colsample_bytree).
    """
    pilih = set(fitur) | set(tambahan)
    return [k for k in df.columns if k in pilih]


# --------------------------------------------------------------------------
# Pembagian berbasis waktu
# --------------------------------------------------------------------------
@dataclass
class Lipatan:
    """Satu lipatan validasi: urutan waktunya latih | (kalibrasi) | jeda | validasi."""
    nomor: int
    batas_latih: pd.Timestamp       # latih: waktu < batas_latih
    mulai_kalibrasi: pd.Timestamp   # kalibrasi: mulai_kalibrasi <= waktu < akhir_kalibrasi
    akhir_kalibrasi: pd.Timestamp   # sama dengan awal jeda
    mulai_valid: pd.Timestamp       # validasi: mulai_valid <= waktu < akhir_valid
    akhir_valid: pd.Timestamp
    idx_latih: np.ndarray
    idx_kalibrasi: np.ndarray
    idx_valid: np.ndarray


def buat_lipatan_waktu(df, n_lipatan=4, panjang_valid_hari=28, jeda_hari=21,
                       panjang_kalibrasi_hari=0, batas_akhir=BATAS_LABEL_MATANG):
    """Membuat lipatan validasi berbasis waktu dengan jendela latih yang makin panjang.

    Lipatan terakhir berakhir tepat di `batas_akhir`; lipatan sebelumnya mundur
    `panjang_valid_hari` setiap langkah, jadi jendela validasi bersambung dan tidak tumpang tindih.

    Jeda meniru keadaan saat model dipakai: model final dilatih dengan label matang sampai
    10 Agustus lalu dipakai mulai 1 September, jadi ada jarak tiga minggu tanpa label.

    `panjang_kalibrasi_hari` > 0 menyisihkan periode kalibrasi tepat sebelum jeda (dipakai di PR 5);
    periode ini dipotong dari ujung data latih sehingga tidak pernah dipakai untuk melatih model.
    """
    if n_lipatan < 1 or panjang_valid_hari < 1 or jeda_hari < 0 or panjang_kalibrasi_hari < 0:
        raise ValueError("Parameter lipatan tidak valid.")

    waktu = df["waktu"]
    panjang = pd.Timedelta(days=panjang_valid_hari)
    jeda = pd.Timedelta(days=jeda_hari)
    kalib = pd.Timedelta(days=panjang_kalibrasi_hari)

    hasil = []
    for i in range(n_lipatan):
        akhir_valid = batas_akhir - (n_lipatan - 1 - i) * panjang
        mulai_valid = akhir_valid - panjang
        akhir_kalibrasi = mulai_valid - jeda
        mulai_kalibrasi = akhir_kalibrasi - kalib
        batas_latih = mulai_kalibrasi

        latih = waktu < batas_latih
        kalibrasi = (waktu >= mulai_kalibrasi) & (waktu < akhir_kalibrasi)
        valid = (waktu >= mulai_valid) & (waktu < akhir_valid)
        if not latih.any() or not valid.any():
            raise ValueError(f"Lipatan {i + 1} kosong; kurangi n_lipatan atau panjang periode.")

        hasil.append(Lipatan(
            nomor=i + 1,
            batas_latih=batas_latih,
            mulai_kalibrasi=mulai_kalibrasi,
            akhir_kalibrasi=akhir_kalibrasi,
            mulai_valid=mulai_valid,
            akhir_valid=akhir_valid,
            idx_latih=df.index[latih.to_numpy()].to_numpy(),
            idx_kalibrasi=df.index[kalibrasi.to_numpy()].to_numpy(),
            idx_valid=df.index[valid.to_numpy()].to_numpy(),
        ))
    return hasil


def ringkas_lipatan(df, lipatan):
    """Tabel ringkas tiap lipatan: periode, jumlah baris, jumlah fraud, dan tingkat fraud."""
    baris = []
    for lp in lipatan:
        latih = df.loc[lp.idx_latih, TARGET]
        valid = df.loc[lp.idx_valid, TARGET]
        baris.append({
            "lipatan": lp.nomor,
            "latih_mulai": df.loc[lp.idx_latih, "waktu"].min().date(),
            "latih_sampai_sebelum": lp.batas_latih.date(),
            "jeda_mulai": lp.akhir_kalibrasi.date(),
            "valid_mulai": lp.mulai_valid.date(),
            "valid_sampai_sebelum": lp.akhir_valid.date(),
            "latih_baris": len(latih),
            "latih_fraud": int(latih.sum()),
            "latih_tingkat": latih.mean(),
            "kalibrasi_baris": len(lp.idx_kalibrasi),
            "valid_baris": len(valid),
            "valid_fraud": int(valid.sum()),
            "valid_tingkat": valid.mean(),
        })
    return pd.DataFrame(baris)
