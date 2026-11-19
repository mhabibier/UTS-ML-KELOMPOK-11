"""Cek format tiga berkas prediksi sebelum push ke main (sesuai panduan UTS).

  prediksi_regresi.csv     : kolom tepat id_paket,p50,p90; tanpa NaN; semua p90 >= p50.
  prediksi_klasifikasi.csv : kolom tepat id_transaksi,prob,tandai; 0 <= prob <= 1; tandai hanya 0 atau 1.
  cluster.csv              : kolom tepat id_pelanggan,cluster; cluster integer (-1 = anomali).
Untuk ketiganya: semua ID sama persis dengan ID di data uji (tidak ada yang hilang, berlebih, atau ganda).

Cara menjalankan dari akar repo:
    python kode/cek_format.py
Keluar dengan kode 1 bila ada berkas yang gagal; berkas yang belum ada hanya dilaporkan.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def cek_id(id_berkas, id_uji):
    """Masalah ID: hilang, berlebih, atau ganda dibanding ID data uji (dibandingkan sebagai teks)."""
    a = pd.Series(id_berkas).astype(str)
    b = pd.Series(id_uji).astype(str)
    masalah = []
    if a.isna().any() or (a == "nan").any():
        masalah.append("ada ID kosong")
    if a.duplicated().any():
        masalah.append(f"{int(a.duplicated().sum())} ID ganda")
    hilang = set(b) - set(a)
    lebih = set(a) - set(b)
    if hilang:
        masalah.append(f"{len(hilang)} ID data uji tidak ada di berkas (contoh: {sorted(hilang)[:3]})")
    if lebih:
        masalah.append(f"{len(lebih)} ID tidak ada di data uji (contoh: {sorted(lebih)[:3]})")
    if len(a) != len(b):
        masalah.append(f"jumlah baris {len(a):,} != jumlah data uji {len(b):,}")
    return masalah


def _cek_kolom(df, kolom):
    if list(df.columns) != kolom:
        return [f"kolom harus tepat {','.join(kolom)}, ditemukan {','.join(map(str, df.columns))}"]
    return []


def _numerik(s):
    return pd.to_numeric(s, errors="coerce")


def cek_regresi(df, id_uji):
    masalah = _cek_kolom(df, ["id_paket", "p50", "p90"])
    if masalah:
        return masalah
    p50, p90 = _numerik(df["p50"]), _numerik(df["p90"])
    if p50.isna().any() or p90.isna().any():
        masalah.append(f"ada NaN/bukan angka: p50 {int(p50.isna().sum())}, p90 {int(p90.isna().sum())}")
    if (p90 < p50).any():
        masalah.append(f"{int((p90 < p50).sum())} baris dengan p90 < p50")
    return masalah + cek_id(df["id_paket"], id_uji)


def cek_klasifikasi(df, id_uji):
    masalah = _cek_kolom(df, ["id_transaksi", "prob", "tandai"])
    if masalah:
        return masalah
    prob, tandai = _numerik(df["prob"]), _numerik(df["tandai"])
    if prob.isna().any():
        masalah.append(f"{int(prob.isna().sum())} prob kosong/bukan angka")
    if ((prob < 0) | (prob > 1)).any():
        masalah.append(f"{int(((prob < 0) | (prob > 1)).sum())} prob di luar [0, 1]")
    if not tandai.isin([0, 1]).all():
        masalah.append(f"{int((~tandai.isin([0, 1])).sum())} nilai tandai selain 0/1")
    return masalah + cek_id(df["id_transaksi"], id_uji)


def cek_cluster(df, id_uji):
    masalah = _cek_kolom(df, ["id_pelanggan", "cluster"])
    if masalah:
        return masalah
    c = _numerik(df["cluster"])
    if c.isna().any() or not np.all(np.mod(c.dropna(), 1) == 0):
        masalah.append("cluster harus bilangan bulat tanpa nilai kosong")
    elif (c < -1).any():
        masalah.append("cluster negatif selain -1 (anomali)")
    return masalah + cek_id(df["id_pelanggan"], id_uji)


SPEK = [
    ("prediksi_regresi.csv", cek_regresi, "regresi_test.csv.gz", "id_paket"),
    ("prediksi_klasifikasi.csv", cek_klasifikasi, "klasifikasi_test.csv.gz", "id_transaksi"),
    ("cluster.csv", cek_cluster, "clustering.csv.gz", "id_pelanggan"),
]


def main():
    gagal = False
    for berkas, fungsi, data_uji, kolom_id in SPEK:
        jalur = ROOT / berkas
        if not jalur.exists():
            print(f"[BELUM ADA] {berkas}")
            continue
        df = pd.read_csv(jalur, dtype={kolom_id: str})
        id_uji = pd.read_csv(DATA_DIR / data_uji, usecols=[kolom_id], dtype={kolom_id: str})[kolom_id]
        masalah = fungsi(df, id_uji)
        if masalah:
            gagal = True
            print(f"[GAGAL] {berkas}:")
            for m in masalah:
                print(f"    - {m}")
        else:
            print(f"[LOLOS] {berkas} ({len(df):,} baris)")
    sys.exit(1 if gagal else 0)


if __name__ == "__main__":
    main()
