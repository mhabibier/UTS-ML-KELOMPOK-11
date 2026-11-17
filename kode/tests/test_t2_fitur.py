"""Unit test anti-kebocoran untuk fitur buatan sendiri (kode/t2_fitur.py).

Data tiruan dipakai HANYA untuk menguji logika (masa lalu saja, hitungan benar),
bukan untuk menghasilkan angka laporan.

Jalankan dari akar repo:  python -m pytest kode/tests -q
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from t2_data import TARGET
from t2_fitur import (KELOMPOK_FITUR, KELOMPOK_TERPILIH, fitur_fraud_historis,
                      fitur_label, fitur_riwayat_pengguna, fitur_statis,
                      gabung_latih_uji, susun_fitur)

BATAS = pd.Timestamp("2025-08-11")
# Aturan yang sudah diputuskan (label matang 3 minggu). Sengaja ditulis ulang di sini, tidak
# di-import dari t2_fitur, supaya tes gagal bila jeda di kode berubah tanpa disengaja.
JEDA_LABEL_HARI = 21


@pytest.fixture
def df_tiruan():
    """3.000 transaksi acak 1 Jan - 31 Okt 2025; label kosong mulai 1 Sep (meniru data uji)."""
    rng = np.random.default_rng(1)
    n = 3000
    waktu = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 303, n)), unit="D")
    df = pd.DataFrame({
        "id_transaksi": [f"t{i:05d}" for i in range(n)],
        "waktu": waktu.floor("min"),               # pembulatan -> ada transaksi dengan waktu sama
        "id_pengguna": rng.choice([f"u{i}" for i in range(60)], n),
        "id_merchant": rng.choice([f"m{i}" for i in range(25)], n),
        "nominal": rng.lognormal(12, 1, n).round(),
        "persen_saldo": rng.uniform(0, 1.5, n),
        "umur_perangkat_hari": rng.choice([0.0, 0.5, 10.0, 300.0], n),
        "perangkat_baru": rng.integers(0, 2, n),
        "menit_sejak_trx_terakhir": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 1e4, n)),
        TARGET: (rng.random(n) < 0.1).astype(float),
    })
    df.loc[df["waktu"] >= "2025-09-01", TARGET] = np.nan
    return df.sample(frac=1, random_state=0)       # urutan baris diacak: fungsi harus mengurutkan sendiri


def _ubah_masa_depan(df, t0, rng):
    """Mengacak dan menghapus sebagian transaksi SESUDAH t0; baris sampai t0 tidak disentuh."""
    df = df.copy()
    depan = df.index[df["waktu"] > t0]
    for kol in ["id_pengguna", "id_merchant", "nominal"]:
        df.loc[depan, kol] = rng.permutation(df.loc[depan, kol].to_numpy())
    df.loc[depan, TARGET] = 1.0
    return df.drop(index=depan[::3])


def test_fitur_statis_tidak_melihat_masa_depan(df_tiruan):
    t0 = pd.Timestamp("2025-06-01")
    asli = fitur_statis(df_tiruan)
    ubah = fitur_statis(_ubah_masa_depan(df_tiruan, t0, np.random.default_rng(2)))
    lalu = df_tiruan.index[df_tiruan["waktu"] <= t0]
    assert_frame_equal(asli.loc[lalu], ubah.loc[lalu])


def test_fitur_latih_sama_dengan_dihitung_bersama_data_uji(df_tiruan):
    """Menambahkan data uji (masa depan) tidak boleh mengubah fitur baris latih."""
    latih = df_tiruan[df_tiruan["waktu"] < "2025-09-01"]
    uji = df_tiruan[df_tiruan["waktu"] >= "2025-09-01"]
    gab = gabung_latih_uji(latih.reset_index(drop=True), uji.reset_index(drop=True))
    f_gab = fitur_statis(gab)
    f_latih = fitur_statis(latih.reset_index(drop=True))
    assert_frame_equal(f_gab.iloc[:len(latih)].reset_index(drop=True), f_latih.reset_index(drop=True))
    assert (gab["sumber"] == "uji").sum() == len(uji)


def test_urutan_baris_tidak_berpengaruh(df_tiruan):
    asli = fitur_statis(df_tiruan)
    acak = fitur_statis(df_tiruan.sample(frac=1, random_state=5))
    assert_frame_equal(asli.sort_index(), acak.sort_index())


def test_riwayat_pengguna_sesuai_hitungan_manual(df_tiruan):
    fitur = fitur_riwayat_pengguna(df_tiruan)
    urut = df_tiruan.sort_values(["waktu", "id_transaksi"])
    for idx in urut.sample(40, random_state=3).index:
        baris = urut.loc[idx]
        posisi = urut.index.get_loc(idx)
        lalu = urut.iloc[:posisi]
        lalu = lalu[lalu["id_pengguna"] == baris["id_pengguna"]]
        assert fitur.loc[idx, "jumlah_trx_sebelum"] == len(lalu)
        assert fitur.loc[idx, "pertama_di_merchant"] == int(baris["id_merchant"] not in set(lalu["id_merchant"]))
        assert fitur.loc[idx, "jumlah_merchant_sebelum"] == lalu["id_merchant"].nunique()
        if len(lalu) >= 2:
            assert fitur.loc[idx, "sb_log_nominal_sebelum"] == pytest.approx(
                np.log1p(lalu["nominal"]).std(ddof=1), abs=1e-9)
        else:
            assert np.isnan(fitur.loc[idx, "sb_log_nominal_sebelum"])


@pytest.mark.parametrize("kunci", ["id_merchant", "id_pengguna"])
def test_fraud_historis_sesuai_hitungan_manual(df_tiruan, kunci):
    """n dan jumlah fraud = label transaksi kunci yang sama dengan waktu < t - 21 hari dan < batas."""
    fitur = fitur_fraud_historis(df_tiruan, kunci, BATAS)
    awalan = "merchant" if kunci == "id_merchant" else "pengguna"
    jeda = pd.Timedelta(days=JEDA_LABEL_HARI)
    for idx in df_tiruan.sample(60, random_state=4).index:
        baris = df_tiruan.loc[idx]
        tahu = df_tiruan[(df_tiruan[kunci] == baris[kunci])
                         & (df_tiruan["waktu"] < baris["waktu"] - jeda)
                         & (df_tiruan["waktu"] < BATAS)]
        assert fitur.loc[idx, f"n_label_{awalan}"] == len(tahu)
        assert fitur.loc[idx, f"fraud_label_{awalan}"] == tahu[TARGET].sum()


def test_fraud_historis_tidak_melihat_label_yang_belum_diketahui(df_tiruan):
    """Mengubah label dalam 21 hari terakhir sebelum t0, atau label sesudah batas, tidak mengubah fitur."""
    t0 = pd.Timestamp("2025-06-01")
    asli = fitur_label(df_tiruan, BATAS)
    ubah = df_tiruan.copy()
    belum_tahu = (ubah["waktu"] >= t0 - pd.Timedelta(days=JEDA_LABEL_HARI)) | (ubah["waktu"] >= BATAS)
    ubah.loc[belum_tahu & ubah[TARGET].notna(), TARGET] = 1 - ubah.loc[belum_tahu & ubah[TARGET].notna(), TARGET]
    hasil = fitur_label(ubah, BATAS)
    lalu = df_tiruan.index[df_tiruan["waktu"] <= t0]
    assert_frame_equal(asli.loc[lalu], hasil.loc[lalu])


def test_fraud_historis_baris_sesudah_batas_hanya_melihat_label_sebelum_batas(df_tiruan):
    """Transaksi uji (sesudah batas) memakai label sampai batas saja, berapa pun jaraknya."""
    fitur = fitur_fraud_historis(df_tiruan, "id_merchant", BATAS)
    uji = df_tiruan[df_tiruan["waktu"] >= "2025-09-15"]
    for idx in uji.sample(20, random_state=6).index:
        m = df_tiruan.loc[idx, "id_merchant"]
        tahu = df_tiruan[(df_tiruan["id_merchant"] == m) & (df_tiruan["waktu"] < BATAS)]
        assert fitur.loc[idx, "n_label_merchant"] == len(tahu)


def test_fraud_historis_batas_jeda_tepat():
    """Label yang terjadi TEPAT 21 hari sebelumnya belum dihitung; 21 hari + 1 menit sudah."""
    t = pd.Timestamp("2025-05-01 12:00")
    df = pd.DataFrame({
        "id_transaksi": ["a", "b", "c"],
        "waktu": [t - pd.Timedelta(days=21, minutes=1), t - pd.Timedelta(days=21), t],
        "id_merchant": ["m1", "m1", "m1"],
        TARGET: [1.0, 1.0, 0.0],
    })
    f = fitur_fraud_historis(df, "id_merchant", BATAS)
    assert f.loc[2, "n_label_merchant"] == 1      # hanya transaksi "a"
    assert f.loc[2, "fraud_label_merchant"] == 1
    assert f.loc[0, "n_label_merchant"] == 0      # transaksi pertama tidak punya label masa lalu


def test_susunan_terpilih():
    """Susunan final: 14 kolom baru (satu menggantikan umur_perangkat_hari), tanpa merchant_terlihat."""
    dasar = ["nominal", "umur_perangkat_hari", "trx_1j"]
    fitur = susun_fitur(dasar, KELOMPOK_TERPILIH)
    assert "merchant_terlihat" not in KELOMPOK_TERPILIH
    assert "umur_perangkat_hari" not in fitur and "umur_perangkat_hari_bersih" in fitur
    assert len(fitur) == len(dasar) - 1 + sum(len(KELOMPOK_FITUR[g][0]) for g in KELOMPOK_TERPILIH)
    assert len(fitur) == len(set(fitur))
    dasar_78 = [f"k{i}" for i in range(77)] + ["umur_perangkat_hari"]
    assert len(susun_fitur(dasar_78, KELOMPOK_TERPILIH)) == 78 - 1 + 14 == 91


def test_penanda(df_tiruan):
    f = fitur_statis(df_tiruan)
    assert (f["melebihi_saldo"] == (df_tiruan["persen_saldo"] > 1)).all()
    assert (f["tanpa_riwayat"] == df_tiruan["menit_sejak_trx_terakhir"].isna()).all()
    tidak_tercatat = (df_tiruan["umur_perangkat_hari"] == 0) & (df_tiruan["perangkat_baru"] == 0)
    assert f.loc[tidak_tercatat, "umur_perangkat_hari_bersih"].isna().all()
    assert (f.loc[~tidak_tercatat, "umur_perangkat_hari_bersih"] == df_tiruan.loc[~tidak_tercatat, "umur_perangkat_hari"]).all()
