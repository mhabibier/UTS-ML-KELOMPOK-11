"""Unit test fungsi bantu pelatihan PR 4 (kode/t2_latih.py).

Data tiruan dipakai HANYA untuk menguji logika, bukan untuk menghasilkan angka laporan.
Jalankan dari akar repo:  python -m pytest kode/tests -q
"""
import numpy as np
import pandas as pd
import pytest

from t2_latih import (bobot_nominal, bobot_waktu, biaya, koreksi_sampel_kurang, metrik_lengkap,
                      rasio_sah_fraud, sampel_kurang, tandai_aturan)


def test_sampel_kurang_menyimpan_semua_fraud():
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.01).astype(int)
    pos, beta = sampel_kurang(y, 20, np.random.default_rng(1))
    n_fraud = y.sum()
    assert y[pos].sum() == n_fraud                          # semua fraud ikut
    assert (y[pos] == 0).sum() == 20 * n_fraud             # 20 sah per fraud
    assert beta == pytest.approx(20 * n_fraud / (len(y) - n_fraud))
    assert len(np.unique(pos)) == len(pos)
    pos2, _ = sampel_kurang(y, 20, np.random.default_rng(1))
    assert (pos == pos2).all()                              # dapat diulang dengan seed sama


def test_koreksi_sampel_kurang_mengembalikan_probabilitas_asli():
    """Bila p_s = probabilitas pada data yang sah-nya disisakan porsi beta, koreksi harus memberi p asli."""
    p = np.array([0.0005, 0.006, 0.05, 0.3, 0.9])
    beta = 0.14
    p_s = p / (p + beta * (1 - p))        # hitungan langsung: transaksi sah tinggal beta bagian
    assert koreksi_sampel_kurang(p_s, beta) == pytest.approx(p)
    assert (np.diff(koreksi_sampel_kurang(p_s, beta)) > 0).all()   # urutan tidak berubah


def test_koreksi_sampel_kurang_secara_empiris():
    """Tingkat fraud pada data undersampling, setelah dikoreksi, sama dengan tingkat fraud asli."""
    rng = np.random.default_rng(3)
    y = (rng.random(200_000) < 0.007).astype(int)
    pos, beta = sampel_kurang(y, 20, rng)
    assert koreksi_sampel_kurang(y[pos].mean(), beta) == pytest.approx(y.mean(), rel=1e-9)


def test_rasio_sah_fraud():
    assert rasio_sah_fraud([0, 0, 0, 1]) == 3


def test_bobot_nominal_sesuai_aturan_keputusan():
    """Skor berbobot q = w1 p / (w1 p + (1 - p)); q > 0,5 harus setara dengan p x nominal > 25.000."""
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 0.3, 5000)
    nominal = rng.lognormal(12, 1.2, 5000)
    w1 = bobot_nominal(np.ones(5000), nominal)
    q = w1 * p / (w1 * p + (1 - p))
    assert ((q > 0.5) == tandai_aturan(p, nominal)).all()
    assert (bobot_nominal(np.zeros(3), [1e6, 10, 5e4]) == 1).all()   # transaksi sah selalu 1
    assert bobot_nominal([1], [20_000])[0] == 0                       # fraud kecil tidak berharga ditandai


def test_bobot_waktu():
    acuan = pd.Timestamp("2025-06-23")
    waktu = pd.to_datetime(["2025-06-23", "2025-05-24", "2025-04-24"]).to_numpy()
    assert bobot_waktu(waktu, acuan.to_datetime64(), 30) == pytest.approx([1.0, 0.5, 0.25])


def test_biaya_dan_metrik():
    y = np.array([1, 1, 0, 0])
    nominal = np.array([1_000_000, 50_000, 200_000, 10_000])
    tandai = np.array([True, False, True, False])
    assert biaya(y, nominal, tandai) == 50_000 + 2 * 25_000          # fraud kedua lolos, 2 ditandai
    prob = np.array([0.5, 0.2, 0.3, 0.001])
    m = metrik_lengkap(y, prob, nominal)                              # aturan: 0,5 x 1 juta dan 0,3 x 200 ribu
    assert m["n_tandai"] == 2 and m["biaya_juta"] == pytest.approx(0.1)
    assert m["rasio_rata2"] == pytest.approx(prob.mean() / 0.5)
