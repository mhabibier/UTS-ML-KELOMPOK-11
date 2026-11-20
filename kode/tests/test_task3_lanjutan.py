"""Unit test fungsi bantu Task 3 bagian e-g (kode/task3_lanjutan.py). Data tiruan kecil, bukan angka laporan."""
import numpy as np
import pandas as pd
import pytest

from task3_lanjutan import (NAMA_AKSI, ari_antar_run, ari_matriks, jackknife_selisih, periksa_nama,
                            pilih_k, ringkas_stabilitas, tandai_anomali, urutkan_label)


def test_run_gagal_tidak_dihitung_sebagai_partisi():
    """Run None atau satu cluster = gagal; dua run gagal tidak boleh menghasilkan ARI 1 (seolah stabil)."""
    gagal = ringkas_stabilitas("x", [None, np.zeros(6, int), np.zeros(6, int)])
    assert gagal["n_run_valid"] == 0 and gagal["n_run_gagal"] == 3
    assert np.isnan(gagal["ari_rata2"]) and not gagal["stabil"]
    a = np.array([0, 0, 1, 1, 2, 2])
    campur = ringkas_stabilitas("y", [a, a[::-1].copy(), None, np.zeros(6, int)])
    assert campur["n_run_valid"] == 2 and campur["n_run_gagal"] == 2
    assert campur["ari_rata2"] == pytest.approx(1.0)


def test_jackknife_selisih():
    """Selisih rata-rata ARI dan galat baku jackknife berpasangan pada matriks yang diketahui."""
    rng = np.random.default_rng(0)
    runs = [rng.integers(0, 3, 300) for _ in range(6)]
    m = ari_matriks(runs)
    assert np.allclose(m, m.T, equal_nan=True) and np.isnan(np.diag(m)).all()
    d, se = jackknife_selisih(m, m)
    assert d == pytest.approx(0) and se == pytest.approx(0)            # representasi identik
    d, se = jackknife_selisih(m + 0.1, m)
    assert d == pytest.approx(0.1) and se == pytest.approx(0, abs=1e-12)  # selisih konstan: tanpa ketidakpastian
    acak = m + rng.normal(0, 0.05, m.shape)
    acak = (acak + acak.T) / 2
    d, se = jackknife_selisih(acak, m)
    assert se > 0


def test_ari_tidak_peduli_nomor_cluster_tertukar():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])          # keanggotaan sama, nomor berbeda
    c = np.array([0, 1, 0, 1, 0, 1])          # keanggotaan berbeda
    ari = ari_antar_run([a, b, c])
    assert len(ari) == 3                       # pasangan (a,b), (a,c), (b,c)
    assert ari[0] == pytest.approx(1.0)
    assert ari[1] < 0.5 and ari[2] < 0.5


def test_urutkan_label_menurut_median():
    label = np.array([5, 5, 7, 7, 9, 9])
    nilai = np.array([10, 12, 100, 90, 50, 55])
    assert urutkan_label(label, nilai).tolist() == [3, 3, 1, 1, 2, 2]   # 1 = median terbesar


def test_tandai_anomali_sekitar_dua_persen():
    jarak = np.random.default_rng(0).gamma(2.0, 1.0, 10_000)
    tanda = tandai_anomali(jarak, 98)
    assert tanda.mean() == pytest.approx(0.02, abs=0.001)
    assert jarak[tanda].min() > jarak[~tanda].max()                      # yang ditandai = yang terjauh


def test_pilih_k_suara_terbanyak_di_antara_k_stabil():
    tabel = pd.DataFrame({
        "k": [2, 3, 4, 5],
        "silhouette": [0.10, 0.20, 0.15, 0.30],          # k=5 terbaik tetapi tidak stabil
        "calinski_harabasz": [500, 900, 800, 1000],
        "davies_bouldin": [2.5, 2.0, 2.1, 1.0],
        "ari_rata2": [0.99, 0.95, 0.90, 0.60],
    })
    assert pilih_k(tabel) == 3                          # k=5 disingkirkan (ARI < 0,8); k=3 menang 3 suara
    tidak_stabil = tabel.assign(ari_rata2=[0.5, 0.6, 0.7, 0.4])
    assert pilih_k(tidak_stabil) == 4                   # semua tidak stabil -> yang paling stabil


def test_periksa_nama_mendeteksi_nama_tidak_sesuai():
    profil = pd.DataFrame({"cluster": [0, 1, 2], "median_frekuensi_90h": [0, 12, 5],
                           "median_belanja_90h": [0, 4e6, 6e5], "median_persen_voucher": [0, 0.3, 0.66],
                           "porsi_jam_malam": [0, 0.18, 0.30]})
    assert periksa_nama(profil) == []
    tertukar = profil.assign(median_belanja_90h=[0, 6e5, 4e6])
    assert periksa_nama(tertukar)
    assert periksa_nama(profil.assign(cluster=[0, 1, 7]))                # segmen tanpa nama
    assert {0, 1, 2, -1} <= set(NAMA_AKSI)
