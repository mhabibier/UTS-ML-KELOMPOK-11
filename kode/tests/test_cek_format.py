"""Unit test skrip cek format tiga berkas prediksi (kode/cek_format.py). Data tiruan kecil."""
import pandas as pd

from cek_format import cek_cluster, cek_id, cek_klasifikasi, cek_regresi, periksa_semua

ID = ["a1", "a2", "a3"]


def test_klasifikasi_lolos_dan_gagal():
    baik = pd.DataFrame({"id_transaksi": ID, "prob": [0.0, 0.5, 1.0], "tandai": [0, 1, 1]})
    assert cek_klasifikasi(baik, ID) == []
    assert cek_klasifikasi(baik.assign(prob=[0.1, 1.2, -0.1]), ID)          # di luar [0, 1]
    assert cek_klasifikasi(baik.assign(tandai=[0, 2, 1]), ID)               # tandai bukan 0/1
    assert cek_klasifikasi(baik[["id_transaksi", "tandai", "prob"]], ID)    # urutan kolom salah
    assert cek_klasifikasi(baik.assign(prob=[0.1, None, 0.2]), ID)          # NaN


def test_regresi_p90_minimal_p50():
    baik = pd.DataFrame({"id_paket": ID, "p50": [1.0, 2.0, 3.0], "p90": [1.0, 4.0, 5.0]})
    assert cek_regresi(baik, ID) == []
    assert cek_regresi(baik.assign(p90=[0.5, 4.0, 5.0]), ID)
    assert cek_regresi(baik.assign(p50=[1.0, None, 3.0]), ID)


def test_cluster_integer_dan_anomali():
    baik = pd.DataFrame({"id_pelanggan": ID, "cluster": [0, 2, -1]})
    assert cek_cluster(baik, ID) == []
    assert cek_cluster(baik.assign(cluster=[0, 1.5, 2]), ID)
    assert cek_cluster(baik.assign(cluster=[0, -2, 1]), ID)


def test_id_hilang_berlebih_ganda():
    assert cek_id(ID, ID) == []
    assert any("ganda" in m for m in cek_id(["a1", "a1", "a3"], ID))
    assert any("tidak ada di berkas" in m for m in cek_id(["a1", "a3"], ID))
    assert any("tidak ada di data uji" in m for m in cek_id(["a1", "a2", "a3", "x9"], ID))
    assert cek_id([1, 2], ["1", "2"]) == []                                 # dibandingkan sebagai teks


def test_berkas_wajib_hilang_menggagalkan_kecuali_diizinkan(tmp_path):
    """Kesiapan pengumpulan: berkas yang belum ada = kode keluar 1; dengan izin saat pengerjaan = 0."""
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"id_transaksi": ID}).to_csv(data / "klasifikasi_test.csv.gz", index=False)
    pd.DataFrame({"id_transaksi": ID, "prob": [0.1, 0.2, 0.9], "tandai": [0, 0, 1]}).to_csv(
        tmp_path / "prediksi_klasifikasi.csv", index=False)

    kode, laporan = periksa_semua(tmp_path, data)
    assert kode == 1
    assert any("[LOLOS] prediksi_klasifikasi.csv" in b for b in laporan)
    assert sum("berkas wajib belum ada" in b for b in laporan) == 2

    kode, laporan = periksa_semua(tmp_path, data, izinkan_belum_ada=True)
    assert kode == 0
    assert sum("[BELUM ADA]" in b for b in laporan) == 2


def test_berkas_salah_tetap_gagal_walau_diizinkan(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"id_transaksi": ID}).to_csv(data / "klasifikasi_test.csv.gz", index=False)
    pd.DataFrame({"id_transaksi": ID, "prob": [0.1, 2.0, 0.9], "tandai": [0, 0, 1]}).to_csv(
        tmp_path / "prediksi_klasifikasi.csv", index=False)
    kode, _ = periksa_semua(tmp_path, data, izinkan_belum_ada=True)
    assert kode == 1
