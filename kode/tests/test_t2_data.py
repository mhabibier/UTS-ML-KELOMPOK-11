"""Unit test pembagian waktu dan daftar fitur Task 2.

Data tiruan dipakai HANYA untuk menguji logika pembagian (tanggal, tumpang tindih, jeda),
bukan untuk menghasilkan angka laporan.

Jalankan dari akar repo:  python -m pytest kode/tests -q
"""
import numpy as np
import pandas as pd
import pytest

from t2_data import (BATAS_LABEL_MATANG, FITUR_BOCOR, KOLOM_ID, TARGET,
                     buat_lipatan_waktu, data_matang, fitur_model, tambah_kolom)


@pytest.fixture
def df_tiruan():
    """Satu transaksi per jam dari 1 Jan s.d. 31 Agt 2025, rentang yang sama dengan data latih."""
    waktu = pd.date_range("2025-01-01", "2025-08-31 23:00", freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id_transaksi": [f"t{i}" for i in range(len(waktu))],
        "id_pengguna": rng.integers(0, 100, len(waktu)),
        "id_merchant": rng.integers(0, 10, len(waktu)),
        "waktu": waktu,
        "nominal": rng.uniform(1, 100, len(waktu)),
        "flag_review_manual": rng.integers(0, 2, len(waktu)),
        "umur_merchant_hari": rng.integers(-10, 500, len(waktu)),
        "jumlah_akun_per_perangkat": rng.integers(1, 4, len(waktu)),
        TARGET: rng.integers(0, 2, len(waktu)),
    })


def test_tanggal_sesuai_rencana(df_tiruan):
    """Empat lipatan, validasi 4 minggu, jeda 3 minggu, semua dimulai hari Senin."""
    lipatan = buat_lipatan_waktu(data_matang(df_tiruan))
    harapan = [  # (latih sebelum, valid mulai, valid sebelum)
        ("2025-03-31", "2025-04-21", "2025-05-19"),
        ("2025-04-28", "2025-05-19", "2025-06-16"),
        ("2025-05-26", "2025-06-16", "2025-07-14"),
        ("2025-06-23", "2025-07-14", "2025-08-11"),
    ]
    assert len(lipatan) == 4
    for lp, (batas, mulai, akhir) in zip(lipatan, harapan):
        assert lp.batas_latih == pd.Timestamp(batas)
        assert lp.mulai_valid == pd.Timestamp(mulai)
        assert lp.akhir_valid == pd.Timestamp(akhir)
        assert lp.mulai_valid.day_name() == "Monday"


def test_tidak_ada_label_belum_matang(df_tiruan):
    """Baris dari 11 Agustus ke atas tidak boleh masuk latih maupun validasi."""
    for lp in buat_lipatan_waktu(df_tiruan):
        dipakai = np.concatenate([lp.idx_latih, lp.idx_kalibrasi, lp.idx_valid])
        assert df_tiruan.loc[dipakai, "waktu"].max() < BATAS_LABEL_MATANG


def test_jeda_dan_tanpa_tumpang_tindih(df_tiruan):
    """Data latih selalu sebelum validasi, dengan jarak minimal 21 hari, dan tidak ada baris yang sama."""
    for lp in buat_lipatan_waktu(df_tiruan):
        assert len(np.intersect1d(lp.idx_latih, lp.idx_valid)) == 0
        selisih = df_tiruan.loc[lp.idx_valid, "waktu"].min() - df_tiruan.loc[lp.idx_latih, "waktu"].max()
        assert selisih >= pd.Timedelta(days=21)


def test_jendela_latih_makin_panjang_dan_validasi_bersambung(df_tiruan):
    lipatan = buat_lipatan_waktu(df_tiruan)
    for sebelum, sesudah in zip(lipatan, lipatan[1:]):
        assert set(sebelum.idx_latih) < set(sesudah.idx_latih)
        assert sebelum.akhir_valid == sesudah.mulai_valid
        assert len(np.intersect1d(sebelum.idx_valid, sesudah.idx_valid)) == 0


def test_periode_kalibrasi_terpisah(df_tiruan):
    """Periode kalibrasi (untuk PR 5) terletak di antara latih dan jeda, tidak tumpang tindih dengan keduanya."""
    for lp in buat_lipatan_waktu(df_tiruan, panjang_kalibrasi_hari=28):
        w = df_tiruan["waktu"]
        assert len(lp.idx_kalibrasi) > 0
        assert len(np.intersect1d(lp.idx_latih, lp.idx_kalibrasi)) == 0
        assert len(np.intersect1d(lp.idx_kalibrasi, lp.idx_valid)) == 0
        assert w[lp.idx_latih].max() < w[lp.idx_kalibrasi].min()
        assert w[lp.idx_valid].min() - w[lp.idx_kalibrasi].max() >= pd.Timedelta(days=21)


def test_lipatan_kosong_ditolak(df_tiruan):
    with pytest.raises(ValueError):
        buat_lipatan_waktu(df_tiruan, n_lipatan=20)


def test_fitur_model_tanpa_id_label_dan_bocor(df_tiruan):
    fitur = fitur_model(df_tiruan)
    for kol in KOLOM_ID + FITUR_BOCOR + ["waktu", TARGET]:
        assert kol not in fitur
    assert "nominal" in fitur
    assert "jumlah_akun_per_perangkat" not in fitur_model(df_tiruan, buang=["jumlah_akun_per_perangkat"])


def test_tambah_kolom_menjaga_urutan(df_tiruan):
    """Memasukkan kembali kolom bocor (untuk uji bukti) tidak boleh mengubah urutan kolom lain."""
    fitur = fitur_model(df_tiruan)
    dengan = tambah_kolom(df_tiruan, fitur, ["umur_merchant_hari"])
    assert dengan == [k for k in df_tiruan.columns if k in set(fitur) | {"umur_merchant_hari"}]
    assert [k for k in dengan if k != "umur_merchant_hari"] == fitur


def test_kamus_kategori_hanya_train():
    from t2_data import matriks_lgbm
    train = pd.DataFrame({'kanal': pd.Categorical([' A ', 'b'], categories=[' A ', 'b', 'masa_depan'])})
    valid = pd.DataFrame({'kanal': ['a', 'masa_depan', None]})
    a, b = matriks_lgbm(train, valid, ['kanal'])
    assert a.kanal.cat.categories.tolist() == ['a', 'b']
    assert b.kanal.iloc[0] == 'a'
    assert b.kanal.iloc[1:].isna().all()
    assert 'masa_depan' in train.kanal.cat.categories
