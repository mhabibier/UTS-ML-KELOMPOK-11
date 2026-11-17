"""Task 2 - fitur buatan sendiri. Semua fitur HANYA melihat masa lalu.

Aturan yang dijaga di setiap fungsi (dan diuji di kode/tests/test_t2_fitur.py):
  * nilai fitur sebuah transaksi tidak boleh berubah bila transaksi SESUDAHNYA diubah atau dihapus;
  * label (is_fraud) hanya dipakai bila sudah "diketahui": transaksinya terjadi lebih dari
    `jeda_hari` sebelumnya DAN sebelum `batas_label` (label sesudah batas itu belum matang /
    tidak tersedia pada saat model dilatih).

Fitur riwayat dihitung pada gabungan data latih + uji yang diurutkan menurut waktu, karena
kolom riwayat bawaan data (menit_sejak_trx_terakhir) juga dihitung dengan cara itu.
"""
import numpy as np
import pandas as pd

from t2_data import TARGET

JEDA_LABEL_HARI = 21     # label dianggap matang 3 minggu setelah transaksi (lihat t2_fraud_per_minggu.png)
BOBOT_PRIOR = 100        # penghalusan tingkat fraud: setara 100 transaksi bertingkat fraud rata-rata


# Kelompok fitur yang dievaluasi: nama -> (kolom yang ditambahkan, kolom bawaan yang digantikan)
KELOMPOK_FITUR = {
    "penanda": (["tanpa_riwayat", "melebihi_saldo", "umur_perangkat_hari_bersih"],
                ["umur_perangkat_hari"]),
    "riwayat_pengguna": (["jumlah_trx_sebelum", "sb_log_nominal_sebelum", "z_log_nominal",
                          "pertama_di_merchant", "jumlah_merchant_sebelum"], []),
    "merchant_terlihat": (["hari_sejak_merchant_terlihat", "jumlah_trx_merchant_sebelum"], []),
    "fraud_merchant": (["n_label_merchant", "fraud_label_merchant", "tingkat_fraud_merchant"], []),
    "fraud_pengguna": (["n_label_pengguna", "fraud_label_pengguna", "tingkat_fraud_pengguna"], []),
}


# Susunan yang dipakai mulai PR 4 (diputuskan 17 Nov 2026 dari t2_fitur_kombinasi_ringkas.csv):
# 78 fitur dasar - umur_perangkat_hari + 14 kolom baru (termasuk umur_perangkat_hari_bersih) = 91 fitur. PR-AUC 0,740 vs dasar 0,709 (+0,031, naik di 20/20
# pasangan lipatan x seed). merchant_terlihat TIDAK dipakai: nilainya terus membesar seiring waktu
# sehingga keluar dari rentang data latih pada data baru (lipatan 4 turun saat diikutkan).
KELOMPOK_TERPILIH = ["fraud_merchant", "fraud_pengguna", "riwayat_pengguna", "penanda"]


def susun_fitur(dasar, kelompok):
    """Fitur dasar + kolom dari kelompok-kelompok terpilih (kolom yang digantikan dibuang)."""
    tambah = [k for g in kelompok for k in KELOMPOK_FITUR[g][0]]
    ganti = {k for g in kelompok for k in KELOMPOK_FITUR[g][1]}
    return [k for k in dasar if k not in ganti] + tambah


def gabung_latih_uji(latih, uji):
    """Menggabungkan data latih dan uji (label uji kosong) dengan penanda asal baris."""
    return pd.concat([latih.assign(sumber="latih"), uji.assign(sumber="uji")], ignore_index=True)


def _urutkan(df):
    """Urutan waktu yang stabil; transaksi dengan waktu sama diurutkan menurut id_transaksi."""
    return df.sort_values(["waktu", "id_transaksi"], kind="mergesort")


# --------------------------------------------------------------------------
# 1. Penanda sederhana
# --------------------------------------------------------------------------
def fitur_penanda(df):
    """Penanda dari kolom yang sudah ada (tidak butuh riwayat)."""
    out = pd.DataFrame(index=df.index)
    # transaksi pertama pengguna dalam data (kolom riwayat bawaan kosong)
    out["tanpa_riwayat"] = df["menit_sejak_trx_terakhir"].isna().astype("int8")
    # nominal lebih besar dari saldo sebelum transaksi
    out["melebihi_saldo"] = (df["persen_saldo"] > 1).astype("int8")
    # umur perangkat 0 padahal bukan perangkat baru: kemungkinan besar "tidak tercatat" -> NaN
    tidak_tercatat = (df["umur_perangkat_hari"] == 0) & (df["perangkat_baru"] == 0)
    out["umur_perangkat_hari_bersih"] = df["umur_perangkat_hari"].mask(tidak_tercatat)
    return out


# --------------------------------------------------------------------------
# 2. Riwayat pengguna (tanpa label)
# --------------------------------------------------------------------------
def fitur_riwayat_pengguna(df):
    """Fitur dari transaksi pengguna SEBELUM transaksi ini (transaksi ini sendiri tidak ikut)."""
    s = _urutkan(df)
    pengguna = s["id_pengguna"]
    log_nom = np.log1p(s["nominal"].astype(float))

    n = s.groupby("id_pengguna", sort=False).cumcount()            # jumlah transaksi sebelumnya
    jml1 = log_nom.groupby(pengguna, sort=False).cumsum() - log_nom   # jumlah log nominal sebelumnya
    jml2 = (log_nom ** 2).groupby(pengguna, sort=False).cumsum() - log_nom ** 2
    rata = jml1 / n.where(n > 0)
    var = (jml2 - n * rata ** 2) / (n - 1).where(n > 1)
    sb = np.sqrt(var.clip(lower=0))

    pertama_merchant = (s.groupby(["id_pengguna", "id_merchant"], sort=False).cumcount() == 0).astype("int8")

    out = pd.DataFrame(index=s.index)
    out["jumlah_trx_sebelum"] = n
    out["sb_log_nominal_sebelum"] = sb
    out["z_log_nominal"] = (log_nom - rata) / sb.where(sb > 0)
    out["pertama_di_merchant"] = pertama_merchant
    out["jumlah_merchant_sebelum"] = pertama_merchant.groupby(pengguna, sort=False).cumsum() - pertama_merchant
    return out.reindex(df.index)


# --------------------------------------------------------------------------
# 3. Merchant terlihat (pengganti aman umur_merchant_hari)
# --------------------------------------------------------------------------
def fitur_merchant_terlihat(df):
    """Hari sejak merchant pertama kali terlihat di data dan jumlah transaksinya sebelum ini.

    Catatan: data baru dimulai 1 Januari 2025, jadi merchant lama tampak "baru" di awal data.
    """
    s = _urutkan(df)
    pertama = s.groupby("id_merchant", sort=False)["waktu"].cummin()
    out = pd.DataFrame(index=s.index)
    out["hari_sejak_merchant_terlihat"] = (s["waktu"] - pertama).dt.total_seconds() / 86400
    out["jumlah_trx_merchant_sebelum"] = s.groupby("id_merchant", sort=False).cumcount()
    return out.reindex(df.index)


# --------------------------------------------------------------------------
# 4. Tingkat fraud historis (memakai label yang sudah diketahui saja)
# --------------------------------------------------------------------------
def fitur_fraud_historis(df, kunci, batas_label, jeda_hari=JEDA_LABEL_HARI, bobot_prior=BOBOT_PRIOR):
    """Tingkat fraud masa lalu per `kunci` (id_merchant atau id_pengguna).

    Untuk transaksi pada waktu t, label transaksi lain dipakai hanya bila
        waktu_label < t - jeda_hari   (labelnya sudah matang pada saat t), dan
        waktu_label < batas_label     (label tersedia saat model dilatih).
    `batas_label` = awal periode tanpa label: batas latih sebuah lipatan, atau 11 Agt 2025 untuk
    model final. Dengan begitu transaksi validasi/uji hanya melihat label sampai batas itu,
    persis seperti keadaan nyata (label September-Oktober tidak pernah tersedia).

    Hasil: jumlah transaksi berlabel, jumlah fraud, dan tingkat fraud yang dihaluskan ke arah
    tingkat fraud keseluruhan (juga hanya dari label yang diketahui) supaya kunci dengan sedikit
    transaksi tidak mendapat nilai ekstrem.
    """
    awalan = {"id_merchant": "merchant", "id_pengguna": "pengguna"}.get(kunci, kunci)
    jeda = pd.Timedelta(days=jeda_hari)

    lab = df.loc[df[TARGET].notna() & (df["waktu"] < batas_label), [kunci, "waktu", TARGET]]
    lab = _urutkan(lab.assign(id_transaksi=df.loc[lab.index, "id_transaksi"]))
    lab = lab.rename(columns={"waktu": "t_label"})
    lab["n_kum"] = lab.groupby(kunci, sort=False).cumcount() + 1
    lab["f_kum"] = lab.groupby(kunci, sort=False)[TARGET].cumsum()
    lab["n_glob"] = np.arange(1, len(lab) + 1)
    lab["f_glob"] = lab[TARGET].cumsum()

    q = pd.DataFrame({kunci: df[kunci], "t_tahu": df["waktu"] - jeda}, index=df.index)
    q["_idx"] = q.index
    q = q.sort_values("t_tahu", kind="mergesort")

    # label per kunci yang terjadi SEBELUM t_tahu (bukan sama dengan)
    per_kunci = pd.merge_asof(q, lab[[kunci, "t_label", "n_kum", "f_kum"]], left_on="t_tahu",
                              right_on="t_label", by=kunci, allow_exact_matches=False)
    glob = pd.merge_asof(q[["t_tahu"]], lab[["t_label", "n_glob", "f_glob"]], left_on="t_tahu",
                         right_on="t_label", allow_exact_matches=False)

    n = per_kunci["n_kum"].fillna(0).to_numpy()
    f = per_kunci["f_kum"].fillna(0).to_numpy()
    prior = (glob["f_glob"] / glob["n_glob"]).to_numpy()   # NaN bila belum ada label sama sekali

    out = pd.DataFrame({
        f"n_label_{awalan}": n,
        f"fraud_label_{awalan}": f,
        f"tingkat_fraud_{awalan}": (f + bobot_prior * prior) / (n + bobot_prior),
    }, index=q["_idx"].to_numpy())
    return out.reindex(df.index)


# --------------------------------------------------------------------------
def fitur_statis(df):
    """Semua fitur yang tidak bergantung pada batas label (cukup dihitung sekali)."""
    return pd.concat([fitur_penanda(df), fitur_riwayat_pengguna(df), fitur_merchant_terlihat(df)], axis=1)


def fitur_label(df, batas_label):
    """Fitur yang memakai label; harus dihitung ulang untuk setiap batas label (setiap lipatan)."""
    return pd.concat([fitur_fraud_historis(df, "id_merchant", batas_label),
                      fitur_fraud_historis(df, "id_pengguna", batas_label)], axis=1)
