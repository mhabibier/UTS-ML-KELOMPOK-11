"""Task 2 - Klasifikasi: deteksi transaksi fraud.

Tahap audit (PR 1):
  1. kualitas data (nilai hilang, ID ganda, nilai tidak wajar, kategori tidak konsisten),
  2. keterlambatan label (tingkat fraud per minggu),
  3. uji kebocoran fitur (dampak membuang kolom yang dicurigai).

Tahap validasi (PR 2):
  4. lipatan validasi berbasis waktu (latih | jeda 3 minggu | validasi 4 minggu),
  5. model dasar (tanpa kemampuan, regresi logistik, LightGBM),
  6. pembanding: validasi acak vs validasi berbasis waktu (termasuk versi setara: data validasi
     dan ukuran data latih sama, hanya asal data latih yang berbeda),
  7. PR-AUC per minggu setelah jeda,
  8. uji ulang kolom "dipantau" dan volume_merchant_30h dengan beberapa seed,
  9. sumbangan umur_merchant_hari (sengaja dimasukkan kembali untuk diukur).

Catatan asumsi: batas label 11 Agustus 2025 dan jeda 21 hari adalah asumsi dari pola tingkat fraud
mingguan (data tidak memuat tanggal konfirmasi fraud). Kolom dikecualikan atas dasar kehati-hatian.

Tahap fitur (PR 3):
  10. evaluasi fitur buatan sendiri (kode/t2_fitur.py) per kelompok, 4 lipatan x 5 seed,
  11. seleksi maju kelompok fitur dengan aturan tetap, lalu susunan terpilih vs dasar
      (selisih per minggu = umur informasi label, dan fitur terpenting).

Tahap imbalance (PR 4):
  12. empat cara menangani kelas tidak seimbang (dasar, bobot kelas, undersampling + koreksi
      probabilitas, bobot nominal peka biaya): PR-AUC, kejujuran probabilitas, dan biaya mentah.

Tahap shift (PR 4):
  13a. adversarial validation (data lama vs data uji, dengan dan tanpa kolom kumulatif),
  13b-d. bobot data baru, awal data latih, dan panjang jeda (data validasi sama),
  13e. diagnosis per minggu lipatan 4,
  13f. kematangan label: fraud teramati vs diharapkan model sampai 31 Agt (kepekaan batas label).

Tahap kalibrasi (PR 5):
  14. kalibrasi probabilitas (tanpa, Platt, isotonik) pada periode kalibrasi 4 minggu yang tidak dilihat
      model penghasil skor kalibrasi (model akhir pendekatan b dilatih ulang termasuk periode itu),
      biaya aturan prob x nominal > 25.000 vs tidak menandai / menandai semua / ambang lain,
      diagram reliabilitas, dan pemilihan kombinasi dengan aturan tetap.

Cara menjalankan dari akar repo:
    python kode/task2_klasifikasi.py                    # semua tahap
    python kode/task2_klasifikasi.py --tahap validasi   # hanya validasi
    python kode/task2_klasifikasi.py --tahap fitur      # hanya evaluasi fitur
    python kode/task2_klasifikasi.py --tahap imbalance  # hanya kelas tidak seimbang (PR 4)
    python kode/task2_klasifikasi.py --tahap shift      # hanya pergeseran pola (PR 4)
    python kode/task2_klasifikasi.py --tahap kalibrasi  # hanya kalibrasi dan biaya (PR 5)

Semua tabel dan gambar ditulis ke kode/hasil/.
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # menggambar ke berkas, tanpa jendela
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from t2_data import (BATAS_LABEL_MATANG, FITUR_DIKECUALIKAN, HASIL_DIR, KOLOM_BINER,
                     KOLOM_DIPANTAU, KOLOM_ID, KOLOM_KATEGORI, SEED, TARGET,
                     bersihkan_kategori, buat_lipatan_waktu, data_matang,
                     fitur_model, muat_data, ringkas_lipatan, tambah_kolom, matriks_lgbm)
from t2_fitur import (KELOMPOK_FITUR, KELOMPOK_TERPILIH, fitur_label, fitur_statis,
                      gabung_latih_uji, susun_fitur)
from t2_latih import (BIAYA_TANDAI, biaya, bobot_nominal, bobot_waktu, kalibrasi_isotonik,
                      kalibrasi_platt, koreksi_sampel_kurang, metrik_lengkap, pilih_ambang,
                      rasio_sah_fraud, sampel_kurang, tabel_reliabilitas)

# Banyaknya seed untuk mengukur derau satu kali latih: SEED, SEED+1, ..., SEED+4
N_SEED_ULANG = 5

# Pengaturan LightGBM (belum di-tuning). deterministic + force_row_wise:
# hasil sama persis setiap kali dijalankan dengan seed yang sama.
# Regularisasi (PR 3, 17 Nov): dengan min_child_samples bawaan (20), sesekali satu kali latih "ambruk"
# di lipatan 4 (PR-AUC 0,27-0,44 vs +-0,60 seed lain; ratusan transaksi sah diberi skor > 0,99).
# Daun minimal 200 transaksi (rata-rata +-1-2 fraud per daun) dan penalti bobot L2 mencegah model
# terlalu yakin pada pola kebetulan. Nilai dipilih dari 3 percobaan kecil di lipatan 4, bukan tuning.
PARAM_LGBM = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=200, reg_lambda=5.0,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    n_jobs=-1, verbose=-1, deterministic=True, force_row_wise=True,
)

FMT = lambda x: f"{x:.4f}"  # noqa: E731


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
        print(tabel.to_string(float_format=FMT))
        tabel.to_csv(HASIL_DIR / "t2_pola_nilai_hilang.csv")

    if "persen_saldo" in df and "kanal" in df:
        luar = (df["persen_saldo"] < 0) | (df["persen_saldo"] > 1.0001)
        tabel = (df.assign(di_luar=luar).groupby("kanal")["di_luar"]
                 .agg(transaksi="size", di_luar="sum", persen="mean"))
        print("\npersen_saldo di luar [0, 1] menurut kanal:")
        print(tabel.to_string(float_format=FMT))
        tabel.to_csv(HASIL_DIR / "t2_persen_saldo_per_kanal.csv")


def uji_titik_waktu(df):
    """Memeriksa apakah kolom profil benar-benar dihitung SAAT transaksi (bukan saat data ditarik).

    Logikanya: bila `umur_akun_hari` dihitung saat transaksi, maka
    tanggal transaksi - umur = tanggal daftar akun, dan nilainya harus sama untuk semua transaksi
    milik pengguna yang sama. Hal yang sama berlaku untuk merchant. Umur negatif mustahil terjadi
    saat transaksi: tanggal tersirat berada setelah transaksi. Ini anomali kualitas/indikasi leakage, bukan bukti mekanisme pencatatan.
    """
    print("\n=== 1c. APAKAH KOLOM PROFIL DIHITUNG SAAT TRANSAKSI? ===")
    hari = (df["waktu"].dt.normalize() - df["waktu"].min().normalize()).dt.days
    baris = []
    for kol, kunci in [("umur_akun_hari", "id_pengguna"), ("umur_merchant_hari", "id_merchant")]:
        if kol in df:
            tersirat = (hari - df[kol]).round()
            g = tersirat.groupby(df[kunci])
            konsisten = ((g.max() - g.min()) == 0).mean()
            baris.append({"kolom": kol, "pemeriksaan": f"tanggal daftar tersirat sama dalam satu {kunci}",
                          "hasil": konsisten})
    for kol in ["refund_merchant_30h", "volume_merchant_30h"]:
        if kol in df:
            g = df.groupby("id_merchant")[kol]
            cukup = g.size() >= 20
            konstan = (g.nunique()[cukup] == 1).mean()
            baris.append({"kolom": kol, "pemeriksaan": "nilai konstan sepanjang Jan-Agt dalam satu merchant "
                          "(merchant >= 20 transaksi)", "hasil": konstan})
    tabel = pd.DataFrame(baris)
    print(tabel.to_string(index=False, float_format=FMT))
    tabel.to_csv(HASIL_DIR / "t2_audit_titik_waktu.csv", index=False)
    print("Catatan: kolom 'per 30 hari' seharusnya berubah dari waktu ke waktu; nilai yang konstan\n"
          "menimbulkan dugaan snapshot; pola konstan saja tidak membuktikan waktu pencatatan.")

    if "umur_merchant_hari" in df:
        neg = df["umur_merchant_hari"] < 0
        tgl_daftar = df["waktu"].dt.normalize() - pd.to_timedelta(df["umur_merchant_hari"], unit="D")
        merchant_neg = df.loc[neg, "id_merchant"].unique()
        sama = df["id_merchant"].isin(merchant_neg)
        tabel = pd.DataFrame([
            {"kelompok": "umur_merchant_hari < 0", "transaksi": int(neg.sum()),
             "tingkat_fraud": df.loc[neg, TARGET].mean()},
            {"kelompok": "merchant yang sama, umur >= 0", "transaksi": int((sama & ~neg).sum()),
             "tingkat_fraud": df.loc[sama & ~neg, TARGET].mean()},
            {"kelompok": "merchant lain", "transaksi": int((~sama).sum()),
             "tingkat_fraud": df.loc[~sama, TARGET].mean()},
        ])
        print(f"\numur_merchant_hari negatif: {int(neg.sum()):,} transaksi dari {len(merchant_neg)} merchant; "
              f"memuat {df.loc[neg, TARGET].sum() / df[TARGET].sum():.1%} dari semua fraud")
        print(f"Tanggal daftar merchant tersirat (baris negatif): {tgl_daftar[neg].min().date()} s.d. "
              f"{tgl_daftar[neg].max().date()}  (data ditarik 31 Agt 2025)")
        print(tabel.to_string(index=False, float_format=FMT))
        tabel.to_csv(HASIL_DIR / "t2_umur_merchant_negatif.csv", index=False)

    if {"umur_perangkat_hari", "perangkat_baru"} <= set(df.columns):
        nol = df["umur_perangkat_hari"] == 0
        print(f"\numur_perangkat_hari = 0: {int(nol.sum()):,} baris, "
              f"{int((nol & (df['perangkat_baru'] == 0)).sum()):,} di antaranya perangkat_baru = 0 "
              f"(bertentangan; kemungkinan 0 = tidak tercatat). Fraud: {df.loc[nol, TARGET].mean():.4f} "
              f"vs {df.loc[~nol, TARGET].mean():.4f}")


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

    print(tabel.drop(columns="rasio_terhadap_dasar").tail(10).to_string(float_format=FMT))

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
# Model dan metrik (dipakai tahap 3 dan seterusnya)
# --------------------------------------------------------------------------
def latih_lgbm(latih, fitur, seed=SEED, bobot=None, param=None):
    """LightGBM dengan PARAM_LGBM; `bobot` = bobot per baris, `param` = pengaturan tambahan (PR 4)."""
    model = lgb.LGBMClassifier(random_state=seed, **(PARAM_LGBM | (param or {})))
    x_train, _ = matriks_lgbm(latih, latih.iloc[:0], fitur)
    model.fit(x_train, latih[TARGET], sample_weight=bobot)
    return model


def prob_lgbm(latih, valid, fitur, seed=SEED):
    """Melatih LightGBM pada `latih` lalu mengembalikan probabilitas fraud untuk `valid`."""
    _, x_valid = matriks_lgbm(latih, valid, fitur)
    return latih_lgbm(latih, fitur, seed).predict_proba(x_valid)[:, 1]


def prob_logistik(latih, valid, fitur):
    """Model dasar linear: imputasi median (+ penanda nilai hilang), standardisasi, one-hot kategori."""
    kat = [k for k in fitur if k in KOLOM_KATEGORI]
    num = [k for k in fitur if k not in kat]
    pra = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler()), num),
        ("kat", OneHotEncoder(handle_unknown="ignore"), kat),
    ])
    model = Pipeline([("pra", pra), ("lr", LogisticRegression(max_iter=2000))])
    model.fit(latih[fitur], latih[TARGET])
    return model.predict_proba(valid[fitur])[:, 1]


def metrik(y, prob):
    return {"pr_auc": average_precision_score(y, prob), "roc_auc": roc_auc_score(y, prob)}


def pr_auc_per_lipatan(df, lipatan, fitur, seed=SEED):
    """PR-AUC LightGBM pada setiap lipatan waktu."""
    hasil = []
    for lp in lipatan:
        valid = df.loc[lp.idx_valid]
        hasil.append(average_precision_score(valid[TARGET],
                                             prob_lgbm(df.loc[lp.idx_latih], valid, fitur, seed)))
    return hasil


# --------------------------------------------------------------------------
# 3. Uji kebocoran fitur
# --------------------------------------------------------------------------
def uji_kebocoran(df, df_bersih):
    """`df` = data mentah (untuk tabel dan AUC tunggal), `df_bersih` = kategori sudah diseragamkan."""
    print("\n=== 3. UJI KEBOCORAN FITUR ===")

    # (a) Hubungan langsung kolom curiga dengan label
    if "flag_review_manual" in df:
        print("\nTingkat fraud menurut flag_review_manual:")
        tabel = df.groupby("flag_review_manual")[TARGET].agg(transaksi="size", tingkat="mean")
        print(tabel.to_string(float_format=FMT))
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
    print(saring.head(12).to_string(index=False, float_format=FMT))

    # (c) Dampak membuang flag_review_manual pada PR-AUC, rata-rata empat lipatan waktu.
    # Di sini flag_review_manual SENGAJA disertakan di "semua fitur" untuk menunjukkan dampaknya.
    # Kolom curiga lain diuji dengan beberapa seed di tahap validasi (bagian 8).
    print("\nDampak membuang flag_review_manual (PR-AUC LightGBM, rata-rata 4 lipatan waktu):")
    dm = data_matang(df_bersih)
    lipatan = buat_lipatan_waktu(dm)
    semua = tambah_kolom(dm, fitur_model(dm), ["flag_review_manual"])
    skenario = [("(tidak ada)", []), ("flag_review_manual", ["flag_review_manual"])]
    hasil = []
    for nama, buang in skenario:
        skor = pr_auc_per_lipatan(dm, lipatan, [k for k in semua if k not in buang])
        hasil.append({"fitur_dibuang": nama, "pr_auc_rata2": np.mean(skor),
                      **{f"pr_auc_lipatan{lp.nomor}": s for lp, s in zip(lipatan, skor)}})
    tabel = pd.DataFrame(hasil)
    # PR-AUC model tanpa kemampuan = tingkat fraud di data validasi
    tabel["pr_auc_tanpa_kemampuan"] = np.mean([dm.loc[lp.idx_valid, TARGET].mean() for lp in lipatan])
    tabel.to_csv(HASIL_DIR / "t2_dampak_kebocoran.csv", index=False)
    print(tabel.to_string(index=False, float_format=FMT))
    print("\nBaca hasilnya: penurunan tajam saat sebuah kolom dibuang berarti model sangat bergantung pada\n"
          "kolom itu. Ini menimbulkan dugaan kebocoran, tetapi tidak membuktikan kapan kolom dicatat;\n"
          "keputusan membuang diambil dari makna kolom di kamus data (kehati-hatian).")


# --------------------------------------------------------------------------
# 4-8. Validasi berbasis waktu
# --------------------------------------------------------------------------
def gambar_lipatan(dm, lipatan):
    """Diagram periode tiap lipatan dan pembagian final (latih s.d. 10 Agt -> uji Sep-Okt)."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    awal = dm["waktu"].min().normalize()
    warna = {"latih": "#1565c0", "jeda": "#bdbdbd", "valid": "#ef6c00", "uji": "#2e7d32"}

    def bar(y, mulai, akhir, jenis):
        ax.broken_barh([(mdates.date2num(mulai), mdates.date2num(akhir) - mdates.date2num(mulai))],
                       (y - 0.35, 0.7), color=warna[jenis])

    label = []
    for i, lp in enumerate(lipatan):
        bar(i, awal, lp.batas_latih, "latih")
        bar(i, lp.akhir_kalibrasi, lp.mulai_valid, "jeda")
        bar(i, lp.mulai_valid, lp.akhir_valid, "valid")
        label.append(f"Lipatan {lp.nomor}")
    y = len(lipatan)
    bar(y, awal, BATAS_LABEL_MATANG, "latih")
    bar(y, BATAS_LABEL_MATANG, pd.Timestamp("2025-09-01"), "jeda")
    bar(y, pd.Timestamp("2025-09-01"), pd.Timestamp("2025-11-01"), "uji")
    label.append("Final")
    ax.axvline(mdates.date2num(BATAS_LABEL_MATANG), color="#c62828", ls="--", lw=1)
    ax.text(mdates.date2num(BATAS_LABEL_MATANG), -0.9, " label belum matang ->",
            color="#c62828", fontsize=8, va="center")

    ax.set_yticks(range(len(label)), label)
    ax.invert_yaxis()
    ax.set_ylim(len(label) - 0.4, -1.3)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in warna.values()]
    ax.legend(handles, ["Latih", "Jeda 3 minggu", "Validasi 4 minggu", "Data uji"],
              loc="lower left", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0, -0.32))
    ax.set_title("Lipatan validasi berbasis waktu (jendela latih makin panjang)")
    fig.tight_layout()
    fig.savefig(HASIL_DIR / "t2_validasi_lipatan.png", dpi=150)
    plt.close(fig)


def validasi(df_bersih):
    print("\n=== 4. LIPATAN VALIDASI BERBASIS WAKTU ===")
    dm = data_matang(df_bersih)
    print(f"Data diasumsikan matang (sebelum {BATAS_LABEL_MATANG.date()}): {len(dm):,} baris, "
          f"{int(dm[TARGET].sum()):,} fraud ({dm[TARGET].mean():.4%})")
    lipatan = buat_lipatan_waktu(dm)
    ringkas = ringkas_lipatan(dm, lipatan)
    ringkas.to_csv(HASIL_DIR / "t2_validasi_lipatan.csv", index=False)
    print(ringkas.to_string(index=False, float_format=FMT))
    gambar_lipatan(dm, lipatan)

    fitur = fitur_model(dm)
    print(f"Jumlah fitur: {len(fitur)} (tanpa ID, waktu, label, dan {', '.join(FITUR_DIKECUALIKAN)})")

    # Simpanan prediksi LightGBM supaya model yang sama tidak dilatih dua kali:
    # kunci (nomor lipatan, seed, daftar fitur)
    cache = {}

    def prob_lgbm_cache(lp, seed, f=None):
        f = fitur if f is None else f
        kunci = (lp.nomor, seed, tuple(f))
        if kunci not in cache:
            cache[kunci] = prob_lgbm(dm.loc[lp.idx_latih], dm.loc[lp.idx_valid], f, seed)
        return cache[kunci]

    # ---- 5. Model dasar ----
    print("\n=== 5. MODEL DASAR (validasi berbasis waktu) ===")
    baris = []
    for lp in lipatan:
        latih, valid = dm.loc[lp.idx_latih], dm.loc[lp.idx_valid]
        y = valid[TARGET]
        tingkat = y.mean()
        # Model tanpa kemampuan: skor sama untuk semua transaksi -> PR-AUC = tingkat fraud, ROC-AUC = 0,5
        baris.append({"lipatan": lp.nomor, "model": "tanpa kemampuan", "pr_auc": tingkat, "roc_auc": 0.5})
        baris.append({"lipatan": lp.nomor, "model": "regresi logistik", **metrik(y, prob_logistik(latih, valid, fitur))})
        baris.append({"lipatan": lp.nomor, "model": "LightGBM", **metrik(y, prob_lgbm_cache(lp, SEED))})
        print(f"  lipatan {lp.nomor} selesai")
    per_lipatan = pd.DataFrame(baris)
    per_lipatan.to_csv(HASIL_DIR / "t2_validasi_model_dasar_per_lipatan.csv", index=False)
    ringkas_model = (per_lipatan.groupby("model", sort=False)[["pr_auc", "roc_auc"]]
                     .agg(["mean", "std"]))
    ringkas_model.columns = [f"{a}_{b}" for a, b in ringkas_model.columns]
    ringkas_model.to_csv(HASIL_DIR / "t2_validasi_model_dasar.csv")
    print(per_lipatan.pivot(index="model", columns="lipatan", values="pr_auc")
          .to_string(float_format=FMT))
    print("\nRata-rata dan simpangan baku 4 lipatan:")
    print(ringkas_model.to_string(float_format=FMT))

    # ---- 6. Validasi acak vs validasi berbasis waktu ----
    # (a) Acak biasa: 5 lipatan bertingkat pada seluruh data. Ukuran data latih DAN periode yang
    #     dinilai berbeda dengan validasi berbasis waktu, jadi selisihnya bukan efek murni pengacakan.
    # (b) Acak setara: untuk setiap lipatan waktu, data validasi SAMA dan ukuran data latih SAMA;
    #     bedanya hanya data latih diambil acak dari seluruh periode (boleh sesudah validasi dan dari
    #     masa jeda), bukan dari masa lalu saja. Selisih (b) - waktu = manfaat "melihat masa depan".
    seeds = [SEED + k for k in range(N_SEED_ULANG)]
    print(f"\n=== 6. VALIDASI ACAK vs VALIDASI BERBASIS WAKTU (LightGBM, fitur sama) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    baris = []
    for i, (il, iv) in enumerate(skf.split(dm, dm[TARGET]), start=1):
        latih, valid = dm.iloc[il], dm.iloc[iv]
        baris.append({"cara": "acak biasa (5 lipatan bertingkat)", "lipatan": i, "seed": SEED,
                      **metrik(valid[TARGET], prob_lgbm(latih, valid, fitur))})
    for lp in lipatan:
        valid = dm.loc[lp.idx_valid]
        calon = dm.index.difference(lp.idx_valid).to_numpy()
        for s in seeds:
            rng = np.random.default_rng(1000 * s + lp.nomor)
            idx_acak = np.sort(rng.choice(calon, size=len(lp.idx_latih), replace=False))
            baris.append({"cara": "acak setara (validasi & ukuran latih sama)", "lipatan": lp.nomor,
                          "seed": s, **metrik(valid[TARGET], prob_lgbm(dm.loc[idx_acak], valid, fitur, s))})
            baris.append({"cara": "berbasis waktu (masa lalu, jeda 3 minggu)", "lipatan": lp.nomor,
                          "seed": s, **metrik(valid[TARGET], prob_lgbm_cache(lp, s))})
        print(f"  lipatan {lp.nomor} selesai")
    banding = pd.DataFrame(baris)
    banding.to_csv(HASIL_DIR / "t2_validasi_acak_vs_waktu.csv", index=False)
    rb = banding.groupby("cara", sort=False)[["pr_auc", "roc_auc"]].agg(["mean", "std"])
    print(rb.to_string(float_format=FMT))
    setara = banding[banding["cara"] != "acak biasa (5 lipatan bertingkat)"]
    per_lip = setara.pivot_table(index="lipatan", columns="cara", values="pr_auc", aggfunc="mean")
    per_lip["selisih_setara_minus_waktu"] = (per_lip["acak setara (validasi & ukuran latih sama)"]
                                             - per_lip["berbasis waktu (masa lalu, jeda 3 minggu)"])
    per_lip.to_csv(HASIL_DIR / "t2_validasi_acak_setara_per_lipatan.csv")
    print("\nPer lipatan (rata-rata seed), data validasi sama:")
    print(per_lip.to_string(float_format=FMT))
    print("\nBaca hasilnya: acak biasa merata-ratakan semua bulan sehingga menyembunyikan penurunan di\n"
          "periode terbaru; acak setara mengukur berapa banyak 'melihat masa depan' membantu bila\n"
          "data validasi dan ukuran data latih disamakan.")

    # ---- 7. PR-AUC per minggu setelah jeda ----
    print("\n=== 7. PR-AUC LightGBM PER MINGGU SETELAH JEDA ===")
    baris = []
    for lp in lipatan:
        valid = dm.loc[lp.idx_valid]
        prob = pd.Series(prob_lgbm_cache(lp, SEED), index=valid.index)
        minggu_ke = ((valid["waktu"] - lp.mulai_valid).dt.days // 7) + 1
        for m, idx in valid.groupby(minggu_ke).groups.items():
            y = valid.loc[idx, TARGET]
            baris.append({"lipatan": lp.nomor, "minggu_ke": int(m), "transaksi": len(y),
                          "fraud": int(y.sum()), "pr_auc": average_precision_score(y, prob[idx])})
    per_minggu = pd.DataFrame(baris)
    per_minggu.to_csv(HASIL_DIR / "t2_validasi_per_minggu.csv", index=False)
    print(per_minggu.pivot(index="lipatan", columns="minggu_ke", values="pr_auc")
          .to_string(float_format=FMT))
    rata = per_minggu.groupby("minggu_ke").agg(fraud=("fraud", "sum"), pr_auc_rata2=("pr_auc", "mean"))
    print("\nRata-rata 4 lipatan (minggu 1 = 4 minggu setelah data latih terakhir):")
    print(rata.to_string(float_format=FMT))
    print("Catatan: tiap sel hanya berisi sekitar 70-90 fraud, jadi selisih antarminggu sangat berderau.")

    # ---- 8. Uji ulang kolom dipantau (dan volume_merchant_30h) dengan beberapa seed ----
    # Untuk tiap kolom: model DENGAN kolom vs model TANPA kolom pada lipatan dan seed yang sama.
    # Kolom dipantau ada di model dasar (diukur dengan membuangnya); volume_merchant_30h sudah
    # dikecualikan (diukur dengan memasukkannya kembali).
    kolom_uji = KOLOM_DIPANTAU + [k for k in ["volume_merchant_30h"] if k in dm.columns]
    print(f"\n=== 8. UJI ULANG KOLOM DIPANTAU ({len(lipatan)} lipatan x {N_SEED_ULANG} seed) ===")
    seeds = [SEED + k for k in range(N_SEED_ULANG)]
    baris = []
    for lp in lipatan:
        y = dm.loc[lp.idx_valid, TARGET]
        for s in seeds:
            for kol in kolom_uji:
                dengan = tambah_kolom(dm, fitur, [kol])
                tanpa = [k for k in dengan if k != kol]
                p_dengan = average_precision_score(y, prob_lgbm_cache(lp, s, dengan))
                p_tanpa = average_precision_score(y, prob_lgbm_cache(lp, s, tanpa))
                baris.append({"kolom_dibuang": kol, "status": "dipantau" if kol in fitur else "dikecualikan",
                              "lipatan": lp.nomor, "seed": s,
                              "pr_auc_dasar": p_dengan, "pr_auc_dibuang": p_tanpa,
                              "selisih": p_tanpa - p_dengan})
        print(f"  lipatan {lp.nomor} selesai")
    ulang = pd.DataFrame(baris)
    ulang.to_csv(HASIL_DIR / "t2_validasi_kolom_dipantau.csv", index=False)

    # Derau seed: simpangan baku PR-AUC model dasar antar-seed dalam lipatan yang sama
    derau_seed = (ulang[ulang["status"] == "dipantau"].drop_duplicates(["lipatan", "seed"])
                  .groupby("lipatan")["pr_auc_dasar"].std().mean())
    ringkas_ulang = ulang.groupby("kolom_dibuang").agg(
        status=("status", "first"),
        selisih_rata2=("selisih", "mean"),
        selisih_sb=("selisih", "std"),
        n_pasangan=("selisih", "size"),
        n_naik=("selisih", lambda s: int((s > 0).sum())),
    )
    # Galat baku selisih rata-rata (pasangan dianggap kasar saling bebas; antarseed dalam satu
    # lipatan sebenarnya berkorelasi, jadi angka ini cenderung terlalu kecil)
    ringkas_ulang["galat_baku"] = ringkas_ulang["selisih_sb"] / np.sqrt(ringkas_ulang["n_pasangan"])
    per_lip = ulang.groupby(["kolom_dibuang", "lipatan"])["selisih"].mean().unstack()
    per_lip.columns = [f"selisih_lipatan{c}" for c in per_lip.columns]
    ringkas_ulang = ringkas_ulang.join(per_lip)
    ringkas_ulang["derau_seed_sb"] = derau_seed
    ringkas_ulang.to_csv(HASIL_DIR / "t2_validasi_kolom_dipantau_ringkas.csv")
    print(ringkas_ulang.to_string(float_format=FMT))
    print("\nBaca hasilnya: selisih = PR-AUC tanpa kolom - PR-AUC dengan kolom.\n"
          "Penurunan besar dan konsisten di semua lipatan (seperti flag_review_manual) menimbulkan\n"
          "dugaan kebocoran. Selisih kecil yang arahnya berganti-ganti tidak menunjukkan tanda itu.\n"
          "Data saja tidak bisa memastikan waktu pencatatan; keputusan juga memakai makna kolom di kamus.\n"
          "volume_merchant_30h tetap dikecualikan walau selisihnya kecil: nilainya konstan per merchant.")

    # ---- 9. Sumbangan umur_merchant_hari ----
    # Kolom ini sudah dikecualikan dari model. Di sini SENGAJA dimasukkan kembali untuk
    # mengukur sumbangannya. Prediksi yang sama (tanpa melatih ulang) dinilai dua kali: pada semua
    # baris validasi dan pada baris yang umurnya tidak negatif. Bila keunggulannya hilang setelah
    # baris negatif disisihkan, sumbangan kolom ini berasal dari baris yang mustahil (umur < 0).
    if "umur_merchant_hari" in dm.columns:
        print(f"\n=== 9. SUMBANGAN umur_merchant_hari ({len(lipatan)} lipatan x {len(seeds)} seed) ===")
        fitur_dengan = tambah_kolom(dm, fitur, ["umur_merchant_hari"])
        baris = []
        for lp in lipatan:
            valid = dm.loc[lp.idx_valid]
            normal = (valid["umur_merchant_hari"] >= 0).to_numpy()
            neg = ~normal
            for s in seeds:
                p_dasar = prob_lgbm_cache(lp, s, fitur_dengan)
                p_buang = prob_lgbm_cache(lp, s)
                for nama, m in [("semua baris", np.ones(len(valid), bool)), ("tanpa umur < 0", normal)]:
                    y = valid[TARGET].to_numpy()[m]
                    baris.append({"evaluasi": nama, "lipatan": lp.nomor, "seed": s,
                                  "baris_negatif": int(neg.sum()),
                                  "fraud_negatif": int(valid[TARGET].to_numpy()[neg].sum()),
                                  "fraud_total": int(valid[TARGET].sum()),
                                  "pr_auc_dengan": average_precision_score(y, p_dasar[m]),
                                  "pr_auc_tanpa": average_precision_score(y, p_buang[m])})
        sumber = pd.DataFrame(baris)
        sumber["selisih"] = sumber["pr_auc_tanpa"] - sumber["pr_auc_dengan"]
        sumber.to_csv(HASIL_DIR / "t2_validasi_umur_merchant.csv", index=False)
        porsi = (sumber.drop_duplicates("lipatan")
                 [["lipatan", "baris_negatif", "fraud_negatif", "fraud_total"]])
        print("Baris umur negatif di tiap lipatan validasi:")
        print(porsi.to_string(index=False))
        ringkas_umur = sumber.groupby("evaluasi", sort=False).agg(
            pr_auc_dengan=("pr_auc_dengan", "mean"), pr_auc_tanpa=("pr_auc_tanpa", "mean"),
            selisih_rata2=("selisih", "mean"), selisih_sb=("selisih", "std"),
            n_turun=("selisih", lambda s: int((s < 0).sum())), n_pasangan=("selisih", "size"))
        ringkas_umur.to_csv(HASIL_DIR / "t2_validasi_umur_merchant_ringkas.csv")
        print(f"\nPR-AUC rata-rata ({len(lipatan)} lipatan x {len(seeds)} seed); "
              "selisih = tanpa kolom - dengan kolom:")
        print(ringkas_umur.to_string(float_format=FMT))


# --------------------------------------------------------------------------
# 10. Evaluasi fitur buatan sendiri (PR 3)
# --------------------------------------------------------------------------
def siapkan_data(latih, uji):
    """Gabungan latih + uji dengan fitur statis, data matang, lipatan waktu, dan 77 fitur dasar.

    `latih` dan `uji` = data mentah. Fitur riwayat dihitung pada gabungan keduanya (masa lalu saja).
    """
    gab = bersihkan_kategori(gabung_latih_uji(latih, uji))
    gab = pd.concat([gab, fitur_statis(gab)], axis=1)
    dm = gab[(gab["sumber"] == "latih") & (gab["waktu"] < BATAS_LABEL_MATANG)]
    return gab, dm, buat_lipatan_waktu(dm), fitur_model(latih)


def evaluasi_fitur(gab, dm, lipatan, dasar):
    """Membandingkan model dasar dengan model + setiap kelompok fitur baru (4 lipatan x N seed).

    Masukan = hasil siapkan_data. Fitur berlabel (tingkat fraud historis) dihitung ulang untuk setiap lipatan dengan batas label
    = batas latih lipatan itu, sehingga baris validasi hanya melihat label yang tersedia saat
    model dilatih, sama seperti data uji nanti.
    """
    print(f"\n=== 10. EVALUASI FITUR BUATAN SENDIRI ===")

    # Gambaran singkat: tingkat fraud menurut penanda biner baru (data matang)
    baris = []
    for kol in ["tanpa_riwayat", "melebihi_saldo", "pertama_di_merchant"]:
        for nilai, sub in dm.groupby(kol)[TARGET]:
            baris.append({"fitur": kol, "nilai": int(nilai), "transaksi": len(sub), "tingkat_fraud": sub.mean()})
    gambaran = pd.DataFrame(baris)
    gambaran.to_csv(HASIL_DIR / "t2_fitur_gambaran.csv", index=False)
    print("Tingkat fraud menurut penanda baru:")
    print(gambaran.to_string(index=False, float_format=FMT))

    penilai = PenilaiFitur(gab, dm, lipatan, dasar)

    # ---- 10. Satu kelompok per konfigurasi, lalu semua kelompok sekaligus ----
    konfigurasi = {"dasar": []} | {g: [g] for g in KELOMPOK_FITUR} | {"semua": list(KELOMPOK_FITUR)}
    print(f"Konfigurasi: {', '.join(konfigurasi)}  ({len(lipatan)} lipatan x {N_SEED_ULANG} seed)")
    hasil = pd.concat([penilai.nilai(k).assign(konfigurasi=n) for n, k in konfigurasi.items()])
    ringkas10 = ringkas_selisih(hasil, "dasar", "t2_fitur")

    # ---- 11. Seleksi maju dengan aturan tetap ----
    # Urutan: kelompok dengan selisih rata-rata positif di bagian 10, dari yang terbesar.
    # Aturan (ditetapkan sebelum melihat hasil seleksi): kelompok DITERIMA bila, dibanding susunan
    # yang sudah diterima, selisih PR-AUC rata-ratanya > 0 DAN naik di sedikitnya 3 dari 4 lipatan.
    print("\n=== 11. SELEKSI MAJU KELOMPOK FITUR ===")
    urutan = [g for g in ringkas10.sort_values("selisih_rata2", ascending=False).index
              if g in KELOMPOK_FITUR and ringkas10.loc[g, "selisih_rata2"] > 0]
    print(f"Urutan dicoba: {', '.join(urutan)}")
    terpilih, langkah = [], []
    for g in urutan:
        acuan, calon = penilai.nilai(terpilih), penilai.nilai(terpilih + [g])
        selisih = calon.set_index(["lipatan", "seed"])["pr_auc"] - acuan.set_index(["lipatan", "seed"])["pr_auc"]
        per_lip = selisih.groupby(level="lipatan").mean()
        lolos = bool(selisih.mean() > 0 and (per_lip > 0).sum() >= 3)
        langkah.append({"langkah": len(langkah) + 1, "kelompok": g, "acuan": "+".join(terpilih) or "dasar",
                        "pr_auc_acuan": acuan["pr_auc"].mean(), "pr_auc_calon": calon["pr_auc"].mean(),
                        "selisih_rata2": selisih.mean(), "n_naik": int((selisih > 0).sum()),
                        "lipatan_naik": int((per_lip > 0).sum()),
                        **{f"selisih_lipatan{i}": v for i, v in per_lip.items()},
                        "keputusan": "diterima" if lolos else "ditolak"})
        if lolos:
            terpilih.append(g)
    seleksi = pd.DataFrame(langkah)
    seleksi.to_csv(HASIL_DIR / "t2_fitur_seleksi.csv", index=False)
    print(seleksi.to_string(index=False, float_format=FMT))
    print(f"\nKelompok terpilih: {terpilih}")
    if sorted(terpilih) != sorted(KELOMPOK_TERPILIH):
        print(f"PERHATIAN: berbeda dengan KELOMPOK_TERPILIH di t2_fitur.py ({KELOMPOK_TERPILIH}); perbarui konstanta itu.")

    # Susunan terpilih dibanding dasar: ringkasan, selisih per minggu, dan fitur terpenting
    akhir = pd.concat([penilai.nilai([]).assign(konfigurasi="dasar"),
                       penilai.nilai(terpilih).assign(konfigurasi="terpilih")])
    ringkas_selisih(akhir, "dasar", "t2_fitur_terpilih")
    print("Catatan: skor ini memakai lipatan yang juga dipakai memilih kelompok fitur; "
          "ini hasil pengembangan, bukan evaluasi akhir yang independen.")
    mg = pd.concat([penilai.mingguan([]).assign(konfigurasi="dasar"),
                    penilai.mingguan(terpilih).assign(konfigurasi="terpilih")])
    acuan = mg[mg["konfigurasi"] == "dasar"].set_index(["lipatan", "seed", "minggu_ke"])["pr_auc"]
    mg["selisih"] = mg["pr_auc"] - mg.set_index(["lipatan", "seed", "minggu_ke"]).index.map(acuan)
    mg.to_csv(HASIL_DIR / "t2_fitur_terpilih_per_minggu.csv", index=False)
    tabel = mg[mg["konfigurasi"] == "terpilih"].groupby("minggu_ke")["selisih"].mean()
    tabel.index = [f"minggu{c} ({21 + 7 * (c - 1)}-{21 + 7 * c} hari)" for c in tabel.index]
    print("\nSelisih PR-AUC terpilih vs dasar per minggu validasi (umur informasi label dalam kurung):")
    print(tabel.to_string(float_format=FMT))
    pentingnya = penilai.pentingnya(terpilih)
    pentingnya.to_csv(HASIL_DIR / "t2_fitur_terpilih_pentingnya.csv")
    baru = {k for g in KELOMPOK_FITUR.values() for k in g[0]}
    print("\n20 fitur terpenting pada susunan terpilih (porsi gain, rata-rata 4 lipatan; * = fitur baru):")
    for kol, nilai in pentingnya["rata2"].head(20).items():
        print(f"  {'*' if kol in baru else ' '} {kol:32s} {nilai:.4f}")


class PenilaiFitur:
    """Melatih dan menilai susunan fitur (4 lipatan x N seed), dengan simpanan hasil per susunan.

    Fitur berlabel dihitung sekali per lipatan (batas label = batas latih lipatan itu). Selain PR-AUC
    per lipatan, disimpan juga PR-AUC per minggu validasi: pada minggu ke-k informasi label berumur
    sekitar 21 + 7(k-1) s.d. 21 + 7k hari (di data uji 21-82 hari), dan gain fitur untuk seed utama.
    """

    def __init__(self, gab, dm, lipatan, dasar):
        self.dm, self.lipatan, self.dasar = dm, lipatan, dasar
        self.seeds = [SEED + k for k in range(N_SEED_ULANG)]
        self.label = {lp.nomor: fitur_label(gab, lp.batas_latih).loc[dm.index] for lp in lipatan}
        self._simpan = {}

    def _jalankan(self, kelompok):
        kunci = tuple(sorted(kelompok))
        if kunci in self._simpan:
            return self._simpan[kunci]
        fitur = susun_fitur(self.dasar, kelompok)
        baris, mingguan, penting = [], [], []
        for lp in self.lipatan:
            data = pd.concat([self.dm, self.label[lp.nomor]], axis=1)
            latih_lp, valid_lp = data.loc[lp.idx_latih], data.loc[lp.idx_valid]
            y = valid_lp[TARGET].to_numpy()
            minggu_ke = (((valid_lp["waktu"] - lp.mulai_valid).dt.days // 7) + 1).to_numpy()
            _, x_valid = matriks_lgbm(latih_lp, valid_lp, fitur)   # kamus kategori dari latih saja
            for s in self.seeds:
                model = latih_lgbm(latih_lp, fitur, s)
                prob = model.predict_proba(x_valid)[:, 1]
                baris.append({"n_fitur": len(fitur), "lipatan": lp.nomor, "seed": s, **metrik(y, prob)})
                for m in np.unique(minggu_ke):
                    pilih = minggu_ke == m
                    mingguan.append({"lipatan": lp.nomor, "seed": s, "minggu_ke": int(m),
                                     "pr_auc": average_precision_score(y[pilih], prob[pilih])})
                if s == SEED:
                    gain = model.booster_.feature_importance(importance_type="gain")
                    penting.append(pd.Series(gain / gain.sum(), index=fitur, name=lp.nomor))
        print(f"  dinilai: {'+'.join(kunci) or 'dasar'} ({len(fitur)} fitur)")
        self._simpan[kunci] = (pd.DataFrame(baris), pd.DataFrame(mingguan), pd.concat(penting, axis=1))
        return self._simpan[kunci]

    def nilai(self, kelompok):
        return self._jalankan(kelompok)[0].copy()

    def mingguan(self, kelompok):
        return self._jalankan(kelompok)[1].copy()

    def pentingnya(self, kelompok):
        p = self._jalankan(kelompok)[2].copy()
        p["rata2"] = p.mean(axis=1)
        p.index.name = "fitur"
        return p.sort_values("rata2", ascending=False)


def ringkas_selisih(hasil, acuan, awalan):
    """Selisih PR-AUC berpasangan (lipatan dan seed sama) setiap konfigurasi terhadap `acuan`."""
    skor_acuan = hasil[hasil["konfigurasi"] == acuan].set_index(["lipatan", "seed"])["pr_auc"]
    hasil = hasil.copy()
    hasil["selisih"] = hasil["pr_auc"] - hasil.set_index(["lipatan", "seed"]).index.map(skor_acuan)
    hasil.to_csv(HASIL_DIR / f"{awalan}_per_lipatan.csv", index=False)
    ringkas = hasil.groupby("konfigurasi", sort=False).agg(
        n_fitur=("n_fitur", "first"),
        pr_auc_rata2=("pr_auc", "mean"),
        pr_auc_median=("pr_auc", "median"),
        selisih_rata2=("selisih", "mean"),
        selisih_sb=("selisih", "std"),
        n_naik=("selisih", lambda s: int((s > 0).sum())),
        n_pasangan=("selisih", "size"),
    )
    per_lip = hasil.groupby(["konfigurasi", "lipatan"], sort=False)["selisih"].mean().unstack()
    per_lip.columns = [f"selisih_lipatan{c}" for c in per_lip.columns]
    ringkas = ringkas.join(per_lip)
    ringkas.to_csv(HASIL_DIR / f"{awalan}_ringkas.csv")
    print(f"\nPR-AUC dan selisih terhadap '{acuan}' (selisih > 0 = fitur membantu):")
    print(ringkas.to_string(float_format=FMT))
    return ringkas


# --------------------------------------------------------------------------
# 12-13. Kelas tidak seimbang dan pergeseran pola (PR 4)
# --------------------------------------------------------------------------
# Derau antar-seed setelah regularisasi +-0,005 (PR 3), jadi 3 seed cukup untuk PR 4.
N_SEED_PR4 = 3
RASIO_UNDERSAMPLING = 20   # transaksi sah per fraud yang disisakan saat undersampling

# Kolom yang nilainya terus bertambah sejak data dimulai (1 Jan 2025). Pada data uji nilainya pasti
# lebih besar daripada di data latih, jadi kolom ini diduga paling membedakan periode lama dan baru.
KOLOM_KUMULATIF = ["jumlah_trx_merchant_sebelum", "hari_sejak_merchant_terlihat", "jumlah_trx_sebelum",
                   "jumlah_merchant_sebelum", "n_label_merchant", "fraud_label_merchant"]


class Eksperimen:
    """Melatih LightGBM (susunan fitur terpilih) dengan berbagai "cara" latih, 4 lipatan x N seed.

    Sebuah cara adalah fungsi (latih, lipatan, seed) -> dict dengan kunci opsional:
      latih   : baris latih yang dipakai (misalnya hasil undersampling atau latih yang dipersingkat),
      fitur   : daftar fitur pengganti (bawaan: susunan terpilih),
      bobot   : bobot per baris latih,   param: pengaturan LightGBM tambahan,
      koreksi : fungsi pengubah probabilitas (misalnya koreksi undersampling),
      tandai  : fungsi (prob, nominal) -> keputusan tandai (bawaan: prob x nominal > 25.000).
    Fitur berlabel dihitung ulang untuk setiap batas latih (disimpan), sama seperti PenilaiFitur.
    """

    def __init__(self, gab, dm, fitur, seeds):
        self.gab, self.dm, self.fitur, self.seeds = gab, dm, fitur, seeds
        self._label, self._hasil = {}, {}
        self.prob = {}   # (nama cara, nomor lipatan) -> probabilitas seed pertama untuk baris validasi

    def label(self, batas):
        """Fitur berlabel untuk SEMUA baris gabungan dengan batas label `batas`."""
        if batas not in self._label:
            self._label[batas] = fitur_label(self.gab, batas)
        return self._label[batas]

    def data_lipatan(self, lp):
        data = pd.concat([self.dm, self.label(lp.batas_latih).loc[self.dm.index]], axis=1)
        return data.loc[lp.idx_latih], data.loc[lp.idx_valid]

    def jalankan(self, nama, lipatan, cara=None):
        kunci = (nama, tuple(lp.batas_latih for lp in lipatan))
        if kunci in self._hasil:
            return self._hasil[kunci].copy()
        baris = []
        for lp in lipatan:
            latih, valid = self.data_lipatan(lp)
            y, nominal = valid[TARGET].to_numpy(), valid["nominal"].to_numpy()
            for s in self.seeds:
                atur = cara(latih, lp, s) if cara else {}
                lt, fitur = atur.get("latih", latih), atur.get("fitur", self.fitur)
                _, x_valid = matriks_lgbm(lt, valid, fitur)
                model = latih_lgbm(lt, fitur, s, atur.get("bobot"), atur.get("param"))
                prob = model.predict_proba(x_valid)[:, 1]
                varian = {nama: prob}
                if "koreksi" in atur:
                    varian = {f"{nama} (tanpa koreksi)": prob, nama: atur["koreksi"](prob)}
                for n, p in varian.items():
                    tandai = atur["tandai"](p, nominal) if "tandai" in atur else None
                    baris.append({"cara": n, "lipatan": lp.nomor, "seed": s, "n_latih": len(lt),
                                  **metrik_lengkap(y, p, nominal, tandai)})
                if s == self.seeds[0]:
                    self.prob[(nama, lp.nomor)] = pd.Series(varian[nama], index=valid.index)
        print(f"  dinilai: {nama}")
        self._hasil[kunci] = pd.DataFrame(baris)
        return self._hasil[kunci].copy()


def ringkas_cara(hasil, acuan, awalan):
    """Ringkasan per cara: PR-AUC dan selisih berpasangan terhadap `acuan`, kejujuran probabilitas, biaya."""
    skor_acuan = hasil[hasil["cara"] == acuan].set_index(["lipatan", "seed"])["pr_auc"]
    hasil = hasil.copy()
    hasil["selisih"] = hasil["pr_auc"] - hasil.set_index(["lipatan", "seed"]).index.map(skor_acuan)
    hasil.to_csv(HASIL_DIR / f"{awalan}_per_lipatan.csv", index=False)
    ringkas = hasil.groupby("cara", sort=False).agg(
        pr_auc=("pr_auc", "mean"), selisih=("selisih", "mean"),
        n_naik=("selisih", lambda s: int((s > 0).sum())), n_pasangan=("selisih", "size"),
        roc_auc=("roc_auc", "mean"), log_loss=("log_loss", "mean"), brier=("brier", "mean"),
        rasio_rata2=("rasio_rata2", "mean"), n_tandai=("n_tandai", "mean"), biaya_juta=("biaya_juta", "mean"),
    )
    per_lip = hasil.groupby(["cara", "lipatan"], sort=False)["selisih"].mean().unstack()
    ringkas["lipatan_naik"] = (per_lip > 0).sum(axis=1)
    per_lip.columns = [f"selisih_lipatan{c}" for c in per_lip.columns]
    ringkas = ringkas.join(per_lip)
    ringkas.to_csv(HASIL_DIR / f"{awalan}_ringkas.csv")
    with pd.option_context("display.width", 250):
        print(f"\nRingkasan (rata-rata {hasil['lipatan'].nunique()} lipatan x {hasil['seed'].nunique()} seed; "
              f"selisih = PR-AUC - PR-AUC '{acuan}', berpasangan):")
        print(ringkas.to_string(float_format=FMT))
    print("Catatan: lipatan yang sama dipakai untuk membandingkan dan memilih cara, jadi skor ini hasil\n"
          "pengembangan (cenderung sedikit optimistis), bukan evaluasi akhir yang independen.")
    return ringkas


def kelas_tidak_seimbang(eks, lipatan):
    print(f"\n=== 12. PENANGANAN KELAS TIDAK SEIMBANG (LightGBM {len(eks.fitur)} fitur, "
          f"{len(lipatan)} lipatan x {len(eks.seeds)} seed) ===")

    def bobot_kelas(latih, lp, s):
        # setiap fraud dihitung seperti (jumlah sah / jumlah fraud) transaksi, +-145 kali
        return {"param": {"scale_pos_weight": rasio_sah_fraud(latih[TARGET])}}

    def undersampling(latih, lp, s):
        posisi, beta = sampel_kurang(latih[TARGET].to_numpy(), RASIO_UNDERSAMPLING,
                                     np.random.default_rng(1000 * s + lp.nomor))
        return {"latih": latih.iloc[posisi], "koreksi": lambda p: koreksi_sampel_kurang(p, beta)}

    def peka_biaya(latih, lp, s):
        # fraud diberi bobot nominal/25.000 - 1, transaksi sah 1 -> skor > 0,5 = layak ditandai
        return {"bobot": bobot_nominal(latih[TARGET], latih["nominal"]), "tandai": lambda p, n: p > 0.5}

    cara = {"dasar": None, "bobot kelas": bobot_kelas,
            f"undersampling 1:{RASIO_UNDERSAMPLING}": undersampling, "bobot nominal": peka_biaya}
    hasil = pd.concat([eks.jalankan(n, lipatan, c) for n, c in cara.items()])
    ringkas_cara(hasil, "dasar", "t2_imbalance")

    # Pembanding biaya tanpa model (rata-rata per lipatan validasi, juta rupiah)
    pemb = []
    for lp in lipatan:
        v = eks.dm.loc[lp.idx_valid]
        pemb.append({"lipatan": lp.nomor, "tidak_tandai": v.loc[v[TARGET] == 1, "nominal"].sum() / 1e6,
                     "tandai_semua": BIAYA_TANDAI * len(v) / 1e6})
    pemb = pd.DataFrame(pemb)
    pemb.to_csv(HASIL_DIR / "t2_imbalance_pembanding_biaya.csv", index=False)
    print("\nPembanding biaya tanpa model per lipatan (juta Rp):")
    print(pemb.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print(f"  rata-rata: tidak menandai apa pun {pemb['tidak_tandai'].mean():,.1f}; "
          f"menandai semua {pemb['tandai_semua'].mean():,.1f}")
    print("\nBaca hasilnya: PR-AUC menilai urutan skor; log-loss, Brier, dan rasio_rata2 menilai kejujuran\n"
          "probabilitas (rasio_rata2 = rata-rata prob / tingkat fraud; 1 = jujur). Biaya memakai aturan\n"
          "prob x nominal > 25.000 dengan probabilitas MENTAH (belum dikalibrasi; kalibrasi di PR 5);\n"
          "'bobot nominal' memakai aturannya sendiri (skor > 0,5).")


def pergeseran_pola(eks, lipatan):
    print(f"\n=== 13. PERGESERAN POLA ===")
    per_kolom = adversarial(eks)
    # Kolom yang sebarannya sendiri bergeser (bukan kolom kumulatif yang memang naik seiring waktu)
    bergeser = per_kolom.loc[(per_kolom["auc_satu_kolom"] > 0.6)
                             & ~per_kolom["fitur"].isin(KOLOM_KUMULATIF), "fitur"].tolist()
    print(f"\nKolom bergeser (AUC satu kolom > 0,6, selain kolom kumulatif): {bergeser or 'tidak ada'}")

    # ---- 13b-d. Bobot data baru, awal data latih, dan panjang jeda ----
    print("\n--- 13b-d. Bobot data baru, awal data latih, dan panjang jeda (validasi sama) ---")

    def peluruhan(paruh):
        return lambda latih, lp, s: {"bobot": bobot_waktu(latih["waktu"].to_numpy(),
                                                          lp.batas_latih.to_datetime64(), paruh)}

    def mulai(tanggal):
        return lambda latih, lp, s: {"latih": latih[latih["waktu"] >= tanggal]}

    def geser(hari):
        return lambda latih, lp, s: {"latih": latih[latih["waktu"] >= lp.batas_latih - pd.Timedelta(days=hari)]}

    cara = {"dasar": None}
    cara |= {f"bobot waktu, paruh {h} hari": peluruhan(h) for h in (30, 60, 90, 180)}
    cara |= {"latih mulai 1 Feb": mulai(pd.Timestamp("2025-02-01")),
             "latih mulai 1 Mar": mulai(pd.Timestamp("2025-03-01")),
             "jendela geser 91 hari": geser(91)}
    if bergeser:
        sisa = [k for k in eks.fitur if k not in bergeser]
        cara[f"buang kolom bergeser ({', '.join(bergeser)})"] = lambda latih, lp, s: {"fitur": sisa}
    hasil = [eks.jalankan(n, lipatan, c) for n, c in cara.items()]
    # Panjang jeda: data validasi tetap, hanya batas latih yang mundur/maju (label fitur ikut batas latih).
    # Jeda 7 dan 14 hari HIPOTETIS: dengan label yang baru matang +-21 hari, model nyata tidak bisa
    # dilatih sedekat itu. Angkanya hanya mengukur seberapa cepat model menua, bukan pilihan yang bisa dipakai.
    for j in (7, 14, 35):
        nama = f"jeda {j} hari" + (" (hipotetis)" if j < 21 else "")
        hasil.append(eks.jalankan(nama, buat_lipatan_waktu(eks.dm, jeda_hari=j)))
    ringkas_cara(pd.concat(hasil), "dasar", "t2_pergeseran")
    print("Baca hasilnya: 'dasar' = bobot rata, latih sejak 1 Jan, jeda 21 hari. Bobot waktu memberi\n"
          "transaksi lama bobot lebih kecil; 'latih mulai' membuang baris awal dari data latih (tetap\n"
          "dipakai sebagai riwayat fitur). Jeda 7 dan 14 hari HANYA sensitivitas hipotetis: menurut asumsi\n"
          "label matang 21 hari, label yang lebih baru belum tersedia saat model dipakai. Selisihnya\n"
          "mengukur seberapa cepat model menua (data uji berjarak 21-82 hari dari label terakhir).")

    diagnosis_minggu_terakhir(eks, lipatan[-1])
    kematangan_label(eks, lipatan[-1])


def adversarial(eks):
    """13a. Adversarial validation: seberapa mudah model membedakan baris lama dari baris uji."""
    print("\n--- 13a. Adversarial validation (tanpa label fraud) ---")
    gab = eks.gab
    data = pd.concat([gab, eks.label(BATAS_LABEL_MATANG)], axis=1)
    uji = data[data["sumber"] == "uji"]
    lama = data[data["sumber"] == "latih"]
    pembanding = {"Jan-Agt vs Sep-Okt": lama, "Jul-Agt vs Sep-Okt": lama[lama["waktu"] >= "2025-07-01"]}
    tanpa_kum = [k for k in eks.fitur if k not in KOLOM_KUMULATIF]
    susunan = {f"{len(eks.fitur)} fitur terpilih": eks.fitur, "tanpa kolom kumulatif": tanpa_kum,
               "tanpa kumulatif & tingkat_fraud_merchant": [k for k in tanpa_kum if k != "tingkat_fraud_merchant"]}
    baris, penting = [], {}
    for np_, lama_ in pembanding.items():
        d = pd.concat([lama_, uji])
        z = (d["sumber"] == "uji").astype(int).to_numpy()
        d = d.assign(**{TARGET: z})
        for ns, f in susunan.items():
            oof = np.zeros(len(d))
            gain = pd.Series(0.0, index=f)
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=SEED).split(d, z):
                model = latih_lgbm(d.iloc[tr], f, SEED, param={"n_estimators": 200})
                _, x_te = matriks_lgbm(d.iloc[tr], d.iloc[te], f)
                oof[te] = model.predict_proba(x_te)[:, 1]
                g = model.booster_.feature_importance(importance_type="gain")
                gain += g / g.sum() / 3
            baris.append({"pembanding": np_, "susunan": ns, "n_fitur": len(f), "auc": roc_auc_score(z, oof)})
            penting[(np_, ns)] = gain.sort_values(ascending=False)
    tabel = pd.DataFrame(baris)
    tabel.to_csv(HASIL_DIR / "t2_adversarial.csv", index=False)
    print(tabel.to_string(index=False, float_format=FMT))
    pd.DataFrame({f"{a} | {b}": s for (a, b), s in penting.items()}).to_csv(
        HASIL_DIR / "t2_adversarial_pentingnya.csv", index_label="fitur")
    for kunci in [("Jul-Agt vs Sep-Okt", s) for s in susunan]:
        print(f"\n10 fitur paling membedakan ({kunci[0]}, {kunci[1]}; porsi gain):")
        print(penting[kunci].head(10).to_string(float_format=FMT))

    # Pergeseran per kolom (Jul-Agt vs Sep-Okt): AUC satu kolom, 0,5 = sebaran sama
    lama_ = pembanding["Jul-Agt vs Sep-Okt"]
    d = pd.concat([lama_, uji])
    z = (d["sumber"] == "uji").astype(int).to_numpy()
    satu = []
    for k in eks.fitur:
        if k in KOLOM_KATEGORI:
            continue
        x = d[k].astype(float)
        auc = roc_auc_score(z, x.fillna(x.min() - 1))
        satu.append({"fitur": k, "auc_satu_kolom": max(auc, 1 - auc),
                     "median_jul_agt": lama_[k].median(), "median_sep_okt": uji[k].median()})
    satu = pd.DataFrame(satu).sort_values("auc_satu_kolom", ascending=False)
    satu.to_csv(HASIL_DIR / "t2_adversarial_per_kolom.csv", index=False)
    print("\n12 kolom dengan pergeseran terbesar (Jul-Agt vs Sep-Okt; AUC satu kolom, 0,5 = tidak bergeser):")
    print(satu.head(12).to_string(index=False, float_format=FMT))
    print("Baca hasilnya: AUC +-0,5 = data uji mirip data lama. AUC tinggi karena kolom kumulatif sudah\n"
          "diduga (nilainya naik seiring waktu). tingkat_fraud_merchant membeku di data uji (label terakhir\n"
          "11 Agt), sedangkan di Jul-Agt masih berubah; bersama kolom profil merchant yang konstan (sidik\n"
          "merchant) model bisa mengenali periode walau sebaran satu kolomnya hampir sama (AUC satu kolom).")
    return satu


def diagnosis_minggu_terakhir(eks, lp):
    """13e. Kenapa minggu terakhir lipatan 4 (4-10 Agt) sulit? Gambaran per minggu validasi."""
    print(f"\n--- 13e. Diagnosis per minggu, lipatan {lp.nomor} (model dasar, seed {eks.seeds[0]}) ---")
    _, valid = eks.data_lipatan(lp)
    valid = valid.assign(prob=eks.prob[("dasar", lp.nomor)],
                         minggu=(valid["waktu"] - lp.mulai_valid).dt.days // 7 + 1)
    baris = []
    for m, v in valid.groupby("minggu"):
        f = v[v[TARGET] == 1]
        atas = v["prob"] >= v["prob"].quantile(0.99)
        baris.append({
            "minggu": f"{(lp.mulai_valid + pd.Timedelta(weeks=m - 1)).date()}", "transaksi": len(v),
            "fraud": len(f), "tingkat": v[TARGET].mean(),
            "pr_auc": average_precision_score(v[TARGET], v["prob"]),
            "rasio_prob": v["prob"].mean() / v[TARGET].mean(),
            "tangkap_top1persen": f.index.isin(v.index[atas]).mean(),
            "median_nominal_fraud": f["nominal"].median(),
            "fraud_tanpa_riwayat": f["tanpa_riwayat"].mean(), "fraud_perangkat_baru": f["perangkat_baru"].mean(),
            "fraud_merchant_lt50trx": (f["jumlah_trx_merchant_sebelum"] < 50).mean(),
        })
    tabel = pd.DataFrame(baris)
    tabel.to_csv(HASIL_DIR / "t2_minggu_lipatan4.csv", index=False)
    with pd.option_context("display.width", 250):
        print(tabel.to_string(index=False, float_format=FMT))
    kanal = (valid[valid[TARGET] == 1].groupby("minggu")["kanal"].value_counts(normalize=True)
             .unstack().fillna(0))
    kanal.to_csv(HASIL_DIR / "t2_minggu_lipatan4_kanal.csv")
    print("\nPorsi fraud per kanal tiap minggu:")
    print(kanal.to_string(float_format=FMT))


def kematangan_label(eks, lp):
    """13f. Uji kepekaan batas label: model yang dilatih sebelum lp.batas_latih menilai minggu-minggu
    sesudahnya, termasuk 11-31 Agt yang tidak pernah dipakai. Bila label minggu akhir belum matang,
    fraud yang teramati jauh di bawah fraud yang diharapkan model, padahal skor model tetap stabil."""
    print(f"\n--- 13f. Kematangan label: fraud teramati vs diharapkan (latih < {lp.batas_latih.date()}) ---")
    data = pd.concat([eks.gab, eks.label(lp.batas_latih)], axis=1)
    data = data[data["sumber"] == "latih"]
    latih = data[data["waktu"] < lp.batas_latih]
    nilai = data[data["waktu"] >= lp.mulai_valid]
    _, x_nilai = matriks_lgbm(latih, nilai, eks.fitur)
    prob = np.mean([latih_lgbm(latih, eks.fitur, s).predict_proba(x_nilai)[:, 1] for s in eks.seeds], axis=0)
    nilai = nilai.assign(prob=prob, minggu=nilai["waktu"].dt.to_period("W-SUN").dt.start_time)
    tabel = nilai.groupby("minggu").agg(transaksi=(TARGET, "size"), fraud_teramati=(TARGET, "sum"),
                                        fraud_diharapkan=("prob", "sum"))
    tabel["rasio"] = tabel["fraud_teramati"] / tabel["fraud_diharapkan"]
    acuan = tabel.loc[tabel.index < BATAS_LABEL_MATANG, "rasio"].mean()
    tabel["rasio_relatif"] = tabel["rasio"] / acuan
    tabel["presisi_top100"] = pd.Series({m: v.nlargest(100, "prob")[TARGET].mean()
                                         for m, v in nilai.groupby("minggu")})
    tabel.to_csv(HASIL_DIR / "t2_kematangan_label.csv")
    print(tabel.to_string(float_format=FMT))
    print(f"rasio_relatif = rasio minggu itu / rata-rata rasio minggu sebelum {BATAS_LABEL_MATANG.date()} "
          f"({acuan:.3f}); 1 = sesuai harapan.")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(tabel.index, tabel["fraud_diharapkan"], marker="o", label="diharapkan model (jumlah prob)")
    ax.plot(tabel.index, tabel["fraud_teramati"], marker="s", label="teramati (label is_fraud)")
    ax.axvline(BATAS_LABEL_MATANG, color="grey", ls="--", lw=1)
    ax.text(BATAS_LABEL_MATANG, ax.get_ylim()[1] * 0.95, " batas label matang", fontsize=8, va="top")
    ax.set_ylabel("jumlah fraud per minggu")
    ax.set_title(f"Fraud teramati vs diharapkan (model dilatih < {lp.batas_latih.date()})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HASIL_DIR / "t2_kematangan_label.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 14. Kalibrasi probabilitas dan aturan biaya (PR 5)
# --------------------------------------------------------------------------
PANJANG_KALIBRASI_HARI = 28
ATURAN_SOAL = "prob x nominal > 25.000"
# Urutan kesederhanaan untuk aturan pemilihan (kiri = lebih sederhana)
URUTAN_MODEL = ["dasar", f"undersampling 1:{RASIO_UNDERSAMPLING}"]
URUTAN_PENDEKATAN = ["b: latih ulang", "a: latih < kalibrasi"]
URUTAN_KALIBRASI = ["tanpa", "Platt", "isotonik"]


def _prob_kandidat(model, latih, sasaran, fitur, seed, nomor):
    """Melatih kandidat model pada `latih`, lalu probabilitas (skala asli) untuk setiap data di `sasaran`."""
    koreksi = None
    if model.startswith("undersampling"):
        posisi, beta = sampel_kurang(latih[TARGET].to_numpy(), RASIO_UNDERSAMPLING,
                                     np.random.default_rng(1000 * seed + nomor))
        latih = latih.iloc[posisi]
        koreksi = beta
    m = latih_lgbm(latih, fitur, seed)
    hasil = []
    for data in sasaran:
        _, x = matriks_lgbm(latih, data, fitur)
        p = m.predict_proba(x)[:, 1]
        hasil.append(koreksi_sampel_kurang(p, koreksi) if koreksi is not None else p)
    return hasil


def kalibrasi_biaya(eks, lipatan):
    """Kalibrasi pada periode terpisah, lalu biaya aturan prob x nominal > 25.000 dan pembandingnya.

    Setiap lipatan: latih | kalibrasi 4 minggu | jeda 3 minggu | validasi 4 minggu (validasi sama
    dengan bagian 5-13). Dua pendekatan:
      a: model dilatih sebelum periode kalibrasi; pengalibrasi dipelajari dari skornya di periode itu.
      b: pengalibrasi dari (a) dipasang pada model yang dilatih ulang sampai awal jeda, TERMASUK data
         periode kalibrasi (lebih segar, sama dengan model di bagian 12-13; sedikit tidak cocok dengan
         pengalibrasinya).
    Pada kedua pendekatan, pengalibrasi dipelajari dari skor model (a) yang belum pernah melihat periode
    kalibrasi. Pada (b), periode kalibrasi terpisah dari model yang dipakai mempelajari pengalibrasi,
    tetapi tidak dari model akhir.
    """
    print(f"\n=== 14. KALIBRASI PROBABILITAS DAN ATURAN BIAYA ({len(lipatan)} lipatan x {len(eks.seeds)} seed) ===")
    lip_kal = buat_lipatan_waktu(eks.dm, panjang_kalibrasi_hari=PANJANG_KALIBRASI_HARI)
    tabel_lip = ringkas_lipatan(eks.dm, lip_kal)
    tabel_lip.to_csv(HASIL_DIR / "t2_kalibrasi_lipatan.csv", index=False)
    print(tabel_lip[["lipatan", "latih_sampai_sebelum", "jeda_mulai", "valid_mulai", "latih_baris",
                     "kalibrasi_baris", "valid_fraud"]].to_string(index=False))

    baris, platt, rel = [], [], []
    for lk, lp in zip(lip_kal, lipatan):
        assert np.array_equal(lk.idx_valid, lp.idx_valid), "data validasi harus sama"
        data_a = pd.concat([eks.dm, eks.label(lk.batas_latih).loc[eks.dm.index]], axis=1)
        latih_a, kal, valid_a = (data_a.loc[lk.idx_latih], data_a.loc[lk.idx_kalibrasi],
                                 data_a.loc[lk.idx_valid])
        latih_b, valid_b = eks.data_lipatan(lp)
        y, nom = valid_a[TARGET].to_numpy(), valid_a["nominal"].to_numpy()
        yk, nomk = kal[TARGET].to_numpy(), kal["nominal"].to_numpy()
        dasar_baris = {"lipatan": lp.nomor}
        for aturan, tandai in (("tidak menandai apa pun", np.zeros(len(y), bool)),
                               ("menandai semua", np.ones(len(y), bool))):
            for s in eks.seeds:
                baris.append({**dasar_baris, "seed": s, "model": "-", "pendekatan": "-", "kalibrasi": "-",
                              "aturan": aturan, "n_tandai": int(tandai.sum()), "biaya_juta": biaya(y, nom, tandai) / 1e6})
        for s in eks.seeds:
            for model in URUTAN_MODEL:
                p_kal, p_va = _prob_kandidat(model, latih_a, [kal, valid_a], eks.fitur, s, lp.nomor)
                (p_vb,) = _prob_kandidat(model, latih_b, [valid_b], eks.fitur, s, lp.nomor)
                f_platt, (a, b) = kalibrasi_platt(p_kal, yk)
                pengubah = {"tanpa": lambda p: p, "Platt": f_platt, "isotonik": kalibrasi_isotonik(p_kal, yk)}
                k_kal, _ = pilih_ambang(p_kal, nomk, yk)
                platt.append({"lipatan": lp.nomor, "seed": s, "model": model, "a": a, "b": b,
                              "k_dari_kalibrasi": k_kal,
                              "rasio_rata2_kalibrasi": p_kal.mean() / yk.mean()})
                for pend, p_v in ((URUTAN_PENDEKATAN[1], p_va), (URUTAN_PENDEKATAN[0], p_vb)):
                    umum = {**dasar_baris, "seed": s, "model": model, "pendekatan": pend}
                    for kn, f in pengubah.items():
                        q = f(p_v)
                        baris.append({**umum, "kalibrasi": kn, "aturan": ATURAN_SOAL, **metrik_lengkap(y, q, nom)})
                        if s == eks.seeds[0] and model == URUTAN_MODEL[0] and pend == URUTAN_PENDEKATAN[0]:
                            rel.append(pd.DataFrame({"kalibrasi": kn, "lipatan": lp.nomor, "y": y, "p": q}))
                    baris.append({**umum, "kalibrasi": "tanpa", "aturan": "prob x nominal > k (k dari periode kalibrasi)",
                                  "k": k_kal, **metrik_lengkap(y, p_v, nom, p_v * nom > k_kal)})
                    k_or, _ = pilih_ambang(p_v, nom, y)
                    baris.append({**umum, "kalibrasi": "tanpa", "aturan": "ambang terbaik di validasi (tidak tercapai)",
                                  "k": k_or, **metrik_lengkap(y, p_v, nom, p_v * nom > k_or)})
        print(f"  lipatan {lp.nomor} selesai")

    hasil = pd.DataFrame(baris)
    hasil.to_csv(HASIL_DIR / "t2_kalibrasi_per_lipatan.csv", index=False)
    platt = pd.DataFrame(platt)
    platt.to_csv(HASIL_DIR / "t2_kalibrasi_platt.csv", index=False)

    # ---- Ringkasan: biaya berpasangan terhadap acuan "dasar | b | tanpa | aturan soal" ----
    kunci = ["model", "pendekatan", "kalibrasi", "aturan"]
    acuan = (URUTAN_MODEL[0], URUTAN_PENDEKATAN[0], "tanpa", ATURAN_SOAL)
    def biaya_kombinasi(nilai):
        """Biaya per (lipatan, seed) untuk satu kombinasi model | pendekatan | kalibrasi | aturan."""
        pilih = np.logical_and.reduce([hasil[k] == v for k, v in zip(kunci, nilai)])
        return hasil[pilih].set_index(["lipatan", "seed"])["biaya_juta"]

    biaya_acuan = biaya_kombinasi(acuan)
    hasil["selisih_biaya"] = hasil["biaya_juta"] - hasil.set_index(["lipatan", "seed"]).index.map(biaya_acuan)
    ringkas = hasil.groupby(kunci, sort=False).agg(
        biaya_juta=("biaya_juta", "mean"), selisih_biaya=("selisih_biaya", "mean"),
        galat_baku=("selisih_biaya", lambda s: s.std() / np.sqrt(len(s))),
        n_tandai=("n_tandai", "mean"), pr_auc=("pr_auc", "mean"), log_loss=("log_loss", "mean"),
        brier=("brier", "mean"), ece=("ece", "mean"), rasio_rata2=("rasio_rata2", "mean"),
    ).reset_index()
    ringkas.to_csv(HASIL_DIR / "t2_kalibrasi_ringkas.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 48):
        print(f"\nRingkasan (rata-rata per lipatan validasi; biaya dalam juta Rp; selisih terhadap "
              f"'{' | '.join(acuan)}', berpasangan):")
        print(ringkas.to_string(index=False, float_format=FMT))

    # ---- Aturan pemilihan (ditetapkan sebelum melihat hasil) ----
    # Hanya kombinasi yang memakai aturan soal (tandai bila prob x nominal > 25.000 pada kolom prob yang
    # dikumpulkan), supaya prob dan tandai konsisten. Pilih biaya rata-rata terendah; kombinasi yang selisih
    # biayanya terhadap yang termurah < 1 galat baku berpasangan dianggap setara, lalu dipilih yang paling
    # sederhana (urutan: model, pendekatan, kalibrasi seperti URUTAN_*).
    calon = ringkas[ringkas["aturan"] == ATURAN_SOAL].copy()
    termurah = calon.loc[calon["biaya_juta"].idxmin()]
    b_termurah = biaya_kombinasi(termurah[kunci].tolist())
    setara = []
    for _, r in calon.iterrows():
        b = biaya_kombinasi(r[kunci].tolist())
        d = b - b_termurah
        if d.mean() <= d.std() / np.sqrt(len(d)):
            setara.append(r)
    setara = pd.DataFrame(setara)
    setara["urut"] = (setara["model"].map(URUTAN_MODEL.index) * 100 + setara["pendekatan"].map(URUTAN_PENDEKATAN.index) * 10
                      + setara["kalibrasi"].map(URUTAN_KALIBRASI.index))
    pilihan = setara.sort_values("urut").iloc[0]

    def nama(r):
        return f"{r['model']} | {r['pendekatan']} | {r['kalibrasi']}"
    print(f"\nTermurah: {nama(termurah)} (biaya {termurah['biaya_juta']:.2f} jt)")
    print("Setara (< 1 galat baku dari termurah): " + "; ".join(nama(r) for _, r in setara.iterrows()))
    print(f"PILIHAN (paling sederhana di antara yang setara): {nama(pilihan)} "
          f"-> biaya {pilihan['biaya_juta']:.2f} jt, PR-AUC {pilihan['pr_auc']:.4f}, rasio_rata2 {pilihan['rasio_rata2']:.3f}")
    pd.DataFrame([pilihan[kunci + ["biaya_juta", "pr_auc", "rasio_rata2"]]]).to_csv(
        HASIL_DIR / "t2_kalibrasi_pilihan.csv", index=False)

    # ---- Koefisien Platt dan ambang dari periode kalibrasi ----
    print("\nPlatt logit(q) = a x logit(p) + b dan ambang k dari periode kalibrasi (rata-rata seed):")
    print(platt.groupby(["model", "lipatan"])[["a", "b", "k_dari_kalibrasi", "rasio_rata2_kalibrasi"]]
          .mean().to_string(float_format=FMT))

    # ---- Diagram reliabilitas (model dasar, pendekatan b, seed pertama, 4 lipatan digabung) ----
    rel = pd.concat(rel)
    tabel_rel = pd.concat([pd.DataFrame(tabel_reliabilitas(g["y"], g["p"])).assign(kalibrasi=kn)
                           for kn, g in rel.groupby("kalibrasi", sort=False)])
    tabel_rel.to_csv(HASIL_DIR / "t2_reliabilitas.csv", index=False)
    print("\nDiagram reliabilitas (dasar, pendekatan b, seed pertama; janji = prob_rata2, kenyataan = tingkat_fraud):")
    print(tabel_rel.to_string(index=False, float_format=FMT))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for kn, g in tabel_rel.groupby("kalibrasi", sort=False):
        g = g[g["transaksi"] >= 50]
        ax.plot(g["prob_rata2"], g["tingkat_fraud"].clip(lower=1e-5), marker="o", label=kn)
    ax.plot([1e-5, 1], [1e-5, 1], color="grey", ls="--", lw=1, label="sempurna")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvspan(0.005, 0.3, color="orange", alpha=0.08, label="wilayah keputusan")
    ax.set_xlabel("probabilitas yang diberikan model (rata-rata kelompok)")
    ax.set_ylabel("tingkat fraud sebenarnya")
    ax.set_title("Diagram reliabilitas (validasi, 4 lipatan)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HASIL_DIR / "t2_reliabilitas.png", dpi=150)
    plt.close(fig)
    print("\nBaca hasilnya: aturan prob x nominal > 25.000 berasal dari perbandingan biaya pasti menandai\n"
          "(Rp 25.000) dengan biaya harapan tidak menandai (prob x nominal); aturan ini hanya optimal bila prob\n"
          "jujur. 'Ambang terbaik di validasi' memakai label validasi sendiri, jadi hanya batas bawah biaya.\n"
          "Skor ini hasil pengembangan pada lipatan yang juga dipakai memilih, bukan evaluasi akhir independen.")


# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Task 2 (klasifikasi fraud)")
    parser.add_argument("--berkas", default="klasifikasi_train.csv.gz",
                        help="nama berkas di folder data/")
    parser.add_argument("--berkas-uji", default="klasifikasi_test.csv.gz",
                        help="nama berkas data uji di folder data/ (untuk riwayat fitur)")
    parser.add_argument("--tahap", choices=["audit", "validasi", "fitur", "imbalance", "shift", "kalibrasi",
                                            "semua"],
                        default="semua")
    args = parser.parse_args(argv)

    HASIL_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    df = muat_data(args.berkas)
    df_bersih = bersihkan_kategori(df)
    if args.tahap in ("audit", "semua"):
        audit_kualitas(df)
        audit_tambahan(df)
        uji_titik_waktu(df)
        fraud_per_minggu(df)
        uji_kebocoran(df, df_bersih)
    if args.tahap in ("validasi", "semua"):
        validasi(df_bersih)
    if args.tahap in ("fitur", "imbalance", "shift", "kalibrasi", "semua"):
        gab, dm, lipatan, dasar = siapkan_data(df, muat_data(args.berkas_uji))
        if args.tahap in ("fitur", "semua"):
            evaluasi_fitur(gab, dm, lipatan, dasar)
        if args.tahap in ("imbalance", "shift", "kalibrasi", "semua"):
            eks = Eksperimen(gab, dm, susun_fitur(dasar, KELOMPOK_TERPILIH),
                             [SEED + k for k in range(N_SEED_PR4)])
            if args.tahap in ("imbalance", "semua"):
                kelas_tidak_seimbang(eks, lipatan)
            if args.tahap in ("shift", "semua"):
                pergeseran_pola(eks, lipatan)
            if args.tahap in ("kalibrasi", "semua"):
                kalibrasi_biaya(eks, lipatan)
    print(f"\nTabel dan gambar tersimpan di: {HASIL_DIR}")


if __name__ == "__main__":
    main()
