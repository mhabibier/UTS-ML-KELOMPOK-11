"""Task 2 - Klasifikasi: deteksi transaksi fraud.

Tahap audit (PR 1):
  1. kualitas data (nilai hilang, ID ganda, nilai tidak wajar, kategori tidak konsisten),
  2. keterlambatan label (tingkat fraud per minggu),
  3. uji kebocoran fitur (dampak membuang kolom yang dicurigai).

Tahap validasi (PR 2):
  4. lipatan validasi berbasis waktu (latih | jeda 3 minggu | validasi 4 minggu),
  5. model dasar (tanpa kemampuan, regresi logistik, LightGBM),
  6. pembanding: validasi acak vs validasi berbasis waktu,
  7. PR-AUC per minggu setelah jeda,
  8. uji ulang kolom "dipantau" dengan beberapa seed,
  9. bukti kebocoran umur_merchant_hari (sengaja dimasukkan kembali untuk diukur).

Cara menjalankan dari akar repo:
    python kode/task2_klasifikasi.py                    # semua tahap
    python kode/task2_klasifikasi.py --tahap validasi   # hanya validasi

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

from t2_data import (BATAS_LABEL_MATANG, FITUR_BOCOR, HASIL_DIR, KOLOM_BINER,
                     KOLOM_DIPANTAU, KOLOM_ID, KOLOM_KATEGORI, SEED, TARGET,
                     bersihkan_kategori, buat_lipatan_waktu, data_matang,
                     fitur_model, muat_data, ringkas_lipatan, tambah_kolom)

# Banyaknya seed untuk mengukur derau satu kali latih: SEED, SEED+1, ..., SEED+4
N_SEED_ULANG = 5

# Pengaturan LightGBM awal (belum di-tuning). deterministic + force_row_wise:
# hasil sama persis setiap kali dijalankan dengan seed yang sama.
PARAM_LGBM = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
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
    saat transaksi: artinya tanggal daftar dicatat SETELAH transaksi (informasi dari masa depan).
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
          "berarti diambil dari satu potret (snapshot), bukan dihitung saat transaksi.")

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
def latih_lgbm(latih, fitur, seed=SEED):
    model = lgb.LGBMClassifier(random_state=seed, **PARAM_LGBM)
    model.fit(latih[fitur], latih[TARGET])
    return model


def prob_lgbm(latih, valid, fitur, seed=SEED):
    """Melatih LightGBM pada `latih` lalu mengembalikan probabilitas fraud untuk `valid`."""
    return latih_lgbm(latih, fitur, seed).predict_proba(valid[fitur])[:, 1]


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
    print("\nBaca hasilnya: bila membuang satu kolom menurunkan PR-AUC sangat tajam, "
          "kolom itu patut dicurigai sebagai fitur bocor. Periksa juga maknanya di kamus data.")


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
    print(f"Data matang (sebelum {BATAS_LABEL_MATANG.date()}): {len(dm):,} baris, "
          f"{int(dm[TARGET].sum()):,} fraud ({dm[TARGET].mean():.4%})")
    lipatan = buat_lipatan_waktu(dm)
    ringkas = ringkas_lipatan(dm, lipatan)
    ringkas.to_csv(HASIL_DIR / "t2_validasi_lipatan.csv", index=False)
    print(ringkas.to_string(index=False, float_format=FMT))
    gambar_lipatan(dm, lipatan)

    fitur = fitur_model(dm)
    print(f"Jumlah fitur: {len(fitur)} (tanpa ID, waktu, label, dan {', '.join(FITUR_BOCOR)})")

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
    print("\n=== 6. VALIDASI ACAK vs VALIDASI BERBASIS WAKTU (LightGBM, fitur sama) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acak = []
    for i, (il, iv) in enumerate(skf.split(dm, dm[TARGET]), start=1):
        latih, valid = dm.iloc[il], dm.iloc[iv]
        acak.append({"cara": "acak (5 lipatan bertingkat)", "lipatan": i,
                     **metrik(valid[TARGET], prob_lgbm(latih, valid, fitur))})
    waktu_ = [{"cara": "berbasis waktu (4 lipatan, jeda 3 minggu)", "lipatan": lp.nomor,
               **metrik(dm.loc[lp.idx_valid, TARGET], prob_lgbm_cache(lp, SEED))} for lp in lipatan]
    banding = pd.DataFrame(acak + waktu_)
    banding.to_csv(HASIL_DIR / "t2_validasi_acak_vs_waktu.csv", index=False)
    rb = banding.groupby("cara", sort=False)[["pr_auc", "roc_auc"]].agg(["mean", "std"])
    print(rb.to_string(float_format=FMT))
    print("\nValidasi acak mencampur transaksi masa depan ke data latih dan mengabaikan jeda label,\n"
          "jadi angkanya cenderung lebih optimistis daripada kinerja saat model benar-benar dipakai.")

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

    # ---- 8. Uji ulang kolom dipantau dengan beberapa seed ----
    print(f"\n=== 8. UJI ULANG KOLOM DIPANTAU ({len(lipatan)} lipatan x {N_SEED_ULANG} seed) ===")
    seeds = [SEED + k for k in range(N_SEED_ULANG)]
    baris = []
    for lp in lipatan:
        y = dm.loc[lp.idx_valid, TARGET]
        for s in seeds:
            dasar = average_precision_score(y, prob_lgbm_cache(lp, s))
            for kol in KOLOM_DIPANTAU:
                buang = average_precision_score(
                    y, prob_lgbm_cache(lp, s, [k for k in fitur if k != kol]))
                baris.append({"kolom_dibuang": kol, "lipatan": lp.nomor, "seed": s,
                              "pr_auc_dasar": dasar, "pr_auc_dibuang": buang,
                              "selisih": buang - dasar})
        print(f"  lipatan {lp.nomor} selesai")
    ulang = pd.DataFrame(baris)
    ulang.to_csv(HASIL_DIR / "t2_validasi_kolom_dipantau.csv", index=False)

    # Derau seed: simpangan baku PR-AUC model dasar antar-seed dalam lipatan yang sama
    derau_seed = (ulang.drop_duplicates(["lipatan", "seed"])
                  .groupby("lipatan")["pr_auc_dasar"].std().mean())
    ringkas_ulang = ulang.groupby("kolom_dibuang").agg(
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
          "Kolom bocor biasanya menunjukkan penurunan besar dan konsisten di semua lipatan\n"
          "(bandingkan dengan flag_review_manual). Selisih kecil yang arahnya berganti-ganti\n"
          "antarlipatan berarti kolom itu tidak terbukti bocor dari data;\n"
          "keputusan akhirnya juga harus memeriksa makna kolom di kamus data.")

    # ---- 9. Bukti kebocoran umur_merchant_hari ----
    # Kolom ini sudah dibuang dari model (FITUR_BOCOR). Di sini SENGAJA dimasukkan kembali untuk
    # mengukur sumbangannya. Prediksi yang sama (tanpa melatih ulang) dinilai dua kali: pada semua
    # baris validasi dan pada baris yang umurnya tidak negatif. Bila keunggulannya hilang setelah
    # baris negatif disisihkan, sumbangan kolom ini berasal dari baris yang mustahil (umur < 0).
    if "umur_merchant_hari" in dm.columns:
        print(f"\n=== 9. BUKTI KEBOCORAN umur_merchant_hari ({len(lipatan)} lipatan x {len(seeds)} seed) ===")
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
def main(argv=None):
    parser = argparse.ArgumentParser(description="Task 2 (klasifikasi fraud)")
    parser.add_argument("--berkas", default="klasifikasi_train.csv.gz",
                        help="nama berkas di folder data/")
    parser.add_argument("--tahap", choices=["audit", "validasi", "semua"], default="semua")
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
    print(f"\nTabel dan gambar tersimpan di: {HASIL_DIR}")


if __name__ == "__main__":
    main()
