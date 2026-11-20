"""Task 3 bagian e-g: stabilitas, profil dan aksi segmen, anomali -1, dan cluster.csv final.

Memakai preprocessing bagian a-d dari kode/task3_clustering.py (fungsi `siapkan`):
pelanggan tanpa transaksi = segmen 0; clustering hanya pada pelanggan aktif.

  e. Stabilitas: ulangi 20-30 kali (subsampel 80%, latih ulang, prediksi seluruh pelanggan aktif), ARI antar-run.
     - per k (2-10) dan representasi (tanpa / dengan komposisi kategori) -> keputusan k dan blok kategori,
     - per metode pada k terpilih: K-Means, MiniBatch K-Means, Gaussian Mixture, HDBSCAN + KNN
       (run yang gagal membentuk >= 2 cluster dicatat terpisah, tidak dihitung sebagai partisi).
  f. Profil tiap segmen: ukuran, median fitur utama, kategori dan jam belanja dominan, nama, dan satu aksi.
  g. Anomali: pelanggan aktif dengan jarak ke centroid di atas persentil 98 diberi label -1 (±2%).

Cara menjalankan dari akar repo (data di data/clustering.csv.gz):
    python kode/task3_lanjutan.py
Keluaran: kode/hasil/task3_eg/ dan cluster.csv di akar repo (id_pelanggan,cluster).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

import task3_clustering as t3

ROOT = t3.ROOT
SEED = t3.SEED
HASIL = ROOT / "kode/hasil/task3_eg"

K_GRID = range(2, 11)
N_POOL = 20_000          # pool tetap untuk grid k (hemat waktu); prediksi tetap ke SEMUA pelanggan aktif
N_RUN_GRID = 20          # ulangan per (representasi, k)
N_RUN_METODE = 30        # ulangan per metode pada k terpilih (subsampel 80% dari semua pelanggan aktif)
N_RUN_HDBSCAN = 20
N_HDBSCAN = 6_000        # HDBSCAN tidak sanggup 90 ribu baris; dilatih pada 6.000 lalu disebar dengan KNN
PORSI_SUBSAMPEL = 0.8
AMBANG_STABIL = 0.80     # panduan: ARI > 0,8 berarti stabil
PERSENTIL_ANOMALI = 98   # g: ±2% pelanggan aktif dengan jarak terjauh ke centroid -> -1
# Blok kategori dipakai hanya bila stabilitas naik >= 0,01 ARI (bermakna secara praktis) DAN > 2 galat baku jackknife.
# Ambang praktis ditambahkan setelah review PR #11: tanpa ambang ini, selisih ±0,001-0,002 (keduanya ARI ±0,99)
# membuat keputusan berbalik tergantung jumlah thread komputer.
AMBANG_PRAKTIS_KATEGORI = 0.01
# Jumlah thread dikunci (sama dengan bagian a-d) supaya hasil K-Means sama di mesin mana pun dengan >= 4 core.
N_THREAD = 4

FITUR_PROFIL = ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "item_per_pesanan",
                "persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod", "retur_90h",
                "sesi_per_minggu", "rasio_konversi", "lama_member_bulan", "umur"]


# --------------------------------------------------------------------------
# Fungsi bantu (diuji di kode/tests/test_task3_lanjutan.py)
# --------------------------------------------------------------------------
def ari_antar_run(label_runs: list[np.ndarray]) -> np.ndarray:
    """ARI untuk setiap pasangan run (nomor cluster boleh tertukar; yang dinilai keanggotaannya)."""
    n = len(label_runs)
    return np.array([adjusted_rand_score(label_runs[i], label_runs[j]) for i in range(n) for j in range(i + 1, n)])


def ari_matriks(label_runs: list[np.ndarray]) -> np.ndarray:
    """Matriks ARI simetris antar-run (diagonal NaN); dipakai untuk jackknife."""
    n = len(label_runs)
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = adjusted_rand_score(label_runs[i], label_runs[j])
    return m


def _rata2_pasangan(m: np.ndarray, buang: int | None = None) -> float:
    pilih = np.ones(len(m), dtype=bool)
    if buang is not None:
        pilih[buang] = False
    sub = m[np.ix_(pilih, pilih)]
    return float(np.nanmean(sub[np.triu_indices(pilih.sum(), 1)]))


def jackknife_selisih(m_a: np.ndarray, m_b: np.ndarray) -> tuple[float, float]:
    """Selisih rata-rata ARI (a - b) dan galat bakunya dengan jackknife leave-one-run-out berpasangan.

    Pasangan ARI saling berbagi run, jadi simpangan baku biasa tidak menghasilkan galat baku yang benar.
    Run ke-r di kedua representasi memakai subsampel yang sama, sehingga run dibuang berpasangan.
    """
    n = len(m_a)
    selisih = _rata2_pasangan(m_a) - _rata2_pasangan(m_b)
    tanpa = np.array([_rata2_pasangan(m_a, r) - _rata2_pasangan(m_b, r) for r in range(n)])
    galat = float(np.sqrt((n - 1) / n * np.sum((tanpa - tanpa.mean()) ** 2)))
    return float(selisih), galat


def urutkan_label(label: np.ndarray, nilai: np.ndarray) -> np.ndarray:
    """Memberi nomor ulang 1..k menurut median `nilai` dari besar ke kecil (1 = median terbesar).

    Nomor cluster K-Means bergantung pada seed; pengurutan membuat nomor dan nama segmen tetap.
    """
    unik = np.unique(label)
    median = {c: np.median(nilai[label == c]) for c in unik}
    peta = {c: i + 1 for i, c in enumerate(sorted(unik, key=lambda c: -median[c]))}
    return np.vectorize(peta.get)(label)


def tandai_anomali(jarak: np.ndarray, persentil: float = PERSENTIL_ANOMALI) -> np.ndarray:
    """True untuk jarak di atas persentil tertentu (jarak terjauh ke centroid clusternya)."""
    return jarak > np.percentile(jarak, persentil)


def pilih_k(tabel: pd.DataFrame, ambang: float = AMBANG_STABIL) -> int:
    """Aturan pemilihan k (ditetapkan sebelum melihat hasil final).

    Hanya k yang stabil (rata-rata ARI >= ambang). Setiap kriteria internal memilih k terbaiknya
    (silhouette dan Calinski-Harabasz tertinggi, Davies-Bouldin terendah); k dengan suara terbanyak
    dipilih, bila seri dipilih yang paling stabil.
    """
    stabil = tabel[tabel["ari_rata2"] >= ambang]
    if stabil.empty:
        return int(tabel.sort_values("ari_rata2", ascending=False).iloc[0]["k"])
    suara = pd.Series(0, index=stabil["k"].to_numpy())
    suara[stabil.loc[stabil["silhouette"].idxmax(), "k"]] += 1
    suara[stabil.loc[stabil["calinski_harabasz"].idxmax(), "k"]] += 1
    suara[stabil.loc[stabil["davies_bouldin"].idxmin(), "k"]] += 1
    calon = suara[suara == suara.max()].index
    terpilih = stabil[stabil["k"].isin(calon)].sort_values("ari_rata2", ascending=False)
    return int(terpilih.iloc[0]["k"])


# --------------------------------------------------------------------------
# e. Stabilitas
# --------------------------------------------------------------------------
def _kmeans(x, k, seed):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)


def stabilitas_grid(rep: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    """Per representasi dan k: kriteria internal (pada pool) dan stabilitas ARI (prediksi semua pelanggan aktif).

    Mengembalikan tabel ringkas dan matriks ARI antar-run per (representasi, k) untuk jackknife.
    Run ke-r memakai subsampel dan seed yang sama di kedua representasi (berpasangan).
    """
    n = len(next(iter(rep.values())))
    rng = np.random.default_rng(SEED)
    pool = np.sort(rng.choice(n, N_POOL, replace=False))
    skor = np.sort(rng.choice(N_POOL, 5_000, replace=False))
    baris, matriks = [], {}
    for nama, x in rep.items():
        xp = x[pool]
        for k in K_GRID:
            lab = _kmeans(xp, k, SEED).labels_
            runs = []
            for r in range(N_RUN_GRID):
                rows = np.sort(np.random.default_rng(SEED + 1000 + r).choice(N_POOL, int(PORSI_SUBSAMPEL * N_POOL),
                                                                             replace=False))
                runs.append(_kmeans(xp[rows], k, SEED + 1000 + r).predict(x))
            m = ari_matriks(runs)
            matriks[(nama, k)] = m
            ari = m[np.triu_indices(len(m), 1)]
            baris.append({"representasi": nama, "k": k, "silhouette": silhouette_score(xp[skor], lab[skor]),
                          "davies_bouldin": davies_bouldin_score(xp, lab),
                          "calinski_harabasz": calinski_harabasz_score(xp, lab),
                          "ari_rata2": ari.mean(), "ari_sb": ari.std(), "ari_min": ari.min(),
                          "porsi_cluster_terkecil": np.bincount(lab).min() / len(lab)})
            print(f"  {nama:16s} k={k:2d} ARI {ari.mean():.3f} ± {ari.std():.3f}", flush=True)
    return pd.DataFrame(baris), matriks


def stabilitas_metode(x: np.ndarray, k: int) -> pd.DataFrame:
    """Stabilitas tiap metode pada k terpilih: latih pada subsampel 80% semua pelanggan aktif, prediksi semua."""
    n = len(x)
    hasil = {}

    def subsampel(r, ukuran):
        return np.sort(np.random.default_rng(SEED + 2000 + r).choice(n, ukuran, replace=False))

    pembuat = {
        "KMeans": lambda xs, s: KMeans(n_clusters=k, n_init=10, random_state=s).fit(xs),
        "MiniBatchKMeans": lambda xs, s: MiniBatchKMeans(n_clusters=k, n_init=10, batch_size=1024,
                                                         random_state=s).fit(xs),
        "GaussianMixture_diag": lambda xs, s: GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                                                              reg_covar=1e-5, random_state=s).fit(xs),
    }
    for nama, buat in pembuat.items():
        runs = [buat(x[rows], SEED + 2000 + r).predict(x)
                for r, rows in ((r, subsampel(r, int(PORSI_SUBSAMPEL * n))) for r in range(N_RUN_METODE))]
        hasil[nama] = (runs, None)
        print(f"  {nama} selesai", flush=True)
    # HDBSCAN + KNN: HDBSCAN dilatih pada 6.000 pelanggan (tidak sanggup 90 ribu baris dan tidak punya predict),
    # lalu pelanggan lain diberi label 15 tetangga terdekat di antara titik non-noise. Jadi stabilitasnya mencakup
    # penyebaran KNN, dan noise HDBSCAN tidak ikut disebar. Run yang tidak menghasilkan >= 2 cluster = GAGAL
    # (dicatat terpisah, tidak dihitung sebagai partisi).
    runs, noise = [], []
    for r in range(N_RUN_HDBSCAN):
        rows = subsampel(r, N_HDBSCAN)
        lab = HDBSCAN(min_cluster_size=50, min_samples=5, copy=True).fit_predict(x[rows])
        noise.append((lab < 0).mean())
        inti = lab >= 0
        if inti.sum() < 15 or len(np.unique(lab[inti])) < 2:
            runs.append(None)
            continue
        runs.append(KNeighborsClassifier(n_neighbors=15).fit(x[rows][inti], lab[inti]).predict(x))
    hasil["HDBSCAN + KNN (min_cluster_size=50, min_samples=5)"] = (runs, np.array(noise))
    print("  HDBSCAN + KNN selesai", flush=True)
    return pd.DataFrame([ringkas_stabilitas(nama, runs, noise) for nama, (runs, noise) in hasil.items()])


def ringkas_stabilitas(nama: str, runs: list, noise: np.ndarray | None = None) -> dict:
    """Ringkasan stabilitas satu metode. Run bernilai None atau dengan < 2 cluster = gagal (tidak dihitung).

    ARI dihitung hanya dari run yang valid; bila run valid kurang dari dua, ARI tidak tersedia (NaN).
    """
    valid = [r for r in runs if r is not None and len(np.unique(r)) >= 2]
    ari = ari_antar_run(valid) if len(valid) >= 2 else np.array([])
    noise = np.full(len(runs), np.nan) if noise is None else np.asarray(noise, dtype=float)
    return {"metode": nama, "n_run": len(runs), "n_run_valid": len(valid), "n_run_gagal": len(runs) - len(valid),
            "ari_rata2": ari.mean() if ari.size else np.nan, "ari_sb": ari.std() if ari.size else np.nan,
            "ari_min": ari.min() if ari.size else np.nan,
            "jumlah_cluster_median": float(np.median([len(np.unique(r)) for r in valid])) if valid else np.nan,
            "noise_persen_rata2": 100 * np.nanmean(noise) if np.isfinite(noise).any() else np.nan,
            "stabil": bool(ari.size and ari.mean() > AMBANG_STABIL)}


# --------------------------------------------------------------------------
# f. Profil dan aksi
# --------------------------------------------------------------------------
NAMA_AKSI = {
    0: ("Belum Bertransaksi",
        "Kirim kupon pembelian pertama (misalnya potongan Rp 20 ribu tanpa minimum) lewat notifikasi aplikasi."),
    1: ("Pelanggan Premium Loyal",
        "Program VIP: gratis ongkir tanpa minimum dan akses awal produk baru, bukan diskon besar."),
    2: ("Pemburu Promo Malam",
        "Jadwalkan flash sale dan voucher terbatas pukul 20.00-23.00, dikirim lewat push notification."),
    -1: ("Anomali (perlu ditinjau)",
         "Tidak diikutkan kampanye massal; tinjau kualitas data atau kemungkinan akun tidak wajar."),
}


def profil_segmen(data: pd.DataFrame, cluster: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabel profil per segmen dan tabel porsi kategori (rata-rata porsi dan lift terhadap semua pelanggan aktif)."""
    d = data.assign(cluster=cluster)
    kat = [c for c in data if c.startswith("porsi_") and not c.startswith("porsi_jam_")]
    jam = [c for c in data if c.startswith("porsi_jam_")]
    aktif = d["frekuensi_90h"] > 0
    rerata_kat = d.loc[aktif, kat].mean()
    baris, porsi = [], []
    for c, g in d.groupby("cluster"):
        r = {"cluster": int(c), "nama": NAMA_AKSI.get(int(c), ("?", ""))[0], "pelanggan": len(g),
             "persen": 100 * len(g) / len(d)}
        r |= {f"median_{f}": g[f].median() for f in FITUR_PROFIL}
        rata_kat = g[kat].mean()
        lift = rata_kat / rerata_kat
        r["kategori_dominan"] = rata_kat.idxmax().removeprefix("porsi_")
        r["kategori_khas_lift"] = ", ".join(f"{k.removeprefix('porsi_')} ({v:.2f}x)"
                                            for k, v in lift.sort_values(ascending=False).head(3).items())
        r["jam_dominan"] = g[jam].mean().idxmax().removeprefix("porsi_jam_")
        r["porsi_jam_malam"] = g["porsi_jam_malam"].mean()
        r["kota_tier_modus"] = g["kota_tier"].mode().iat[0]
        r["perangkat_modus"] = g["perangkat"].mode().iat[0]
        r["aksi"] = NAMA_AKSI.get(int(c), ("", "?"))[1]
        baris.append(r)
        porsi.append(pd.DataFrame({"cluster": int(c), "kategori": [k.removeprefix("porsi_") for k in kat],
                                   "porsi_rata2": rata_kat.to_numpy(), "lift": lift.to_numpy()}))
    return pd.DataFrame(baris), pd.concat(porsi, ignore_index=True)


def periksa_nama(profil: pd.DataFrame) -> list[str]:
    """Memastikan nama segmen sesuai datanya; daftar kosong = semua sesuai."""
    p = profil.set_index("cluster")
    masalah = []
    if set(p.index) - set(NAMA_AKSI):
        masalah.append(f"segmen tanpa nama: {sorted(set(p.index) - set(NAMA_AKSI))}")
        return masalah
    aktif = p.loc[[c for c in p.index if c > 0]]
    if 0 in p.index and p.loc[0, "median_frekuensi_90h"] != 0:
        masalah.append("segmen 0 seharusnya tanpa transaksi")
    if len(aktif) == 2:
        if aktif["median_belanja_90h"].idxmax() != 1:
            masalah.append("segmen 1 bukan yang belanjanya terbesar")
        if aktif["median_persen_voucher"].idxmax() != 2 or aktif["porsi_jam_malam"].idxmax() != 2:
            masalah.append("segmen 2 bukan yang paling banyak memakai voucher dan berbelanja malam")
    return masalah


# --------------------------------------------------------------------------
def run(data_dir: Path | None = None) -> dict:
    """Menjalankan bagian e-g dengan jumlah thread dikunci (N_THREAD) supaya hasil sama di mesin lain."""
    with threadpool_limits(limits=N_THREAD):
        return _run(data_dir)


def _run(data_dir: Path | None = None) -> dict:
    mulai = time.perf_counter()
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    HASIL.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_dir / "clustering.csv.gz")
    active, x, blocks, _, _ = t3.siapkan(data)
    rep = {"utama": x, "utama+kategori": np.column_stack([x, blocks["komposisi_kategori"]]).astype("float32")}
    print(f"Pelanggan: {len(data):,}; tanpa transaksi (segmen 0): {len(data) - len(active):,}; aktif: {len(active):,}")

    # ---- e1. Grid k x representasi ----
    print("\n=== e. STABILITAS: grid k dan representasi (K-Means, 20 subsampel 80% dari pool 20.000) ===")
    grid, matriks = stabilitas_grid(rep)
    grid.to_csv(HASIL / "stabilitas_per_k.csv", index=False)
    pilihan = {nama: pilih_k(g.reset_index(drop=True)) for nama, g in grid.groupby("representasi", sort=False)}
    # Blok kategori dimasukkan hanya bila menaikkan stabilitas pada k terpilih lebih dari 2 galat baku.
    # Galat baku selisih dihitung dengan jackknife leave-one-run-out berpasangan (pasangan ARI saling berbagi run,
    # dan run ke-r memakai subsampel yang sama di kedua representasi). Panduan: fitur yang menurunkan stabilitas
    # atau hanya menambah noise dikeluarkan.
    k = pilihan["utama"]
    g_u = grid[(grid.representasi == "utama") & (grid.k == k)].iloc[0]
    g_k = grid[(grid.representasi == "utama+kategori") & (grid.k == k)].iloc[0]
    selisih, galat = jackknife_selisih(matriks[("utama+kategori", k)], matriks[("utama", k)])
    pakai_kategori = bool(selisih >= AMBANG_PRAKTIS_KATEGORI and selisih > 2 * galat)
    nama_rep = "utama+kategori" if pakai_kategori else "utama"
    if pakai_kategori:
        k = pilihan["utama+kategori"]
    xf = rep[nama_rep]
    lab_u = _kmeans(rep["utama"], k, SEED).predict(rep["utama"])
    lab_k = _kmeans(rep["utama+kategori"], k, SEED).predict(rep["utama+kategori"])
    ari_rep = adjusted_rand_score(lab_u, lab_k)
    keputusan_kat = pd.DataFrame([{"k": k, "ari_tanpa_kategori": g_u.ari_rata2, "ari_dengan_kategori": g_k.ari_rata2,
                                   "selisih_dengan_minus_tanpa": selisih, "galat_baku_jackknife": galat,
                                   "silhouette_tanpa": g_u.silhouette, "silhouette_dengan": g_k.silhouette,
                                   "ari_partisi_tanpa_vs_dengan": ari_rep, "pakai_kategori": pakai_kategori}])
    keputusan_kat.to_csv(HASIL / "keputusan_kategori.csv", index=False)
    with pd.option_context("display.width", 200):
        print(grid.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"k terpilih per representasi: {pilihan}; representasi final: {nama_rep}, k = {k}")
    print(f"Kategori: ARI {g_u.ari_rata2:.4f} (tanpa) vs {g_k.ari_rata2:.4f} (dengan); selisih {selisih:+.5f}, "
          f"galat baku jackknife {galat:.5f}; partisi tanpa vs dengan kategori ARI {ari_rep:.3f} -> "
          f"{'dipakai' if pakai_kategori else f'tidak dipakai (perlu naik >= {AMBANG_PRAKTIS_KATEGORI} dan > 2 galat baku)'}")

    # ---- e2. Stabilitas per metode pada k terpilih ----
    print(f"\n=== e. STABILITAS PER METODE (k = {k}, {N_RUN_METODE} subsampel 80% dari semua pelanggan aktif) ===")
    metode = stabilitas_metode(xf, k)
    metode.to_csv(HASIL / "stabilitas_metode.csv", index=False)
    print(metode.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- Model final + g. anomali ----
    model = _kmeans(xf, k, SEED)
    lab = model.predict(xf)
    jarak = np.linalg.norm(xf - model.cluster_centers_[lab], axis=1)
    lab = urutkan_label(lab, active["belanja_90h"].to_numpy())
    anomali = tandai_anomali(jarak)
    cluster = np.zeros(len(data), dtype=int)                 # 0 = belum bertransaksi
    cluster[active.index.to_numpy()] = np.where(anomali, -1, lab)
    rk = (data.loc[active.index, "rasio_konversi"] > 1).to_numpy()
    gmm = GaussianMixture(n_components=k, covariance_type="diag", n_init=3, reg_covar=1e-5,
                          random_state=SEED).fit(xf)
    ll = gmm.score_samples(xf)
    gmm_rendah = ll < np.percentile(ll, 100 - PERSENTIL_ANOMALI)
    ringkas_anomali = pd.DataFrame([{
        "anomali": int(anomali.sum()), "persen_semua_pelanggan": 100 * anomali.sum() / len(data),
        "persen_pelanggan_aktif": 100 * anomali.mean(), "ambang_jarak": float(np.percentile(jarak, PERSENTIL_ANOMALI)),
        "irisan_dengan_loglik_gmm_terendah": float((anomali & gmm_rendah).sum() / anomali.sum()),
        "rasio_konversi_di_atas_1_aktif": int(rk.sum()),
        "rasio_konversi_di_atas_1_yang_anomali": int((rk & anomali).sum()),
    }])
    ringkas_anomali.to_csv(HASIL / "anomali_ringkas.csv", index=False)
    print("\n=== g. ANOMALI (jarak ke centroid > persentil 98 pelanggan aktif) ===")
    print(ringkas_anomali.T.to_string(header=False, float_format=lambda v: f"{v:,.3f}"))

    # ---- f. Profil dan aksi ----
    profil, porsi = profil_segmen(data, cluster)
    profil.to_csv(HASIL / "profil_segmen.csv", index=False)
    porsi.to_csv(HASIL / "profil_kategori.csv", index=False)
    print("\n=== f. PROFIL DAN AKSI ===")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(profil.set_index("cluster").T.to_string())
    masalah = periksa_nama(profil)
    if masalah:
        raise SystemExit("Nama segmen tidak sesuai data, tinjau NAMA_AKSI: " + "; ".join(masalah))

    jawaban = pd.DataFrame({"id_pelanggan": data["id_pelanggan"], "cluster": cluster})
    # Pembanding: kandidat sementara bagian a-d (MiniBatch K-Means k=2 pada sampel 6.000), dihitung ulang dari kode
    # a-d, bukan dibaca dari cluster.csv (berkas itu ditimpa di bawah, jadi run ulang akan membandingkan dengan dirinya).
    rng = np.random.default_rng(SEED)
    idx_ad = np.sort(rng.choice(len(active), t3.N_FIT, replace=False))
    sementara = np.zeros(len(data), dtype=int)
    sementara[active.index.to_numpy()] = t3._kmeans(x[idx_ad], 2).predict(x).astype(int) + 1
    ari_vs_sementara = float(adjusted_rand_score(sementara, cluster))
    print(f"\nARI cluster.csv final vs kandidat sementara a-d (dihitung ulang): {ari_vs_sementara:.3f}")
    jawaban.to_csv(ROOT / "cluster.csv", index=False)

    # Grafik: stabilitas dan silhouette per k
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for nama, g in grid.groupby("representasi", sort=False):
        axes[0].errorbar(g.k, g.ari_rata2, yerr=g.ari_sb, marker="o", capsize=3, label=nama)
        axes[1].plot(g.k, g.silhouette, marker="o", label=nama)
    axes[0].axhline(AMBANG_STABIL, color="grey", ls="--", lw=1)
    axes[0].set(xlabel="k (pelanggan aktif)", ylabel="ARI antar-run", title="Stabilitas (20 subsampel 80%)")
    axes[1].set(xlabel="k (pelanggan aktif)", ylabel="silhouette", title="Silhouette")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HASIL / "stabilitas_per_k.png", dpi=150)
    plt.close(fig)

    meta = {"seed": SEED, "representasi": nama_rep, "blok": list(t3.BLOCKS_TERPILIH) + (["komposisi_kategori"] if pakai_kategori else []),
            "metode": "KMeans (n_init=10)", "n_thread": N_THREAD, "k_aktif": k, "persentil_anomali": PERSENTIL_ANOMALI,
            "ukuran_segmen": {str(c): int(n) for c, n in pd.Series(cluster).value_counts().sort_index().items()},
            "ari_vs_cluster_sementara_ad": ari_vs_sementara,
            "durasi_menit": round((time.perf_counter() - mulai) / 60, 1)}
    (HASIL / "keputusan_eg.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\ncluster.csv ditulis: {ROOT / 'cluster.csv'}")
    print(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 bagian e-g")
    parser.add_argument("--data-dir", type=Path, default=None)
    run(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
