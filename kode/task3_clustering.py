"""Task 3 a-d: audit, CLR, ablation, tiga metode, dan kandidat cluster sementara.

Semua keluaran dibuat dari data tanpa label. Pilihan cluster masih perlu diuji pada e-g.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_FIT = 6000
N_SCORE = 1500
K_GRID = range(2, 13)
EPS_CLR = 1e-4
BLOCKS_TERPILIH = ("perilaku", "nilai_belanja", "demografi", "komposisi_waktu")


def clr(frame: pd.DataFrame, epsilon: float = EPS_CLR) -> np.ndarray:
    """Centered log ratio per pelanggan; nol diberi pseudocount lalu ditutup ke 1."""
    x = frame.astype(float).to_numpy(copy=True)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Komposisi harus finite dan tidak negatif")
    x += epsilon
    x /= x.sum(axis=1, keepdims=True)
    log = np.log(x)
    return log - log.mean(axis=1, keepdims=True)


def numerik(frame: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    x = frame.astype(float).replace([np.inf, -np.inf], np.nan).copy()
    rows = []
    for c in x:
        lo, hi = x[c].quantile([.01, .99])
        log = bool(x[c].min() >= 0 and x[c].skew() > 1)
        rows.append({"fitur": c, "clip_p01": lo, "clip_p99": hi,
                     "log1p": log, "missing": int(x[c].isna().sum())})
        x[c] = x[c].clip(lo, hi)
        if log:
            x[c] = np.log1p(x[c])
    z = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(x)
    return StandardScaler().fit_transform(z), rows


def bobot_blok(x: np.ndarray, weight: float = 1.) -> np.ndarray:
    x = x - x.mean(axis=0)
    norm = np.sqrt(np.mean(np.sum(x * x, axis=1)))
    return (x * np.sqrt(weight) / max(norm, 1e-12)).astype("float32")


def siapkan(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Pelanggan tanpa transaksi dipisahkan sebelum clustering pelanggan aktif."""
    inactive = data["frekuensi_90h"].eq(0)
    if not inactive.equals(data["rata2_keranjang"].isna()):
        raise ValueError("Definisi pelanggan tanpa transaksi berubah; audit dahulu")
    active = data.loc[~inactive].copy()
    cat = [c for c in data if c.startswith("porsi_") and not c.startswith("porsi_jam_")]
    times = [c for c in data if c.startswith("porsi_jam_")]
    crm = [c for c in data if c[0] == "n" and c[1:].isdigit()]
    monetary = ["belanja_90h", "rata2_keranjang", "item_per_pesanan", "belanja_30h"]
    demo_num = ["lama_member_bulan", "umur"]
    demo_cat = ["kota_tier", "perangkat"]
    numeric = active.select_dtypes("number").columns.tolist()
    behavior = [c for c in numeric if c not in set(cat + times + crm + monetary + demo_num)]
    z_behavior, rows_b = numerik(active[behavior])
    z_money, rows_m = numerik(active[monetary])
    z_demo, rows_d = numerik(active[demo_num])
    z_cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
        active[demo_cat].astype("string").fillna("__missing__"))
    # Korelasi tinggi pada fitur numerik; data tanpa label, jadi ini diagnostik fitur saja.
    all_num = pd.DataFrame(np.column_stack([z_behavior, z_money]),
                           columns=behavior + monetary, index=active.index)
    corr = all_num.corr().abs()
    keep, drop = [], {}
    for c in all_num:
        prior = next((k for k in keep if corr.loc[c, k] > .9), None)
        if prior:
            drop[c] = prior
        else:
            keep.append(c)
    behavior_kept = [c for c in behavior if c in keep]
    money_kept = [c for c in monetary if c in keep]
    blocks = {
        "perilaku": bobot_blok(all_num[behavior_kept].to_numpy()),
        "nilai_belanja": bobot_blok(all_num[money_kept].to_numpy()),
        "demografi": bobot_blok(np.column_stack([z_demo, z_cat]), .25),
        "komposisi_kategori": bobot_blok(clr(active[cat])),
        "komposisi_waktu": bobot_blok(clr(active[times])),
    }
    # Pada ablation awal, menghapus komposisi kategori menaikkan silhouette dan
    # stabilitas; blok tetap disimpan untuk eksperimen sensitivitas.
    x = np.column_stack([blocks[n] for n in BLOCKS_TERPILIH]).astype("float32")
    if not np.isfinite(x).all():
        raise ValueError("Preprocessing menghasilkan nilai nonfinite")
    params = pd.DataFrame(rows_b + rows_m + rows_d)
    keputusan = pd.DataFrame([{"fitur": c, "keputusan":
                              "ID" if c == "id_pelanggan" else
                              "CRM anonim (tidak dipakai)" if c in crm else
                              f"redundan dengan {drop[c]}" if c in drop else "dipakai"}
                             for c in data.columns])
    return active, x, blocks, keputusan, params


def _kmeans(sample: np.ndarray, k: int, seed: int = SEED) -> MiniBatchKMeans:
    return MiniBatchKMeans(n_clusters=k, n_init=10, batch_size=1024, random_state=seed).fit(sample)


def _gmm(sample: np.ndarray, k: int) -> GaussianMixture:
    return GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                           max_iter=300, reg_covar=1e-5, random_state=SEED).fit(sample)


def _score(sample: np.ndarray, labels: np.ndarray, score_idx: np.ndarray) -> tuple[float, float]:
    subset = labels[score_idx]
    if len(np.unique(subset)) < 2 or len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return float(silhouette_score(sample[score_idx], subset)), float(davies_bouldin_score(sample, labels))


def run(data_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    path = data_dir / "clustering.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Taruh clustering.csv.gz di {ROOT / 'data'} atau beri --data-dir: {path}")
    out = Path(out_dir) if out_dir else ROOT / "kode/hasil/task3_ad_revisi"
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(path)
    if data.id_pelanggan.isna().any() or not data.id_pelanggan.is_unique:
        raise ValueError("ID pelanggan kosong atau duplikat")
    active, x, blocks, keputusan, params = siapkan(data)
    keputusan.to_csv(out / "keputusan_fitur.csv", index=False)
    params.to_csv(out / "parameter_preprocessing.csv", index=False)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(active), N_FIT, replace=False))
    score_idx = np.sort(rng.choice(N_FIT, N_SCORE, replace=False))
    sample = x[idx]
    results, models = [], {}
    with threadpool_limits(limits=4):
        for k in K_GRID:
            km = _kmeans(sample, k)
            gm = _gmm(sample, k)
            for name, model in (("MiniBatchKMeans", km), ("GaussianMixture_diag", gm)):
                labels = model.predict(sample)
                sil, db = _score(sample, labels, score_idx)
                results.append({"metode": name, "k": k, "silhouette": sil,
                                "davies_bouldin": db, "bic": gm.bic(sample) if name.startswith("Gaussian") else np.nan,
                                "cluster_terkecil": int(np.bincount(labels).min())})
                models[(name, k)] = model
        for min_samples in (2, 5, 10):
            for min_size in (20, 50):
                hb = HDBSCAN(min_cluster_size=min_size, min_samples=min_samples, copy=True).fit(sample)
                labels = hb.labels_
                core = labels >= 0
                if core.sum() > N_SCORE and len(np.unique(labels[core])) > 1:
                    ids = np.flatnonzero(core)
                    si = np.sort(rng.choice(ids, N_SCORE, replace=False))
                    sil, db = _score(sample[core], labels[core], np.searchsorted(np.flatnonzero(core), si))
                else:
                    sil, db = np.nan, np.nan
                results.append({"metode": "HDBSCAN", "k": len(set(labels) - {-1}),
                                "min_cluster_size": min_size, "min_samples": min_samples,
                                "silhouette": sil, "davies_bouldin": db, "noise_persen": 100 * (~core).mean(),
                                "cluster_terkecil": int(np.bincount(labels[core]).min()) if core.any() else 0})
    metrics = pd.DataFrame(results)
    metrics.to_csv(out / "perbandingan_metode_k.csv", index=False)
    # Pilihan sementara dari kesepakatan silhouette/DB; BIC dilaporkan terpisah.
    candidates = metrics[(metrics.metode != "HDBSCAN") & (metrics.cluster_terkecil > 0)].copy()
    candidates["rank_sil"] = candidates.silhouette.rank(ascending=False)
    candidates["rank_db"] = candidates.davies_bouldin.rank()
    candidates["rank_gabungan"] = candidates.rank_sil + candidates.rank_db
    chosen = candidates.sort_values(["rank_gabungan", "k", "metode"]).iloc[0]
    method, k = str(chosen.metode), int(chosen.k)
    model = models[(method, k)]
    ablation = []
    variants = [("tidak_ada", list(BLOCKS_TERPILIH))]
    variants += [(n, [b for b in BLOCKS_TERPILIH if b != n]) for n in BLOCKS_TERPILIH]
    variants += [("tambah_komposisi_kategori", list(BLOCKS_TERPILIH) + ["komposisi_kategori"])]
    for excluded, names in variants:
        z = np.column_stack([blocks[n] for n in names]).astype("float32")
        ss = z[idx]
        fit = _kmeans(ss, k)
        lab = fit.predict(ss)
        sil, db = _score(ss, lab, score_idx)
        # Stabilitas eksploratif 3 subsampel 80%; uji e harus memperluas ke 20-50 run.
        runs = []
        for seed in range(3):
            rows = np.sort(np.random.default_rng(SEED + seed).choice(N_FIT, int(.8 * N_FIT), replace=False))
            runs.append(_kmeans(ss[rows], k, SEED + seed).predict(ss))
        ari = [adjusted_rand_score(runs[i], runs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        ablation.append({"perubahan": excluded, "silhouette": sil,
                         "davies_bouldin": db, "stabilitas_ari_3_run": float(np.mean(ari))})
    pd.DataFrame(ablation).to_csv(out / "ablation_kelompok_fitur.csv", index=False)
    all_labels = np.zeros(len(data), dtype=int)  # 0 = tanpa transaksi, 1..k = aktif
    all_labels[active.index.to_numpy()] = model.predict(x).astype(int) + 1
    answer = pd.DataFrame({"id_pelanggan": data.id_pelanggan, "cluster": all_labels})
    answer.to_csv(ROOT / "cluster.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, group in metrics[metrics.metode != "HDBSCAN"].groupby("metode"):
        axes[0].plot(group.k, group.silhouette, "o-", label=name)
        axes[1].plot(group.k, group.davies_bouldin, "o-", label=name)
    for ax, title in zip(axes, ["Silhouette", "Davies-Bouldin"]):
        ax.set(xlabel="k pelanggan aktif", title=title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pemilihan_k.png", dpi=150)
    plt.close(fig)
    meta = {"status": "SEMENTARA; e-g belum selesai", "seed": SEED, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pelanggan": len(data), "tanpa_transaksi": int((all_labels == 0).sum()), "aktif": len(active),
            "metode_aktif": method, "k_aktif": k, "blok_utama": BLOCKS_TERPILIH,
            "ukuran_cluster": answer.cluster.value_counts().sort_index().to_dict(),
            "bic_gmm_pilih_k": int(metrics[metrics.metode == "GaussianMixture_diag"].sort_values("bic").iloc[0].k),
            "catatan": "Silhouette/DB dipakai untuk kandidat; BIC dan HDBSCAN diagnostik. Perlu uji stabilitas 20-50 run, profil/aksi, anomali -1."}
    (out / "keputusan_ad.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.data_dir)


if __name__ == "__main__":
    main()
