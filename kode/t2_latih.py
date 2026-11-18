"""Task 2 - fungsi bantu pelatihan (PR 4): kelas tidak seimbang, bobot waktu, dan biaya.

Semua fungsi di sini murni (tanpa membaca data) sehingga bisa diuji di kode/tests/test_t2_latih.py.
"""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

# Biaya menandai satu transaksi (dari soal). Fraud yang ditandai dianggap tertahan tanpa kerugian;
# fraud yang lolos merugi sebesar nominalnya.
BIAYA_TANDAI = 25_000


# --------------------------------------------------------------------------
# Kelas tidak seimbang
# --------------------------------------------------------------------------
def rasio_sah_fraud(y):
    """Jumlah transaksi sah dibagi jumlah fraud (dipakai sebagai scale_pos_weight LightGBM)."""
    y = np.asarray(y)
    n_fraud = y.sum()
    return (len(y) - n_fraud) / n_fraud


def sampel_kurang(y, rasio, rng):
    """Undersampling: semua fraud + transaksi sah acak sebanyak `rasio` x jumlah fraud.

    Mengembalikan (posisi baris terpilih, beta), dengan beta = porsi transaksi sah yang dipertahankan.
    beta diperlukan untuk mengoreksi probabilitas (lihat koreksi_sampel_kurang).
    """
    y = np.asarray(y)
    fraud, sah = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_ambil = min(len(sah), int(round(rasio * len(fraud))))
    ambil = rng.choice(sah, n_ambil, replace=False)
    return np.sort(np.concatenate([fraud, ambil])), n_ambil / len(sah)


def koreksi_sampel_kurang(p, beta):
    """Mengembalikan probabilitas model yang dilatih pada data undersampling ke skala data asli.

    Membuang transaksi sah dengan porsi beta membuat odds fraud naik 1/beta kali, jadi
    odds asli = beta x odds sampel  ->  p = beta p_s / (beta p_s - p_s + 1)   (Dal Pozzolo dkk., 2015).
    Urutan skor tidak berubah, sehingga PR-AUC sama; yang berubah hanya skala probabilitasnya.
    """
    p = np.asarray(p, dtype=float)
    return beta * p / (beta * p - p + 1)


def bobot_nominal(y, nominal, biaya_tandai=BIAYA_TANDAI):
    """Bobot peka biaya: transaksi sah = 1, fraud = max(nominal / biaya_tandai - 1, 0).

    Alasannya: salah menandai transaksi sah merugikan biaya_tandai, sedangkan membiarkan fraud lolos
    merugikan nominal - biaya_tandai lebih banyak daripada menandainya. Model yang dilatih dengan bobot
    ini menghasilkan skor q dengan q > 0,5 tepat bila p x nominal > biaya_tandai (aturan keputusan soal),
    asalkan q terkalibrasi. Fraud bernominal <= biaya_tandai tidak berharga untuk ditandai (bobot 0).
    """
    y, nominal = np.asarray(y), np.asarray(nominal, dtype=float)
    return np.where(y == 1, np.maximum(nominal / biaya_tandai - 1, 0.0), 1.0)


# --------------------------------------------------------------------------
# Pergeseran pola
# --------------------------------------------------------------------------
def bobot_waktu(waktu, acuan, paruh_hari):
    """Bobot peluruhan eksponensial: 1 pada `acuan`, 0,5 pada paruh_hari sebelumnya, 0,25 pada dua kali itu."""
    umur = (acuan - waktu) / np.timedelta64(1, "D")
    return np.power(0.5, np.asarray(umur, dtype=float) / paruh_hari)


# --------------------------------------------------------------------------
# Metrik dan biaya
# --------------------------------------------------------------------------
def biaya(y, nominal, tandai, biaya_tandai=BIAYA_TANDAI):
    """Biaya total = nominal fraud yang lolos + biaya_tandai x jumlah transaksi yang ditandai."""
    y, nominal, tandai = np.asarray(y), np.asarray(nominal, dtype=float), np.asarray(tandai, dtype=bool)
    return float(nominal[(y == 1) & ~tandai].sum() + biaya_tandai * tandai.sum())


def tandai_aturan(prob, nominal, biaya_tandai=BIAYA_TANDAI):
    """Aturan keputusan soal: tandai bila prob x nominal > biaya_tandai."""
    return np.asarray(prob) * np.asarray(nominal, dtype=float) > biaya_tandai


def metrik_lengkap(y, prob, nominal, tandai=None):
    """PR-AUC, ROC-AUC, kejujuran probabilitas (log-loss, Brier, rasio rata-rata), dan biaya.

    rasio_rata2 = rata-rata probabilitas / tingkat fraud sebenarnya (1 = skala probabilitas jujur,
    > 1 = probabilitas membengkak). `tandai` bawaan = aturan prob x nominal > 25.000.
    """
    y, prob = np.asarray(y), np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    if tandai is None:
        tandai = tandai_aturan(prob, nominal)
    return {
        "pr_auc": average_precision_score(y, prob),
        "roc_auc": roc_auc_score(y, prob),
        "log_loss": log_loss(y, prob, labels=[0, 1]),
        "brier": brier_score_loss(y, prob),
        "rasio_rata2": prob.mean() / y.mean(),
        "n_tandai": int(np.sum(tandai)),
        "biaya_juta": biaya(y, nominal, tandai) / 1e6,
    }
