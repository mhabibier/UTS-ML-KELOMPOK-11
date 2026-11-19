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
    """Aturan keputusan soal: tandai bila prob x nominal > biaya_tandai.

    Asal aturan: untuk satu transaksi dengan peluang fraud p dan nominal N,
      - menandai   -> biaya pasti 25.000 (fraud yang ditandai tertahan tanpa kerugian),
      - tidak      -> biaya harapan p x N (rugi N bila ternyata fraud, 0 bila sah).
    Tandai bila p x N > 25.000. Aturan ini hanya meminimalkan biaya bila p adalah peluang sebenarnya
    (terkalibrasi); bila p terlalu kecil, fraud bernilai besar ikut lolos.
    Contoh: N = Rp 500.000 -> ditandai bila p > 0,05; N = Rp 5 juta -> bila p > 0,005.
    """
    return np.asarray(prob) * np.asarray(nominal, dtype=float) > biaya_tandai


# --------------------------------------------------------------------------
# Kalibrasi probabilitas (PR 5)
# --------------------------------------------------------------------------
# Tepi kelompok untuk diagram reliabilitas dan ECE. Probabilitas fraud hampir selalu kecil, jadi
# kelompok dibuat rapat di bawah (skala log); wilayah 0,005-0,3 adalah wilayah keputusan
# (ambang 25.000 / nominal untuk nominal Rp 80 ribu - Rp 5 juta).
TEPI_RELIABILITAS = np.array([0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def kalibrasi_platt(p_kal, y_kal):
    """Platt: regresi logistik logit(q) = a x logit(p) + b, dipelajari di periode kalibrasi.

    Mengembalikan (fungsi pengubah, (a, b)). Urutan skor tidak berubah (a > 0), jadi PR-AUC tetap.
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=np.inf, max_iter=1000)   # tanpa regularisasi
    model.fit(_logit(p_kal).reshape(-1, 1), np.asarray(y_kal))
    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    return (lambda p: 1 / (1 + np.exp(-(a * _logit(p) + b)))), (a, b)


def kalibrasi_isotonik(p_kal, y_kal):
    """Isotonik: fungsi tangga naik bebas bentuk, dipelajari di periode kalibrasi.

    Tangga membuat banyak skor bernilai sama dan bisa menurunkan PR-AUC; karena itu dicampur sedikit
    dengan skor asli, q = (1 - 1e-6) x tangga + 1e-6 x p, supaya urutan asli sebagian besar terjaga.
    Keterbatasan: untuk skor sangat dekat 1 (selisih antar-skor < 1e-10) presisi angka komputer tidak
    cukup membedakannya, jadi sebagian skor teratas tetap seri dan PR-AUC bisa turun sedikit.
    """
    from sklearn.isotonic import IsotonicRegression
    model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    model.fit(np.asarray(p_kal, dtype=float), np.asarray(y_kal, dtype=float))
    eps = 1e-6
    return lambda p: (1 - eps) * model.predict(np.asarray(p, dtype=float)) + eps * np.asarray(p, dtype=float)


def pilih_ambang(prob, nominal, y, grid=None, biaya_tandai=BIAYA_TANDAI):
    """Ambang k yang meminimalkan biaya untuk aturan prob x nominal > k (dicari pada grid log).

    Dipakai sebagai pembanding kalibrasi: bila dipilih di periode kalibrasi, ini cara langsung
    menyetel keputusan tanpa memperbaiki probabilitas. Mengembalikan (k terbaik, biaya pada k itu).
    """
    grid = np.geomspace(1e3, 1e7, 161) if grid is None else grid
    skor = np.asarray(prob, dtype=float) * np.asarray(nominal, dtype=float)
    hasil = [biaya(y, nominal, skor > k, biaya_tandai) for k in grid]
    i = int(np.argmin(hasil))
    return float(grid[i]), float(hasil[i])


def ece(y, prob, tepi=TEPI_RELIABILITAS):
    """Expected calibration error: rata-rata berbobot |rata-rata prob - tingkat fraud| per kelompok."""
    y, prob = np.asarray(y, dtype=float), np.asarray(prob, dtype=float)
    kelompok = np.clip(np.digitize(prob, tepi[1:-1]), 0, len(tepi) - 2)
    total = 0.0
    for g in np.unique(kelompok):
        pilih = kelompok == g
        total += pilih.mean() * abs(prob[pilih].mean() - y[pilih].mean())
    return total


def tabel_reliabilitas(y, prob, tepi=TEPI_RELIABILITAS):
    """Per kelompok probabilitas: jumlah transaksi, rata-rata prob (janji), dan tingkat fraud (kenyataan)."""
    y, prob = np.asarray(y, dtype=float), np.asarray(prob, dtype=float)
    kelompok = np.clip(np.digitize(prob, tepi[1:-1]), 0, len(tepi) - 2)
    baris = []
    for g in np.unique(kelompok):
        pilih = kelompok == g
        baris.append({"kelompok": f"{tepi[g]:g}-{tepi[g + 1]:g}", "transaksi": int(pilih.sum()),
                      "fraud": int(y[pilih].sum()), "prob_rata2": prob[pilih].mean(),
                      "tingkat_fraud": y[pilih].mean()})
    return baris


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
        "ece": ece(y, prob),
        "n_tandai": int(np.sum(tandai)),
        "biaya_juta": biaya(y, nominal, tandai) / 1e6,
    }
