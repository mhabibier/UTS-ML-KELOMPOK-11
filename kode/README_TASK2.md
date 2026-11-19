# Task 2 - Klasifikasi fraud

Memberi peluang fraud (`prob`) untuk setiap transaksi September-Oktober 2025 dan memutuskan transaksi
mana yang ditandai (`tandai`), dengan biaya review Rp 25.000 per transaksi yang ditandai.
Keluaran akhir: `prediksi_klasifikasi.csv` di akar repo (kolom `id_transaksi,prob,tandai`).

## Berkas

| Berkas | Isi |
| --- | --- |
| `task2_klasifikasi.py` | Skrip utama: semua tahap (bagian 1-17), menulis tabel/gambar ke `hasil/` dan `prediksi_klasifikasi.csv` |
| `t2_data.py` | Konstanta, muat data, daftar fitur, kolom yang dikecualikan, lipatan validasi berbasis waktu |
| `t2_fitur.py` | Fitur riwayat buatan sendiri yang hanya melihat masa lalu, termasuk tingkat fraud historis merchant |
| `t2_latih.py` | Undersampling + koreksi, bobot, kalibrasi Platt/isotonik, aturan biaya, metrik |
| `cek_format.py` | Cek format tiga berkas prediksi (`prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, `cluster.csv`) |
| `tests/` | Unit test: anti-kebocoran fitur, pembagian waktu, kalibrasi, aturan biaya, cek format |

## Cara menjalankan

Data asli diletakkan di folder `data/` di akar repo (`klasifikasi_train.csv.gz`, `klasifikasi_test.csv.gz`).
Semua perintah dijalankan dari akar repo.

```bash
python kode/task2_klasifikasi.py                    # semua tahap (±4,7 jam di laptop pengembang)
python kode/task2_klasifikasi.py --tahap final      # hanya model final + prediksi_klasifikasi.csv (±3 menit)
python -m pytest kode/tests -q                      # unit test (36 tes)
python kode/cek_format.py                           # cek format tiga berkas prediksi
```

Setiap tahap bisa dijalankan sendiri dengan `--tahap <nama>`. Lama tiap tahap pada run penuh terakhir
(19 Nov 2026), di luar waktu memuat data (±1-2 menit):

| `--tahap` | Bagian | Lama | Berkas utama di `hasil/` |
| --- | --- | --- | --- |
| `audit` | 1-3: kualitas data, label per minggu, uji kebocoran | 5 menit | `t2_audit_*`, `t2_fraud_per_minggu.png`, `t2_dampak_kebocoran.csv` |
| `validasi` | 4-9: lipatan waktu, model dasar, acak vs waktu, kolom dipantau | 60 menit | `t2_validasi_*` |
| `fitur` | 10-11: evaluasi dan seleksi maju kelompok fitur | 84 menit | `t2_fitur_*` |
| `imbalance` | 12: empat cara menangani kelas tidak seimbang | 28 menit | `t2_imbalance_*` |
| `shift` | 13: adversarial validation, bobot data baru, jeda, kematangan label | 95 menit | `t2_adversarial_*`, `t2_pergeseran_*`, `t2_kematangan_label.png` |
| `kalibrasi` | 14: Platt/isotonik pada periode terpisah, aturan biaya | 7 menit | `t2_kalibrasi_*`, `t2_reliabilitas.png` |
| `analisis` | 15-16: analisis error, fraud per bulan x kanal, pentingnya fitur antarperiode | 2 menit | `t2_error_*`, `t2_fraud_bulan_*`, `t2_pentingnya_antarperiode.csv` |
| `final` | 17: model final, kalibrasi, aturan biaya, cek format | 1 menit | `t2_final_*`, `../prediksi_klasifikasi.csv` |

`SEED = 42` (di `t2_data.py`). Eksperimen pembanding memakai beberapa seed (42-46) untuk mengukur derau;
model final memakai seed 42. LightGBM berjalan dengan `deterministic=True`, sehingga hasilnya sama setiap kali.

## Alur dan keputusan utama

1. **Kebocoran.** `flag_review_manual` dibuang (tinjauan manual terjadi setelah transaksi; PR-AUC turun
   0,871 -> 0,719 tanpa kolom ini). `umur_merchant_hari` (negatif di 955 transaksi) dan `volume_merchant_30h`
   (konstan per merchant sepanjang Jan-Okt) dikecualikan sebagai kehati-hatian.
2. **Label belum matang.** Tingkat fraud mingguan turun tajam mulai 11 Agt. Model yang dilatih sebelum 23 Jun
   mengharapkan jumlah fraud yang stabil, tetapi fraud teramati hanya 0,72 / 0,74 / 0,16 kali harapan pada minggu
   11 / 18 / 25 Agt. Transaksi mulai 11 Agt tidak dipakai untuk latih maupun validasi.
3. **Validasi berbasis waktu.** 4 lipatan dengan jendela latih makin panjang, jeda 21 hari, validasi 4 minggu,
   berakhir 10 Agt. Jeda 21 hari meniru jarak antara label terakhir (10 Agt) dan awal data uji (1 Sep).
4. **Fitur.** 77 fitur bawaan + 10 fitur buatan (riwayat pengguna, merchant terlihat, tingkat fraud historis
   merchant dari label yang sudah matang saat itu) = 87 fitur, dipilih dengan seleksi maju beraturan tetap.
   PR-AUC 0,719 -> 0,757.
5. **Kelas tidak seimbang.** Tanpa penanganan terbaik (PR-AUC 0,758); bobot kelas, undersampling, dan bobot
   nominal lebih rendah.
6. **Pergeseran pola.** Bobot data baru dan jendela latih pendek tidak lolos aturan pemilihan. Minggu 4-10 Agt
   memperlihatkan lonjakan fraud lewat QRIS (porsi fraud 5% -> 37%).
7. **Kalibrasi dan aturan biaya.** Model tanpa kalibrasi terlalu rendah hati (rata-rata prob 0,79 kali tingkat
   fraud). Platt pada periode kalibrasi terpisah menurunkan biaya validasi dari Rp 43,8 jt ke Rp 41,0 jt per
   4 minggu (tidak menandai apa pun: Rp 386 jt; menandai semua: Rp 1.181 jt), tanpa mengubah PR-AUC.
   Aturan: tandai bila `prob x nominal > 25.000` (biaya pasti menandai vs biaya harapan tidak menandai).
8. **Model final.** LightGBM 87 fitur dilatih pada semua transaksi sebelum 11 Agt; Platt dipelajari dari model
   yang dilatih sebelum 14 Jul dan dinilai pada 14 Jul - 10 Agt. Data uji 1 Sep - 31 Okt (106.674 transaksi),
   1.455 ditandai (1,36%).

## Keterbatasan

- Batas label 11 Agt dan jeda 21 hari adalah perkiraan dari pola data; data tidak memuat tanggal konfirmasi fraud.
- Skor validasi adalah hasil pengembangan: lipatan yang sama dipakai untuk memilih fitur, cara, dan kalibrasi,
  sehingga cenderung sedikit optimistis.
- Koefisien Platt model final (a = 0,77, b = −0,43) berbeda dari koefisien di lipatan validasi (a ±1,0-1,17,
  b 0,5-2,1). Dugaan penyebab: periode kalibrasi final (14 Jul - 10 Agt) memuat lonjakan fraud QRIS minggu 4-10 Agt.
  Dampaknya ke keputusan kecil: dibanding memakai koefisien lipatan 4, hanya ±0,15% transaksi uji berbeda keputusan.
- Model menua: PR-AUC turun ±0,01 bila jeda diperpanjang dari 21 ke 35 hari, sedangkan data uji berjarak
  21-82 hari dari label terakhir.

## Versi paket

Python 3.12.0; numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, lightgbm 4.7.0, matplotlib 3.11.2,
scipy 1.18.1, pytest 9.2.0.
