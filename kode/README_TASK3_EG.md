# Task 3 bagian e-g - stabilitas, profil segmen, dan anomali

Lanjutan bagian a-d (`task3_clustering.py`, lihat `README_TASK3.md`). Preprocessing yang sama dipakai lewat
fungsi `siapkan`: pelanggan tanpa transaksi (6.736) menjadi segmen 0, clustering hanya pada 113.264 pelanggan aktif.

```bash
python kode/task3_lanjutan.py                  # ±6,5 menit; menulis kode/hasil/task3_eg/ dan cluster.csv
python -m pytest kode/tests/test_task3_lanjutan.py -q
```

`SEED = 42` (dari `task3_clustering.py`). Data: `data/clustering.csv.gz`.

## e. Stabilitas

Setiap ulangan: ambil subsampel 80%, latih ulang, prediksi **seluruh** pelanggan aktif, lalu hitung ARI antar-run
(nomor cluster boleh tertukar; yang dinilai keanggotaannya). ARI > 0,8 = stabil.

**Grid k = 2-10** (K-Means, 20 ulangan dari pool tetap 20.000 pelanggan; `stabilitas_per_k.csv`, `stabilitas_per_k.png`):

| k | silhouette | Davies-Bouldin | Calinski-Harabasz | ARI (rata-rata ± sb) |
| --- | --- | --- | --- | --- |
| **2** | **0,145** | **2,243** | **3.677** | **0,989 ± 0,004** |
| 3 | 0,120 | 2,372 | 2.588 | 0,980 ± 0,005 |
| 4 | 0,094 | 2,683 | 2.205 | 0,958 ± 0,014 |
| 5 | 0,090 | 2,560 | 1.913 | 0,740 ± 0,175 |
| 6 | 0,089 | 2,376 | 1.737 | 0,937 ± 0,012 |

Aturan pemilihan (`pilih_k`, ditetapkan sebelum hasil final): hanya k yang stabil; silhouette, Davies-Bouldin, dan
Calinski-Harabasz masing-masing memberi satu suara; suara terbanyak menang (seri: paling stabil).
Ketiga kriteria sepakat pada **k = 2** untuk pelanggan aktif.

**Komposisi kategori** (keputusan yang ditunda dari bagian a-d; `keputusan_kategori.csv`): pada k = 2 stabilitas
0,989 (tanpa) vs 0,990 (dengan), dan kedua partisi hampir sama (ARI 0,959). Blok ini hanya dipakai bila menaikkan
stabilitas lebih dari 2 galat baku, jadi **tidak dipakai** untuk clustering. Kategori tetap dipakai di profil (bagian f).

**Per metode pada k = 2** (30 subsampel 80% dari semua pelanggan aktif; HDBSCAN 20 ulangan pada 6.000 pelanggan lalu
disebar dengan KNN; `stabilitas_metode.csv`):

| Metode | ARI rata-rata ± sb | ARI minimum | Catatan |
| --- | --- | --- | --- |
| **K-Means** | **0,993 ± 0,003** | 0,982 | dipakai untuk model final |
| Gaussian Mixture (diagonal) | 0,985 ± 0,008 | 0,961 | |
| MiniBatch K-Means | 0,850 ± 0,234 | 0,000 | sesekali hasilnya berbeda total |
| HDBSCAN | 0,314 ± 0,453 | 0,000 | ±60% titik dianggap noise; tidak cocok untuk data ini |

Ketidakstabilan yang terlihat pada bagian a-d (MiniBatch K-Means) berasal dari algoritmanya, bukan dari datanya.

## f. Profil dan aksi (`profil_segmen.csv`, `profil_kategori.csv`)

| Segmen | Nama | Pelanggan | Ciri utama (median) | Aksi marketing |
| --- | --- | --- | --- | --- |
| 1 | Pelanggan Premium Loyal | 45.701 (38,1%) | belanja 90 hari Rp 3,94 jt, keranjang Rp 350 rb, 12 transaksi, voucher 31%, belanja pagi, umur 37; kategori khas elektronik, makanan-minuman, ibu-bayi | Program VIP: gratis ongkir tanpa minimum dan akses awal produk baru, bukan diskon besar |
| 2 | Pemburu Promo Malam | 65.297 (54,4%) | belanja Rp 604 rb, keranjang Rp 126 rb, 5 transaksi, voucher 67%, flash sale 52%, porsi belanja malam 35%, umur 29; kategori khas fashion, voucher digital, kecantikan | Flash sale dan voucher terbatas pukul 20.00-23.00 lewat push notification |
| 0 | Belum Bertransaksi | 6.736 (5,6%) | 0 transaksi dalam 90 hari, recency 38 hari, sesi 2,6 per minggu | Kupon pembelian pertama (misalnya potongan Rp 20 ribu tanpa minimum) |
| -1 | Anomali (perlu ditinjau) | 2.266 (1,9%) | jarak terjauh ke centroid | Tidak diikutkan kampanye massal; tinjau kualitas data |

Skrip memeriksa bahwa nama sesuai data (misalnya segmen 1 harus berbelanja terbesar) dan berhenti bila tidak.
Nomor segmen aktif diurutkan menurut median belanja sehingga tidak bergantung pada seed.

## g. Anomali (`anomali_ringkas.csv`)

Pelanggan aktif dengan jarak ke centroid clusternya di atas **persentil 98** diberi label **-1**: 2.266 pelanggan
(1,9% dari semua pelanggan), dalam rentang 1-3% yang disarankan. Pembanding: 34% di antaranya juga termasuk 2%
log-likelihood GMM terendah. Dari 656 pelanggan dengan `rasio_konversi` > 1 (anomali definisi data), hanya 24 yang
tergolong anomali jarak; sisanya tetap diberi segmen biasa karena perilakunya tidak menyimpang.

## cluster.csv final

`id_pelanggan,cluster` untuk 120.000 pelanggan: 1 = Premium Loyal, 2 = Pemburu Promo Malam, 0 = Belum Bertransaksi,
-1 = anomali. ARI terhadap `cluster.csv` sementara bagian a-d = 0,727 (perbedaan dari K-Means penuh vs MiniBatch dan
label anomali).

## Keterbatasan

- k = 2 dipilih karena kriteria internal sepakat dan stabil; segmen referensi dosen bisa berjumlah lain. k = 3, 4, dan 6
  juga stabil (ARI > 0,93), tetapi silhouette dan kriteria lain lebih rendah.
- Grid k memakai pool 20.000 pelanggan (alasan waktu); perbandingan metode memakai subsampel 80% dari semua pelanggan aktif.
- Silhouette rendah (0,145): batas antarsegmen tidak tegas, segmen lebih mirip kontinuum belanja tinggi vs promo.
- Data tanpa label; ARI di sini mengukur kestabilan, bukan kecocokan dengan segmen referensi.
- K-Means memakai banyak thread, sehingga run ulang bisa berbeda sangat kecil (< 0,001) pada beberapa baris
  `stabilitas_per_k.csv` untuk k > 2. Baris k = 2, keputusan, dan `cluster.csv` identik antar-run.
