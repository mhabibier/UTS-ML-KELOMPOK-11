# Task 3 bagian e-g - stabilitas, profil segmen, dan anomali

Lanjutan bagian a-d (`task3_clustering.py`, lihat `README_TASK3.md`). Preprocessing yang sama dipakai lewat
fungsi `siapkan`: pelanggan tanpa transaksi (6.736) menjadi segmen 0, clustering hanya pada 113.264 pelanggan aktif.

```bash
python kode/run_all.py --task 3                # alur final: a-d lalu e-g (±15-25 menit)
python kode/task3_lanjutan.py                  # e-g saja; menulis kode/hasil/task3_eg/ dan cluster.csv final
python -m pytest kode/tests/test_task3_lanjutan.py -q
```

`SEED = 42` (dari `task3_clustering.py`). Data: `data/clustering.csv.gz`. Jumlah thread dikunci 4 (`N_THREAD`,
sama dengan bagian a-d) supaya hasil K-Means sedekat mungkin di mesin lain. Angka stabilitas tetap bisa berbeda di
digit ketiga atau keempat antarmesin (versi Python/NumPy/BLAS); keputusan dan `cluster.csv` sama di kedua mesin anggota.
Tabel di `kode/hasil/task3_eg/` berasal dari run terakhir yang di-commit.

## e. Stabilitas

Setiap ulangan: ambil subsampel 80%, latih ulang, prediksi **seluruh** pelanggan aktif, lalu hitung ARI antar-run
(nomor cluster boleh tertukar; yang dinilai keanggotaannya). ARI > 0,8 = stabil.

**Grid k = 2-10** (K-Means, 20 ulangan dari pool tetap 20.000 pelanggan; `stabilitas_per_k.csv`, `stabilitas_per_k.png`):

| k | silhouette | Davies-Bouldin | Calinski-Harabasz | ARI (rata-rata ± sb) |
| --- | --- | --- | --- | --- |
| **2** | **0,145** | **2,243** | **3.677** | **0,989 ± 0,003** |
| 3 | 0,120 | 2,372 | 2.588 | 0,980 ± 0,005 |
| 4 | 0,094 | 2,683 | 2.205 | 0,958 ± 0,015 |
| 5 | 0,090 | 2,560 | 1.913 | 0,740 ± 0,174 |
| 6 | 0,089 | 2,376 | 1.737 | 0,937 ± 0,012 |

Aturan pemilihan (`pilih_k`, ditetapkan sebelum hasil final): hanya k yang stabil; silhouette, Davies-Bouldin, dan
Calinski-Harabasz masing-masing memberi satu suara; suara terbanyak menang (seri: paling stabil).
Ketiga kriteria sepakat pada **k = 2** untuk pelanggan aktif.

**Komposisi kategori** (keputusan yang ditunda dari bagian a-d; `keputusan_kategori.csv`): pada k = 2 stabilitas
0,9894 (tanpa) vs 0,9904 (dengan): selisih 0,00104, galat baku 0,00055 (run yang di-commit; di mesin lain 0,00098 dan
0,00056), dan kedua partisi hampir sama (ARI 0,959).
Aturan: blok dipakai hanya bila stabilitas naik **minimal 0,01 ARI** (bermakna secara praktis) **dan** lebih dari
2 galat baku. Galat baku selisih dihitung dengan jackknife *leave-one-run-out* berpasangan, karena pasangan ARI saling
berbagi run dan run ke-r memakai subsampel yang sama di kedua representasi. Hasil: **tidak dipakai** untuk clustering;
kategori tetap dipakai di profil (bagian f).

Catatan jujur: ambang praktis 0,01 ditambahkan setelah review PR #11. Dengan aturan "2 galat baku" saja, selisih
sekecil ini (keduanya ±0,99) membuat keputusan berbalik tergantung jumlah thread komputer (12 thread: selisih 0,0017,
galat baku 0,00067 -> dipakai; 4 thread: ±0,001 vs ±0,00055 -> tidak dipakai).

**Per metode pada k = 2** (30 subsampel 80% dari semua pelanggan aktif; `stabilitas_metode.csv`). HDBSCAN tidak
sanggup 90 ribu baris dan tidak punya `predict`, jadi dilatih pada 6.000 pelanggan lalu pelanggan lain diberi label
15 tetangga terdekat di antara titik non-noise (**HDBSCAN + KNN**). Run yang gagal membentuk minimal 2 cluster dicatat
terpisah dan tidak dihitung sebagai partisi; ARI dihitung dari run valid saja.

| Metode | Run valid | ARI rata-rata ± sb | ARI minimum | Catatan |
| --- | --- | --- | --- | --- |
| **K-Means** | 30/30 | **0,993 ± 0,003** | 0,982 | dipakai untuk model final |
| Gaussian Mixture (diagonal) | 30/30 | 0,985 ± 0,008 | 0,961 | |
| MiniBatch K-Means | 30/30 | 0,850 ± 0,234 | 0,000 | sesekali hasilnya berbeda total |
| HDBSCAN + KNN | 12/20 (8 gagal) | 0,480 ± 0,469 | 0,000 | ±60% titik dianggap noise; tidak cocok untuk data ini |

Ketidakstabilan yang terlihat pada bagian a-d (MiniBatch K-Means) berasal dari algoritmanya, bukan dari datanya.

## f. Profil dan aksi (`profil_segmen.csv`, `profil_kategori.csv`)

| Segmen | Nama | Pelanggan | Ciri utama (median) | Aksi marketing |
| --- | --- | --- | --- | --- |
| 1 | Pelanggan Premium Loyal | 45.701 (38,1%) | belanja 90 hari Rp 3,94 jt, keranjang Rp 350 rb, 12 transaksi, voucher 31%, belanja pagi, umur 37; kategori khas elektronik, makanan-minuman, ibu-bayi | Program VIP: gratis ongkir tanpa minimum dan akses awal produk baru, bukan diskon besar |
| 2 | Pemburu Promo Malam | 65.297 (54,4%) | belanja Rp 604 rb, keranjang Rp 126 rb, 5 transaksi, voucher 67%, flash sale 52%, porsi belanja malam 35%, umur 29; kategori khas fashion, voucher digital, kecantikan | Flash sale dan voucher terbatas pukul 20.00-23.00 lewat push notification |
| 0 | Tidak Aktif 90 Hari | 6.736 (5,6%) | 0 transaksi dalam 90 hari, recency 38 hari, sesi 2,6 per minggu, lama keanggotaan ±21 bulan (kemungkinan pelanggan lama yang berhenti, bukan yang belum pernah membeli) | Kampanye reaktivasi (voucher kembali berbelanja dengan masa berlaku singkat); cek riwayat lama dulu |
| -1 | Anomali (perlu ditinjau) | 2.266 (1,9%) | jarak terjauh ke centroid | Tidak diikutkan kampanye massal; tinjau kualitas data |

Skrip memeriksa bahwa nama sesuai data (misalnya segmen 1 harus berbelanja terbesar) dan berhenti bila tidak.
Nomor segmen aktif diurutkan menurut median belanja sehingga tidak bergantung pada seed.

## g. Anomali (`anomali_ringkas.csv`)

Pelanggan aktif dengan jarak ke centroid clusternya di atas **persentil 98** diberi label **-1**: 2.266 pelanggan
(1,9% dari semua pelanggan), dalam rentang 1-3% yang disarankan. Pembanding: 34% di antaranya juga termasuk 2%
log-likelihood GMM terendah. Dari 656 pelanggan dengan `rasio_konversi` > 1 (anomali definisi data), hanya 24 yang
tergolong anomali jarak; sisanya tetap diberi segmen biasa karena perilakunya tidak menyimpang.

## cluster.csv final

`id_pelanggan,cluster` untuk 120.000 pelanggan: 1 = Premium Loyal, 2 = Pemburu Promo Malam, 0 = Tidak Aktif 90 Hari,
-1 = anomali. ARI terhadap `cluster.csv` sementara bagian a-d = 0,727 (perbedaan dari K-Means penuh vs MiniBatch dan
label anomali).

## Keterbatasan

- k = 2 dipilih karena kriteria internal sepakat dan stabil; segmen referensi dosen bisa berjumlah lain. k = 3, 4, dan 6
  juga stabil (ARI > 0,93), tetapi silhouette dan kriteria lain lebih rendah.
- Grid k memakai pool 20.000 pelanggan (alasan waktu); perbandingan metode memakai subsampel 80% dari semua pelanggan aktif.
- Silhouette rendah (0,145): batas antarsegmen tidak tegas, segmen lebih mirip kontinuum belanja tinggi vs promo.
- Data tanpa label; ARI di sini mengukur kestabilan, bukan kecocokan dengan segmen referensi.
- Hasil K-Means bergantung (sangat sedikit) pada jumlah thread; karena itu thread dikunci 4. Pada mesin dengan
  kurang dari 4 core, angka bisa bergeser di digit ketiga atau keempat.
- Stabilitas HDBSCAN + KNN mencakup efek penyebaran KNN, bukan HDBSCAN murni.
