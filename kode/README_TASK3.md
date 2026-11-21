# Task 3 bagian a–d (revisi setelah review kelompok)

Data mentah `clustering.csv.gz` diletakkan di `data/` pada root repository dan
tetap diabaikan Git. Dari root repo, jalankan:

```bash
python kode/run_task3_ad.py
python -m unittest discover -s kode -p test_task3_ad.py
```

Untuk susunan lokal lama, `python kode/run_task3_ad.py --data-dir ..` juga bisa.
Notebook `kode/Task3_Clustering.ipynb` memanggil fungsi `run` dari
`kode/task3_clustering.py`; klik **Run All** untuk hasil yang sama. Modul `.py`
tidak mengimpor IPython dan dapat dipakai teman untuk bagian e–g. Lingkungan
yang diuji: Python 3.13, versi pustaka pada `requirements-task3.txt`.

## Keputusan a–d

- Pelanggan dengan `frekuensi_90h=0` selalu memiliki `rata2_keranjang` kosong
  (6.736 pelanggan). Mereka dipisahkan sebagai kandidat segmen 0. Clustering
  dilakukan pada 113.264 pelanggan aktif. Ini mencegah indikator missing menjadi
  satu-satunya pembeda dua cluster.
- Komposisi kategori dan jam belanja ditransformasi dengan CLR setelah
  pseudocount `0,0001`. Fitur numerik diimputasi median, dibatasi persentil 1–99,
  log1p bila miring, lalu distandardisasi. ID dan fitur CRM anonim tidak dipakai.
- Keputusan komposisi kategori **belum final**. Tiga run awal tidak cukup
  untuk menyimpulkan blok kategori merugikan stabilitas. Reviewer melaporkan
  10 run: tanpa kategori ARI 0,736 ± 0,316, dengan kategori 0,867 ± 0,072.
  Angka reviewer belum direproduksi oleh skrip a–d ini. Bagian e membandingkan
  keduanya pada 20–50 subsampel yang sama sebelum memilih representasi final.
  CSV sementara tetap memakai konfigurasi lama agar hasil review dapat dilacak.
  Profil kategori dominan tetap dapat dihitung dari data asli pada bagian f.
- MiniBatch K-Means, Gaussian Mixture diagonal, dan HDBSCAN dibandingkan pada
  6.000 pelanggan aktif yang sama. K-Means/GMM dievaluasi untuk k=2–12 dengan
  silhouette dan Davies–Bouldin; BIC hanya dibandingkan dalam keluarga GMM.
  HDBSCAN diuji dengan beberapa ukuran cluster dan `min_samples`.
- Kandidat sementara pelanggan aktif ialah MiniBatch K-Means k=2 menurut
  silhouette/DB. BIC GMM memilih k=11, jadi kriteria belum sepakat.
  Jangan klaim k=2 sebagai jumlah segmen final sebelum stabilitas dan profil.

Hasil revisi ada di `kode/hasil/task3_ad_revisi/`: tabel metode/k, ablation per
kelompok, keputusan fitur, parameter preprocessing, grafik, dan metadata.
Hasil eksperimen lama dihapus dari working tree; riwayatnya tetap tersedia di Git.

## cluster.csv sementara dan alur final

Skrip a–d membuat `cluster.csv` **sementara** di root (segmen 0 = tanpa transaksi;
segmen 1–2 = kandidat MiniBatch K-Means pada pelanggan aktif). Bagian e–g
(`kode/task3_lanjutan.py`, lihat [README_TASK3_EG.md](README_TASK3_EG.md)) menetapkan
representasi, metode, k, profil, dan anomali `-1`, lalu menimpa berkas itu dengan
versi final. Alur final Task 3:

```bash
python kode/run_all.py --task 3     # a–d lalu e–g; cluster.csv akhir = versi final
```

Menjalankan `run_task3_ad.py` saja akan menimpa `cluster.csv` final dengan versi
sementara; jalankan `python kode/task3_lanjutan.py` sesudahnya untuk mengembalikannya.
Notebook menjalankan keduanya secara berurutan. ARI internal ablation bukan skor referensi dosen.

Codex membantu audit, implementasi, eksperimen, tes dan dokumentasi. Muhammad
Habibie Rabbani (101032300005) perlu meninjau hasil; kontribusi teman untuk e–g
dicatat melalui PR sendiri. Belum ada persetujuan kelompok atas pilihan final.
