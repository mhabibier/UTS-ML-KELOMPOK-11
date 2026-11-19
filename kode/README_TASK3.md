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
- Ablation kelompok fitur menunjukkan menambah komposisi kategori menurunkan
  silhouette dari 0,141 menjadi 0,116 dan stabilitas eksploratif dari 0,897 ke
  0,894, sehingga blok itu tidak masuk representasi utama. Menghilangkan
  komposisi waktu menaikkan silhouette tetapi menurunkan stabilitas ke 0,879,
  maka blok itu tetap dipakai. Ablation ini pada sampel yang sama dan belum
  menjadi bukti akhir; bagian e perlu 20–50 subsampel.
- MiniBatch K-Means, Gaussian Mixture diagonal, dan HDBSCAN dibandingkan pada
  6.000 pelanggan aktif yang sama. K-Means/GMM dievaluasi untuk k=2–12 dengan
  silhouette dan Davies–Bouldin; BIC hanya dibandingkan dalam keluarga GMM.
  HDBSCAN diuji dengan beberapa ukuran cluster dan `min_samples`.
- Kandidat sementara pelanggan aktif ialah MiniBatch K-Means k=2 menurut
  silhouette/DB. BIC GMM memilih k=11, jadi kriteria belum sepakat.
  Jangan klaim k=2 sebagai jumlah segmen final sebelum stabilitas dan profil.

Hasil revisi ada di `kode/hasil/task3_ad_revisi/`: tabel metode/k, ablation per
kelompok, keputusan fitur, parameter preprocessing, grafik, dan metadata.
Hasil `kode/hasil/task3_ad/` adalah eksperimen lama sebelum masukan review;
gunakan folder revisi untuk laporan berikutnya.

## cluster.csv sementara

Skrip membuat `cluster.csv` di root dengan tepat satu baris per pelanggan.
Segmen 0 = tanpa transaksi; segmen 1–2 = kandidat K-Means pada pelanggan aktif.
Ini **sementara untuk integrasi**. Bagian e–g masih perlu uji stabilitas penuh,
profil dan aksi pemasaran per segmen, serta aturan anomali `-1`. Ganti file ini
sebelum pengumpulan final. ARI internal ablation bukan skor referensi dosen.

Codex membantu audit, implementasi, eksperimen, tes dan dokumentasi. Muhammad
Habibie Rabbani (101032300005) perlu meninjau hasil; kontribusi teman untuk e–g
dicatat melalui PR sendiri. Belum ada persetujuan kelompok atas pilihan final.
