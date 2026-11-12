# Task 1 — menjalankan notebook dan reproduksi

Sumber utama adalah [Task1_Regresi.ipynb](Task1_Regresi.ipynb), sudah berada di
folder `kode/` sesuai ketentuan pengumpulan. Notebook tidak membutuhkan skrip
arsip, hasil eksperimen lama, atau file Python proyek tambahan.

## Persiapan dari clone baru

Gunakan Python 3.13 (lingkungan eksperimen: 3.13.0, Windows x64).
Dari root repository:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r kode/requirements-notebook.txt
```

Requirements notebook mencakup requirements eksperimen dan kernel Jupyter.
Untuk terminal saja, `requirements-task1.txt` cukup; tampilan IPython memiliki
fallback teks. Jangan menyalin `.venv` atau dataset ke GitHub.

Dataset `regresi_train.csv.gz` dan `regresi_test.csv.gz` disediakan terpisah.
Default lokasi dataset adalah folder induk repository. Untuk lokasi lain,
tambahkan `DATA_DIR = r"D:\lokasi\dataset"` di cell pengaturan, sebelum baris
`DATA_DIR = Path(...)`. Hasil selalu ditulis ke `kode/hasil/` dan prediksi final
ke `prediksi_regresi.csv` di root repository, terlepas dari lokasi dataset.

## Menjalankan dengan klik

1. Buka folder repository di VS Code, dengan extension Python dan Jupyter.
2. Buka `kode/Task1_Regresi.ipynb` dan pilih kernel `.venv` yang baru disiapkan.
3. Klik **Run All**. Bisa dimulai dari root repository maupun folder `kode/`.
4. Tunggu hingga ringkasan dan seluruh tes selesai; durasi tergantung perangkat.

Notebook menemukan root dari lokasi kerja kernel. Jika kernel dibuka dari lokasi
lain, isi `PROJECT_DIR` pada cell pengaturan dengan root repository.

## Satu perintah reproduksi

```powershell
.\.venv\Scripts\python.exe kode/run_all.py --task 1 --data-dir "D:\lokasi\dataset"
```

Peluncur menjalankan cell notebook yang sama, termasuk semua eksperimen, tiga
grafik laporan, prediksi, laporan Markdown, dan 21 pemeriksaan otomatis.
Untuk pemeriksaan kecil tanpa dataset, gunakan `--mode tests`. `--mode report`
hanya menyusun laporan dari hasil yang sudah ada; jalankan penuh terlebih dulu.
Mode tests/report tidak menggantikan reproduksi penuh. Task 2/3 belum terintegrasi.

Manifest peluncur: `kode/hasil/reproduksi_task1/manifest_notebook_full.json`.
Metadata tiap eksperimen menyimpan parameter, versi, serta batasan. Metadata
prediksi final mencatat SHA256 kedua dataset dan CSV prediksi. Bukti uji salinan
bersih disimpan dalam `kode/hasil/reproduksi_task1/verifikasi_bersih.json`.

## Protokol evaluasi

- Pengembangan: Mei–Juni; baseline, LightGBM quantile, dan Ridge log-target.
- Evaluasi historis satu bulan: fit sebelum Juli, kalibrasi Juli matang sebelum
  Agustus, evaluasi seluruh label Agustus.
- Diagnosis dua bulan: fit sebelum Juni, kalibrasi Juni matang sebelum Juli,
  model dan koreksi yang sama untuk Juli–Agustus. Metrik tiap bulan dan gabungan
  tersedia pada `hasil/validasi_dua_bulan/`. Ini **eksploratif** karena parameter
  dan periode historis sudah diperiksa; bukan holdout independen.
- Final: fit sebelum Agustus, kalibrasi Agustus matang sebelum September,
  model tetap untuk September–Oktober. Tidak melakukan refit setelah kalibrasi.

Waktu label tersedia diasumsikan sama dengan pickup + durasi; keterlambatan
pencatatan tidak diketahui. Seleksi label matang dapat mengurangi paket lambat.
Semua label evaluasi tetap dinilai, termasuk yang tiba setelah akhir periode.
Coverage global tidak menjamin coverage tiap kurir; K07 perlu perhatian khusus.
Hasil diagnosis dua bulan tidak digunakan untuk menyesuaikan ulang prediksi test.

## Keputusan fitur sesuai kamus

`f01–f50` dihitung saat pickup dan tetap dipakai. Kolom hujan merupakan prakiraan,
sedangkan antrean hub serta kemacetan diketahui saat pickup. `jumlah_scan`
dikecualikan dari model utama karena waktu snapshot-nya belum jelas; eksperimen
scan hanya diagnosis. `id_paket` hanya identitas dan `durasi_jam` hanya target.

Laporan Task 1: [hasil/laporan_task1.md](hasil/laporan_task1.md). Masih perlu
digabung dengan Task 2/3 menjadi `laporan.pdf` maksimal 18 halaman di luar lampiran.
Anggota wajib memeriksa kontribusi, PR, dan catatan penggunaan AI. Dataset dan
arsip skrip lokal tidak menjadi dependensi pengumpulan.
