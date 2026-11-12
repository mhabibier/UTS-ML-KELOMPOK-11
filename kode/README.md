# Task 1 — cara menjalankan

Seluruh kode Task 1 berada dalam [Task1_Regresi.ipynb](../Task1_Regresi.ipynb).
Notebook mandiri: tidak mengimpor atau menjalankan skrip proyek `.py`.

`arsip_skrip_sebelum_notebook/` menyimpan 20 skrip asli sebagai cadangan migrasi.
Folder tersebut tidak dipakai notebook maupun peluncur saat ini. Pengembangan
selanjutnya dilakukan di notebook; arsip tidak perlu diperbarui.
Arsip hanya disimpan lokal dan diabaikan Git; teman kelompok menggunakan notebook.

1. Buka notebook di VS Code dengan extension Python dan Jupyter dari Microsoft.
2. Pilih kernel `.venv` repository ini.
3. Klik **Run All**, atau jalankan cell berurutan dari atas. Tunggu setiap cell selesai.
4. Proses penuh sekitar 8 menit pada lingkungan lokal; tabel/grafik dan prediksi diperbarui.

## Isi notebook

Audit awal dan lanjutan → validasi waktu → baseline → perbandingan validasi →
LightGBM → Ridge → audit leakage → uji ketahanan → kalibrasi/evaluasi →
analisis kesalahan → interpretasi → prediksi final → draf laporan → tes → ringkasan.
Setiap tahap mempunyai penjelasan, definisi fungsi, lalu cell untuk menjalankannya.
Jika kernel di-restart, jalankan ulang cell definisi dari atas sebelum memakai fungsi.

## File yang tetap diperlukan

- `Task1_Regresi.ipynb`: sumber utama kode, termasuk 19 tes otomatis.
- `.venv/`: Python dan library untuk eksekusi lokal.
- `kode/requirements-task1.txt`: versi library eksperimen.
- `kode/requirements-notebook.txt`: tambahan kernel Jupyter.
- `kode/hasil/`: tabel, grafik, konfigurasi, dan draf laporan.
- `prediksi_regresi.csv`: hasil akhir test, 60.000 baris.
- Catatan audit/interpretasi dan draf kontribusi: bahan penjelasan ujian dan laporan.

Dataset disimpan di luar repository, default satu tingkat di atas notebook.
Atur `DATA_DIR` pada cell pengaturan bila lokasinya berbeda. Jangan commit dataset.
Google Colab cloud memerlukan upload dataset dan pemasangan library tersendiri;
file D: laptop tidak tersedia otomatis di cloud. Lingkungan yang diuji adalah
Windows x64, Python 3.13.0, sesuai requirements; Colab cloud belum diuji.

## Peluncur opsional untuk reproduksi kelompok

`run_all.py` hanya membaca dan menjalankan cell notebook, bukan salinan model.
Anda tidak perlu memakainya jika menjalankan notebook dengan klik.

```powershell
.\.venv\Scripts\python.exe kode/run_all.py --task 1
.\.venv\Scripts\python.exe kode/run_all.py --task 1 --mode tests
.\.venv\Scripts\python.exe kode/run_all.py --task 1 --mode report
```

Mode tests hanya menjalankan tes kecil; mode report hanya merangkum hasil yang
sudah ada. Keduanya bukan pelatihan ulang penuh. Task 2/3 belum diintegrasikan.
Tidak ada commit atau push otomatis.

## Batasan hasil

Pemilihan kandidat memakai Mei–Juni. Evaluasi historis: latihan sebelum Juli,
kalibrasi Juli, evaluasi Agustus. Agustus pernah dieksplorasi sehingga bukan
holdout yang sepenuhnya belum dilihat. Label tersedia berdasarkan asumsi waktu
pickup + durasi, dengan potensi bias seleksi label yang sudah tiba.

Model final dilatih sebelum Agustus, dikalibrasi ulang memakai label Agustus
yang tersedia sebelum September, lalu dibekukan untuk test September–Oktober.
Ini menyisihkan data kalibrasi, berbeda dari saran refit seluruh train.
Coverage global tidak menjamin coverage setiap kurir; skor test belum diketahui.
PDP/gain bukan hubungan sebab-akibat. Dataset mentah tidak diubah.

`kode/hasil/laporan_task1.md` masih draf, bukan laporan.pdf final kelompok.
Kontribusi/PR/review anggota dan pengujian di clone bersih masih perlu dilengkapi.
Dokumen dan manifest lama menyebut nama skrip historis sebelum migrasi notebook;
untuk cara menjalankan saat ini gunakan panduan ini.
