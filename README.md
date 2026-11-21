# UTS Machine Learning Kelompok 11

- Muhammad Habibie Rabbani - 101032300005: Task 1 dan Task 3 a-d.
- Azmi Aziz syah Putra - 101032300216: Task 2 dan Task 3 e-g.

Task 1-3 telah diintegrasikan. [Laporan gabungan](laporan.pdf) memuat hasil,
keputusan, serta keterbatasan; [kontribusi](kontribusi.md) memuat pembagian kerja,
bantuan AI, dan status persetujuan anggota.

## Reproduksi

Pasang `requirements.txt`, letakkan lima dataset resmi di `data/`, lalu jalankan:

```bash
python kode/run_all.py --task all
```

Perintah tersebut menghitung ketiga task, memeriksa format prediksi, lalu
membangun laporan. Task 2 penuh memerlukan beberapa jam. Lihat
[panduan kode](kode/README.md) untuk menjalankan satu task atau membangun ulang
PDF dari tabel yang sudah tersedia.

Untuk klik Run All, buka [notebook Task 1](kode/Task1_Regresi.ipynb) atau
[notebook Task 3](kode/Task3_Clustering.ipynb) dengan kernel lingkungan proyek.
Task 3 harus dijalankan sampai e-g agar `cluster.csv` final dibuat.

## Pengumpulan

Enam komponen: `laporan.pdf`, `kode/`, `prediksi_regresi.csv`,
`prediksi_klasifikasi.csv`, `cluster.csv`, dan `kontribusi.md`.
Data mentah, `.venv`, cache, dan render sementara tidak diunggah.
Kedua anggota perlu menyetujui isi laporan/kontribusi, menyinkronkan fork, serta
memastikan PR final masuk `main` sebelum 24 November 2026 pukul 23.59.

Untuk persiapan lisan, baca [panduan latihan](kode/PANDUAN_UJIAN_LISAN.md).
