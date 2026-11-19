# Kode UTS Machine Learning Kelompok 11

Letakkan dataset mentah di `data/` pada root repository (diabaikan Git).
Instal dependensi bersama dengan `python -m pip install -r requirements.txt`.
Snapshot lokal memakai Python 3.13; reviewer juga mereproduksi Task 1 pada
Python 3.12 dengan NumPy 2.5.5 (snapshot lokal NumPy 2.5.4).

- Task 1: [notebook](Task1_Regresi.ipynb), [panduan](README_TASK1.md).
- Task 2: `python kode/task2_klasifikasi.py`; `--tahap final` tersedia setelah PR final digabung.
- Task 3: [notebook](Task3_Clustering.ipynb), [panduan](README_TASK3.md).

Setelah PR ketiga task digabung, `python kode/run_all.py --task all`
menjalankan ketiganya. Task 3 saat ini hanya a–d; hasil e–g belum tercakup.
Sebelum integrasi lengkap, pilih `--task 1`, `--task 2`, atau `--task 3`.
Peluncur memeriksa keberadaan kode dan berhenti dengan pesan jelas bila PR belum digabung.
`--data-dir` dapat dipakai untuk lokasi alternatif. Task 1 mendukung `--mode tests`.
Notebook tetap bisa dijalankan dengan klik Run All.

Jangan commit data mentah, virtual environment, atau cache.
