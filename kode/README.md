# Kode UTS Machine Learning Kelompok 11

Letakkan dataset mentah di `data/` pada root repository (diabaikan Git).
Instal dependensi bersama dengan `python -m pip install -r requirements.txt`.
Snapshot lokal memakai Python 3.13; reviewer juga mereproduksi Task 1 pada
Python 3.12 dengan NumPy 2.5.5 (snapshot lokal NumPy 2.5.4).

- Task 1: [notebook](Task1_Regresi.ipynb), [panduan](README_TASK1.md).
- Task 2: `python kode/task2_klasifikasi.py`; `--tahap final` tersedia setelah PR final digabung.
- Task 3: [notebook](Task3_Clustering.ipynb), [panduan a–d](README_TASK3.md), [panduan e–g](README_TASK3_EG.md).

`python kode/run_all.py --task all` menjalankan ketiga task. `--task 3` menjalankan
`task3_clustering.py` (a–d) lalu `task3_lanjutan.py` (e–g); langkah kedua menimpa
`cluster.csv` sementara dari a–d dengan versi final. Satu task saja: `--task 1`, `--task 2`, atau `--task 3`.
Peluncur memeriksa keberadaan kode dan berhenti dengan pesan jelas bila PR belum digabung.
`--data-dir` dapat dipakai untuk lokasi alternatif. Task 1 mendukung `--mode tests`.
Notebook tetap bisa dijalankan dengan klik Run All.

Jangan commit data mentah, virtual environment, atau cache.
