# Task 3 bagian a-d

Buka Task3_Clustering.ipynb, pilih Python environment proyek, lalu Run All.
Notebook dapat dibuka dari root atau folder kode. Dataset clustering.csv.gz berada
satu tingkat di atas root repository; bila berbeda isi DATA_DIR pada cell pertama.

Lingkungan pengujian: Python 3.13.0 Windows, versi di requirements-task3.txt.
Instal pada environment baru dengan `python -m pip install -r kode/requirements-task3.txt`.
Reproduksi tanpa UI: `python kode/run_task3_ad.py --data-dir LOKASI_DATA`.
Tes helper: `python -m unittest discover -s kode -p test_task3_ad.py`.

Notebook dan peluncur menggunakan kode yang sama. Hasil ada di hasil/task3_ad/.
Metadata keputusan mencatat versi, seed dan hash input. Data mentah tidak diubah.

## Lingkup

a: audit populasi, missing, skala, skew, komposisi, kategori, dan outlier.
b: keputusan fitur, korelasi, bobot blok dan eksperimen sensitivitas fitur.
c: MiniBatch K-Means, Gaussian Mixture diagonal, hierarchical Ward pada sampel
6.000 pelanggan yang sama; preprocessing memakai populasi tanpa label.
d: k=2..12, silhouette pada sampel tetap 1.500 pelanggan, DB dan CH pada 6.000,
serta BIC untuk membandingkan k dalam GMM saja.

Kandidat dipilih dengan rata-rata peringkat silhouette/DB. Kandidat dua cluster
berkaitan kuat dengan missing basket; belum boleh dianggap jumlah segmen bisnis
final. Ablation juga menguji penghilangan indikator missing. Skor internal rendah,
kriteria dapat berbeda, dan preprocessing/fit/evaluasi bukan holdout generalisasi.

partisi_sampel_eksploratif.csv hanya hasil 6.000 pelanggan. Bukan cluster.csv untuk
pengumpulan. Bagian e (stabilitas resampling), f (profil dan aksi), g (anomali -1)
belum dikerjakan dan perlu dilanjutkan bersama. Tidak ada klaim skor ARI referensi
tersembunyi; ARI ablation hanya sensitivitas fitur.

## cluster.csv sementara

Run All juga membuat `cluster.csv` di akar repository untuk kebutuhan integrasi
kelompok: satu baris per pelanggan, kolom tepat `id_pelanggan,cluster`. Kandidat
Gaussian Mixture diagonal k=2 dilatih pada sampel 6.000, lalu memprediksi seluruh
120.000 pelanggan dengan preprocessing a-d yang sama. Metadata ada di
`hasil/task3_ad/cluster_sementara_meta.json`. File ini sementara: belum ada
uji stabilitas, profil dan aksi segmen, atau label anomali `-1`. Ganti hasilnya
setelah bagian e-g selesai sebelum pengumpulan final.

Codex membantu audit, implementasi, eksperimen, tes, dan draf laporan. Muhammad
Habibie Rabbani (101032300005) perlu meninjau dan memahami hasil; kontribusi teman
pada tahap berikutnya dicatat melalui PR sendiri. Belum persetujuan kelompok.
