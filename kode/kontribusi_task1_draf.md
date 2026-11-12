# Draf kontribusi Task 1

Untuk ditinjau dan digabung ke kontribusi.md kelompok. Bukan pernyataan
persetujuan anggota; PR berikut masih menunggu review kelompok.

- Akun GitHub: mhabibier.
- Nama lengkap: Muhammad Habibie Rabbani.
- NIM: 101032300005.
- Peran: Task 1, regresi estimasi durasi pengiriman P50/P90.
- Pekerjaan: audit data/risiko leakage, validasi waktu dan pembanding acak,
  baseline, LightGBM dan Ridge, sensitivitas outlier/kategori baru, kalibrasi,
  analisis error, PDP, prediksi test, reproduksi, dan draf laporan.
- PR untuk pekerjaan analisis/model ini: https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/2
- Commit implementasi: e67ff1d (Complete Task 1 regression notebook, evaluation, and predictions).
- Review oleh anggota lain dan persetujuan isi: belum dilakukan/dikonfirmasi.

## Penggunaan AI

Codex membantu menyusun dan memperbaiki kode Python, menjalankan eksperimen,
menyusun pengujian, tabel/grafik, dokumentasi dan draf laporan, serta memberikan
penjelasan konsep. Pemilik tugas menjalankan skrip melalui VS Code dan
membandingkan output dengan hasil yang didiskusikan.

Verifikasi teknis mencakup contoh hitung manual metrik/kuantil, pemisahan waktu
dan label matang, kategori baru tanpa mempelajari validasi, target validasi
tetap utuh, agregasi metrik kelompok, serta kelengkapan CSV prediksi. Jangan
mengklaim telah membaca seluruh kode atau memahami semua konsep sebelum
benar-benar dilakukan. Semua anggota tetap perlu memahami ketiga task untuk
ujian lisan. Tidak ada data eksternal yang ditambahkan ke fitur model.

## Sebelum diserahkan

- Identitas sudah diberikan pemilik tugas; lengkapi tautan commit/PR yang benar.
- Tinjau pilihan final yang menyisihkan Agustus untuk kalibrasi.
- Pertahankan keterbatasan snapshot, backtest historis, horizon test dua bulan,
  dan coverage rendah K07 dalam laporan.
- Gabungkan Task 1–3 menjadi satu laporan.pdf (maksimal 18 halaman di luar
  lampiran), satu perintah kelompok, serta kontribusi.md yang disetujui anggota.

## Penyempurnaan Task 1 — 12 November 2026

Codex membantu memeriksa lokasi notebook dan peluncur, melengkapi diagnosis
historis dua bulan dengan model tetap, memperjelas kamus fitur serta keterbatasan
K07/kematangan label/kalibrasi dalam laporan, dan menjalankan reproduksi dari
salinan sumber tanpa arsip atau hasil eksperimen lama. Dua tes tambahan memeriksa
batas waktu dan bahwa model tidak dilatih ulang selama horizon dua bulan.
Bukti eksekusi tersedia pada `hasil/reproduksi_task1/verifikasi_bersih.json`.
Revisi ini belum dinyatakan disetujui anggota atau masuk PR; tinjau lalu catat
commit/PR aktual setelah perubahan dikumpulkan melalui alur kelompok.
