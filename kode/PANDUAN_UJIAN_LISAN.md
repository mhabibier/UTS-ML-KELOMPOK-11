# Latihan ujian lisan

Kedua anggota harus dapat menjelaskan semua task. Gunakan ini untuk latihan,
bukan naskah hafalan atau pengganti membaca notebook dan hasil.

## Demonstrasi singkat

Siapkan environment dan dataset sebelum ujian. Contoh reproduksi prediksi Task 2:

```bash
python kode/task2_klasifikasi.py --tahap final
python kode/cek_format.py
```

Perintah pertama melatih ulang model final serta membuat prediksi. Perintah kedua
hanya memeriksa format/ID, bukan membuktikan akurasi. Jika diminta angka validasi,
jalankan tahap yang benar; prediksi final tidak menghasilkan skor hidden test.
Untuk Task 1, jalankan setup/definisi notebook dan cell baseline agar dapat
menunjukkan satu angka hasil hitung ulang. Jangan mengaku reproduksi bila hanya
membuka tabel CSV yang sudah tersimpan.

## Task 1: jawaban dengan bahasa sehari-hari

- **P50 dan P90 itu apa?** P50 perkiraan tengah; P90 batas yang diharapkan
  mencakup sekitar 90% paket. Bukan janji semua paket datang sebelum batas itu.
- **Mengapa jumlah_scan dibuang?** Kita tidak tahu apakah jumlahnya sudah tercatat
  saat pickup. Skor yang membaik belum membuktikan fitur itu boleh dipakai.
- **Mengapa validasi waktu?** Saat dipakai, model menebak bulan yang belum terjadi.
  Acak bisa mencampur masa depan ke latihan.
- **Mengapa tidak refit setelah kalibrasi?** Agar koreksi dihitung untuk model
  yang sama. Refit seluruh train sudah diuji sebagai kandidat, tetapi koreksi lama
  belum tentu cocok. Jelaskan perbedaan model historis dan model submission.
- **Mengapa coverage belum sempurna?** Pola berubah, label lambat tersisih, dan
  koreksi global tidak menjamin tiap kurir. Tunjukkan masalah K07 dan diagnosis
  dua bulan yang justru memburuk setelah kalibrasi.
- **SHAP/PDP membuktikan sebab?** Tidak. Keduanya menjelaskan perilaku prediksi.

## Task 2

- **Mengapa label diberi jeda?** Fraud bisa baru diketahui kemudian. Jeda 21 hari
  adalah asumsi dari pola data karena timestamp konfirmasi tidak tersedia.
- **Fitur riwayat dibuat bagaimana?** Hanya kejadian lebih awal; transaksi pada
  waktu sama tidak saling mengintip. Label historis juga harus sudah matang.
- **Mengapa fraud tidak semuanya ditandai?** Ada biaya review. Menandai masuk akal
  bila peluang fraud dikali nominal lebih besar daripada Rp 25.000.
- **Mengapa PR-AUC?** Fokus pada kemampuan menemukan fraud yang jarang; accuracy
  bisa tinggi meskipun semua transaksi dianggap sah.
- **Apa fungsi Platt?** Menyesuaikan skor agar lebih mendekati peluang, sehingga
  keputusan biaya lebih masuk akal. Kalibrasi tetap dapat memburuk saat pola berubah.
- **Apa kelemahannya?** QRIS dan horizon yang makin jauh; validasi juga digunakan
  untuk memilih model sehingga bukan skor independen baru.

## Task 3

- **Mengapa CLR?** Porsi kategori/jam saling terikat karena jumlahnya satu.
  CLR membandingkan relatif antarporsi; nol ditangani dengan pseudocount.
- **Kenapa pelanggan tanpa transaksi dipisah?** Missing keranjang akan sangat
  mendominasi pemisahan. Nol transaksi 90 hari bukan berarti belum pernah membeli.
- **Mengapa dua cluster aktif?** Beberapa k diuji, lalu kriteria kualitas dan
  stabilitas dipakai bersama. Itu tidak membuktikan referensi dosen juga dua.
- **ARI di sini apa?** Kesamaan keanggotaan antar-run. Nomor cluster boleh tertukar.
  ARI stabilitas bukan ARI terhadap label referensi tersembunyi.
- **Apa masalah HDBSCAN?** Banyak run tidak membentuk cluster memadai. Run gagal
  dipisahkan; hasil valid juga memakai KNN untuk memberi label ke pelanggan lain.
- **Anomali -1 berarti fraud?** Tidak. Artinya jauh dari centroid menurut fitur
  yang dipakai; perlu ditinjau dan tidak otomatis diberi kesimpulan bisnis.

## Saat ditanya tentang AI

Jelaskan bagian yang dibantu Codex/Claude Code dan cara memeriksanya: tes manual,
tes waktu, pengujian ulang, pemeriksaan format, dan review silang. Jangan mengaku
menulis atau memahami bagian yang belum dibaca. Gunakan `kontribusi.md` sebagai
catatan yang telah disepakati bersama.
