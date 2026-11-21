# Kontribusi Kelompok 11

## Anggota dan pekerjaan

| Anggota | NIM | Pekerjaan utama | Bukti pull request |
| --- | --- | --- | --- |
| Muhammad Habibie Rabbani | 101032300005 | Task 1 regresi: audit, validasi, baseline, Ridge/LightGBM, kalibrasi, analisis, interpretasi, prediksi; Task 3 a-d; review silang dan integrasi laporan/peluncur | [Task 1 #2](https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/2), [Task 3 a-d #6](https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/6) |
| Azmi Aziz syah Putra | 101032300216 | Task 2 klasifikasi: audit, fitur riwayat, imbalance, drift, kalibrasi, biaya, analisis, prediksi; Task 3 e-g: stabilitas, profil/aksi, anomali; review silang | [#4](https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/4), [#5](https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/5), [#7](https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/7), [#8](https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/8), [#9](https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/9), [#10](https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/10), [Task 3 e-g #11](https://github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11/pull/11) |

Pengerjaan menggunakan fork masing-masing (`mhabibier` dan `Son0fGuilliman`).
Review silang mencakup dugaan leakage, transaksi pada waktu sama, kematangan label,
reproduksi prediksi, perhitungan stabilitas, dan integrasi runner. Pembagian ini
tidak membatasi tanggung jawab ujian lisan: kedua anggota perlu memahami semua task.

## Penggunaan AI dan cara memeriksa hasil

- **Codex / ChatGPT**, pada pekerjaan Habibie dan review: membantu merancang dan
  menulis/debug kode, membaca audit, menjalankan eksperimen, membuat tes, menyusun
  grafik serta draf laporan, dan meninjau PR. Keluaran diverifikasi dengan tes
  hitungan manual/temporal, pemeriksaan ID dan format CSV, perbandingan hasil
  reproduksi, review kode, serta review silang Azmi. Laporan PDF disusun otomatis
  dari artefak eksperimen; manifest menyimpan sumber angka.
- **Claude Code (Anthropic)**, pada pekerjaan Azmi (Task 2: PR #4, #5, #7-#10; Task 3
  e-g: PR #11; `cek_format.py`; bagian Task 2/3 e-g pada laporan): membantu menganalisis
  data dan audit kebocoran, merancang dan menulis kode (validasi berbasis waktu dengan
  jeda, fitur riwayat tanpa kebocoran, perbandingan penanganan kelas tidak seimbang,
  analisis pergeseran pola, kalibrasi dan aturan biaya, analisis error, model final;
  stabilitas, profil, dan anomali Task 3), menulis unit test, menjalankan eksperimen,
  serta menyusun README dan deskripsi PR. Verifikasi: unit test, termasuk uji mutasi
  untuk tes anti-kebocoran (kebocoran yang sengaja disisipkan harus tertangkap); uji
  analitis dan empiris untuk koreksi undersampling, kalibrasi Platt, dan ambang biaya;
  run penuh pada data asli dengan beberapa seed; run ulang seluruh skrip Task 2 yang
  menghasilkan tabel identik; cek format dan ID; serta review silang Habibie (menemukan
  kasus transaksi serentak, klaim kebocoran yang terlalu pasti, dan perhitungan
  stabilitas HDBSCAN serta galat baku).
- Label test/referensi tersembunyi tidak digunakan; tidak ada dataset eksternal.
  Bantuan AI tidak dianggap bukti kebenaran dan tidak menggantikan pemahaman anggota.

## Persetujuan isi kontribusi dan laporan gabungan

Dokumen ini disiapkan untuk ditinjau kedua anggota. Approval PR kode sebelumnya
tidak otomatis dianggap persetujuan terhadap laporan dan kontribusi gabungan ini.

- [x] Muhammad Habibie Rabbani: menyetujui bagian miliknya dalam laporan gabungan
      serta catatan kontribusi/penggunaan AI untuk dirinya. Bukti: konfirmasi
      langsung melalui chat Codex pada 20 November 2026, setelah PR #12 disiapkan
      (versi `543db9c`): "Saya setuju untuk bagian saya".
- [ ] Azmi Aziz syah Putra: telah membaca, membenarkan kontribusi/penggunaan AI,
      dan menyetujui isi laporan gabungan. Bukti persetujuan: belum dicatat.

Isi tautan komentar/review persetujuan setelah benar-benar diberikan. Sebelum
pengumpulan, kedua persetujuan harus tercatat, semua PR pengumpulan harus masuk
`main`, dan kedua fork disinkronkan dengan repository kelompok.
