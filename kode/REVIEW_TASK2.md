# Review audit dan validasi awal Task 2

Basis: PR #5 a0cb053, yang mencakup riwayat PR #4 a9e85a0.

Perbaikan: kamus kategori LightGBM kini hanya dipelajari pada train tiap fold;
kategori baru menjadi missing. Dugaan snapshot dan label matang dibedakan dari
kepastian. Perbandingan acak/waktu tidak diinterpretasikan sebagai efek murni
pengacakan karena ukuran train dan periode evaluasinya juga berbeda.

Verifikasi: 9 tes pytest lulus. Fold waktu pertama LightGBM dilatih ulang pada
data resmi, PR-AUC 0.7562706943767096 dan ROC-AUC 0.9783440795671535, identik
hasil sebelumnya. Bukti: hasil/t2_review_verification.json. Seluruh eksperimen
multi-seed belum diulang; tabel lama tidak diklaim sebagai hasil rerun perbaikan.

Tanggal 11 Agustus dan jeda 21 hari adalah asumsi dari pola label snapshot.
Tanpa timestamp konfirmasi fraud, kematangan setiap label tidak dapat dipastikan.
Penurunan fraud juga bisa mencerminkan drift. Uji sensitivitas cutoff/jeda masih
diperlukan untuk klaim yang lebih kuat. umur_merchant_hari negatif dan skor tinggi
adalah alasan konservatif mengecualikan fitur, bukan bukti kausal kebocoran.
volume_merchant_30h tetap dipantau; pastikan ketersediaan sebelum transaksi
sebelum menjadikannya fitur model final.

## Lingkup yang belum selesai

PR audit/validasi bukan penyelesaian seluruh Task 2. Berikut masih diperlukan:
fitur riwayat pengguna/merchant tanpa masa depan; minimal tiga pendekatan
imbalance; pengujian adaptasi drift; kalibrasi probabilitas pada periode terpisah;
keputusan biaya dengan pembanding tandai semua/tidak ada; analisis per kanal,
kategori merchant, dan pola fraud; prediksi_klasifikasi.csv berisi id_transaksi,
prob,tandai; integrasi reproduksi dan laporan kelompok. Jangan approve sebagai
Task 2 final hanya karena audit dan baseline berhasil.

Perbaikan ini tidak mengubah kepemilikan kontribusi asli. Muhammad Habibie Rabbani
melakukan review dan koreksi dengan bantuan Codex; pekerjaan dasar milik penulis
PR #4/#5. Tidak ada dataset mentah disertakan.

## Tindak lanjut oleh penulis PR #5

- Seluruh eksperimen multi-seed diulang dengan kode perbaikan kamus kategori (sebelum perubahan
  berikut): kesembilan tabel `t2_validasi_*.csv` identik dengan versi a0cb053 (selisih 0), karena
  semua kategori validasi dan uji sudah muncul di data latih lipatan pertama.
- `volume_merchant_30h` kini dikecualikan secara konservatif (nilai konstan per merchant
  sepanjang Jan-Okt). Konstanta `FITUR_BOCOR` diganti `FITUR_DIKECUALIKAN`.
- Bagian 6 ditambah perbandingan acak setara: data validasi dan ukuran data latih sama, hanya asal
  data latih yang berbeda (masa lalu vs acak dari seluruh periode).
- Semua tabel di `kode/hasil/` dibuat ulang dari kode terbaru. `t2_review_verification.json`
  adalah catatan verifikasi reviewer pada kode sebelum perubahan di atas.
