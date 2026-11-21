# Interpretasi model regresi

## Apa yang dijelaskan

Partial dependence (PDP) memperlihatkan rata-rata prediksi model ketika satu
fitur diubah untuk semua baris referensi, sementara fitur lain tetap.
Model adalah LightGBM 31 daun yang dilatih sebelum Juli, dengan koreksi P90
dari Juli sebesar +4,592367 jam. Tidak melatih ulang berdasarkan hasil Agustus.

Referensi memakai 2.000 baris train dengan seed 42. Lima fitur dipilih dari
rata-rata porsi gain model P50 dan P90. Gain menunjukkan pengurangan loss
selama pembentukan pohon, bukan persentase sebab-akibat atau perbaikan akurasi
test. Peringkat gain dapat dipengaruhi banyaknya kandidat pemisahan dan
kardinalitas; kota dengan banyak kategori perlu dibaca dengan keterbatasan ini.

## Temuan dari grafik

| Fitur | Pola prediksi yang terlihat | Batas interpretasi |
| --- | --- | --- |
| kota_tujuan | Model memberi perkiraan berbeda untuk kode kota yang berbeda | Grafik hanya 12 kota teratas frekuensi train; mengganti kota tanpa pulau/jarak bisa tidak realistis |
| layanan | Rata-rata prediksi ekspres lebih pendek daripada reguler, hemat, dan kargo | Bukan pengukuran dampak nyata mengganti layanan pada paket yang sama |
| jarak_km | Prediksi umumnya bertambah saat jarak dinaikkan; tidak sepenuhnya linear | Jarak berkaitan dengan rute dan fitur lain; jangan mengekstrapolasi di luar rentang grafik |
| f04 | P90 meningkat lebih tajam ketika nilai fitur mendekati dan melewati sekitar 0,8–1,3 pada grid | Fitur anonim; tidak bisa diberi nama seperti cuaca/risiko tanpa definisi |
| f17 | P50/P90 meningkat terutama pada bagian tengah rentang fitur | Fitur anonim; pola tidak membuktikan mekanisme operasional |

Contoh angka PDP layanan pada referensi yang sama:

| Nilai layanan yang dipasang | Rata-rata prediksi P50 (jam) | Rata-rata prediksi P90 setelah koreksi (jam) |
| --- | ---: | ---: |
| ekspres | 41,13 | 63,09 |
| reguler | 61,74 | 88,14 |
| hemat | 78,57 | 113,67 |
| kargo | 84,14 | 115,30 |

Angka di tabel adalah rata-rata keluaran model pada data hipotetis, bukan
median/kuantil pengiriman nyata per layanan dan bukan janji waktu pengiriman
untuk semua paket. Kuantil dari gabungan distribusi tidak sama dengan rata-rata
kuantil tiap paket.

Pada grid jarak 91,2 sampai 2.636,8 km, rata-rata prediksi P50 berubah dari
47,86 menjadi 76,22 jam, sedangkan P90 dari 80,52 menjadi 110,60 jam.
Ini perbandingan kedua ujung grid, bukan klaim setiap kenaikan jarak memberi
kenaikan yang sama. Numerik kontinu menggunakan rentang persentil 5–95 train.

## Korelasi dan batasan

Spearman train antara f04 dan f05 sekitar 0,700; jarak_km dan f04 sekitar
0,485. Mengubah satu fitur sambil mempertahankan pasangannya dapat menciptakan
kombinasi yang jarang muncul. Korelasi pada tabel hanya antar fitur numerik;
ketergantungan kategori seperti kota–pulau juga perlu diperhatikan.

PDP merangkum rata-rata dan dapat menutupi pola berbeda antar paket. Tidak ada
interval ketidakpastian pada grafik dan tidak diuji stabilitas antar sampel.
Analisis ini tidak menjelaskan penyebab pasti rendahnya coverage K07, tidak
membuktikan leakage, serta tidak mengubah model atau koreksi untuk K07.

Panduan metode: [Partial dependence scikit-learn](https://scikit-learn.org/stable/modules/partial_dependence.html).
Implementasi memakai perhitungan brute yang setara: salin data referensi,
ganti satu kolom, prediksi P50/P90, terapkan koreksi urutan dan kalibrasi,
lalu rata-ratakan. Tiga tes memeriksa hasil manual, grid, dan tipe kategori.

## Latihan ujian lisan

"Kami memakai partial dependence untuk melihat perilaku model. Misalnya,
ketika nilai layanan diganti menjadi ekspres, model memberi rata-rata prediksi
lebih pendek. Tetapi grafik ini bukan bukti bahwa mengganti layanan pasti
mempercepat setiap paket, karena fitur lain saling berkaitan. Fitur anonim
tetap dijelaskan sebagai fitur anonim, tanpa mengarang artinya."
