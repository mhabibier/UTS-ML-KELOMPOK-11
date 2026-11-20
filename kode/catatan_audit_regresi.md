# Catatan audit Task 1

## Tujuan

Menemukan masalah sebelum membuat prediksi P50/P90 pada saat pickup.
Angka di bawah berasal dari skrip audit pada dataset yang tersedia, bukan
asumsi dari panduan. Dataset sumber tidak diubah.

## Temuan dan keputusan sementara

| Temuan | Bukti | Keputusan sementara |
| --- | --- | --- |
| Train dan test terpisah waktu | Train 250.000 baris, Januari–Agustus 2025; test 60.000 baris, September–Oktober 2025 | Gunakan validasi waktu untuk pengembangan model |
| Berat tidak lengkap | Train 13.897 nilai kosong; test 3.265 | Rencanakan imputasi median dari train tiap fold dan indikator nilai kosong; bandingkan dampaknya nanti |
| Berat kosong terkait jenis layanan | Pada train, 32,87% paket kargo tidak memiliki berat; layanan lain sekitar 2,5% | Jangan menganggap kehilangan data seragam atau langsung menghapus baris; indikator missing mungkin bermanfaat |
| Durasi ekstrem | Median 61,7 jam; maksimum 25.452 jam; 370 baris di atas 720 jam | Investigasi dan analisis sensitivitas; ambang 720 jam hanya penanda audit, bukan aturan penghapusan |
| Mitra baru di test | Satu kategori mitra baru, mencakup 2.342 baris test | Encoder harus dapat menangani kategori baru tanpa error |
| Kota tujuan berkardinalitas tinggi | 405 kategori pada train | Pelajari encoding hanya dari train tiap fold; bandingkan penanganan yang sesuai model |
| Risiko fitur pascapickup | Korelasi Spearman jumlah_scan dengan durasi sekitar 0,769 | Tahan jumlah_scan dari fitur utama sampai waktu ketersediaannya jelas; lakukan eksperimen dengan/tanpa fitur pada fold yang sama |
| Ketidaksesuaian waktu label dengan snapshot | Pickup + durasi pada 2.804 baris mencapai atau melewati 1 September 2025 | Catat keterbatasan data; periksa kematangan label saat mendesain fold |

Kamus menyebut penarikan data pada 31 Agustus 2025. Audit mengasumsikan
akhir hari tersebut sebagai batas. Waktu tiba dihitung dari tanggal, jam pickup,
dan durasi; bukan kolom waktu tiba yang diverifikasi secara independen.
Karena itu, temuan snapshot belum membuktikan penyebab kesalahan label.

Tidak ditemukan ID kosong/duplikat, baris duplikat penuh, ID yang tumpang tindih
antara train dan test, tanggal tidak terbaca, atau target nol/negatif/non-finite.
Aturan rentang numerik yang diperiksa skrip lanjutan juga tidak dilanggar.
Ini tidak berarti seluruh kualitas data telah terbukti baik.

Normalisasi huruf dan spasi tidak mengurangi jumlah kategori pada keenam
kolom yang diperiksa. Banyak baris berubah menjadi huruf kecil karena kode
awalnya memakai huruf besar; perubahan ini sendiri bukan bukti inkonsistensi.

## Hasil uji ketahanan target dan kategori

Eksperimen `uji_ketahanan_regresi.py` memakai LightGBM 31 daun tanpa scan,
parameter tetap dan validasi Mei–Juni yang sama. Target validasi tetap asli.

| Perlakuan | Pinball P50 semua validasi | Pinball P90 semua validasi | Rata-rata kedua loss |
| --- | ---: | ---: | ---: |
| Target asli | 10,092032 | 9,749500 | 9,920766 |
| Cap target train di persentil 99 | 10,089605 | 9,926397 | 10,008001 |

Batas train pada fold Mei adalah 211,809 jam (1.136 target dibatasi), dan
fold Juni 213,4 jam (1.463 target dibatasi). P50 sedikit membaik, tetapi P90
memburuk sehingga skor pemilihan gabungan lebih buruk. Keputusan: pertahankan
target asli pada pipeline utama, setelah syarat kematangan label yang sudah
ditetapkan. Ini tidak membuktikan semua nilai ekstrem benar; hanya strategi
cap yang diuji tidak memperbaiki kriteria pemilihan pada periode ini.

Seed 42 memilih K01 dan K10 hanya dari kategori train. Pada simulasi kurir
baru, 22.757 baris dikeluarkan dari train fold Mei dan 29.406 dari fold Juni.
Total 13.099 paket validasi dari kedua kurir tetap dievaluasi pada semua kondisi.

| Kondisi pada paket K01/K10 yang sama | Pinball P50 | Pinball P90 | Coverage P90 |
| --- | ---: | ---: | ---: |
| Model mempelajari K01/K10 | 10,742493 | 11,007173 | 91,8696% |
| K01/K10 tidak ada di train | 11,152038 | 11,054647 | 92,5261% |

Model tidak gagal pada kategori baru, tetapi loss meningkat. Coverage yang
naik bukan berarti prediksi lebih baik. Keputusan: pertahankan mekanisme
unknown sebagai missing dan laporkan keterbatasannya. Dampak menggabungkan
ketiadaan kategori dengan pengurangan jumlah/komposisi data train. Ini satu
simulasi dua kurir, bukan jaminan ketahanan semua kurir atau kota baru.
Perhitungan ulang kondisi asli identik dengan hasil LightGBM sebelumnya.
Pemeriksaan otomatis memverifikasi cap tidak mengubah sumber, pemilihan kurir
tidak memakai target, dan kategori yang disembunyikan benar-benar unknown.

## Ketersediaan fitur saat prediksi

### Hasil eksperimen jumlah_scan

Skrip `audit_leakage_regresi.py` membandingkan LightGBM 31 daun dengan dan
tanpa scan, pada 64.782 baris validasi Mei–Juni yang sama. Parameter, seed,
preprocessing fitur bersama, dan pembagian waktu identik. Tidak tuning ulang.
Hasil tanpa scan sudah dicocokkan dengan eksperimen LightGBM sebelumnya.

| Kondisi | Pinball P50 | Pinball P90 | Coverage P90 |
| --- | ---: | ---: | ---: |
| Tanpa jumlah_scan | 10,092032 | 9,749500 | 91,7246% |
| Dengan jumlah_scan | 8,455126 | 8,293575 | 91,1704% |

Penambahan scan menurunkan pinball P50 sebesar 16,22% dan P90 sebesar 14,93%.
Ini bukti pengaruh prediktif pada eksperimen ini, bukan bukti pasti leakage.
Jika scan dihitung sepanjang perjalanan, manfaat tersebut berasal dari
informasi yang belum tersedia saat pickup. Jika hanya scan sampai pickup,
fitur mungkin sah. Waktu pengukuran tidak dijelaskan dalam kamus, sehingga
model utama tetap mengecualikannya. Jangan menyebut fitur terbukti bocor
hanya berdasarkan korelasi, gain, atau penurunan loss.

### Daftar keputusan fitur

- `id_paket`: identifier untuk output dan pemeriksaan duplikat, bukan fitur utama.
- `durasi_jam`: target, tidak boleh menjadi input model.
- `tgl_pickup`, `jam_pickup`, karakteristik paket/rute/layanan dan profil:
  kandidat fitur berdasarkan kamus. Prakiraan hujan boleh digunakan karena
  kamus menyebut prakiraan, bukan pengamatan cuaca setelah pengiriman.
- `f01`–`f50`: kamus menyatakan dihitung saat pickup; jangan membuang hanya
  karena anonim atau berkorelasi dengan target.
- `jumlah_scan`: waktu pengukuran tidak dijelaskan. Jumlah scan yang bertambah
  selama pengiriman berisiko memuat informasi masa depan. Korelasi hanya
  petunjuk; konfirmasi makna dan eksperimen dampak masih diperlukan.

## Langkah berikutnya: validasi, kemudian baseline

1. Rancang fold berurutan (contoh: validasi Juni, Juli, Agustus). Pada tiap
   cutoff, data train harus berasal dari sebelumnya; periksa juga apakah
   label telah tersedia menggunakan perkiraan pickup + durasi. Laporkan
   jumlah baris yang tidak memenuhi syarat kematangan label.
2. Untuk meniru horizon test dua bulan, pertimbangkan evaluasi dua bulan
   berurutan dengan cutoff tetap. Tetapkan periode pemilihan parameter,
   kalibrasi, dan evaluasi akhir sebelum eksperimen.
3. Jangan melaporkan coverage pada data kalibrasi sebagai evaluasi independen.
   Penilaian setelah kalibrasi memerlukan periode yang tidak dipakai untuk
   memilih parameter maupun menghitung koreksi.
4. Baseline: kuantil 0,5 dan 0,9 per layanan × pulau_tujuan, dihitung hanya
   dari train fold. Gunakan kuantil global train bila kombinasi tidak dikenal.
5. Bandingkan dengan validasi acak menggunakan prosedur yang sama, lalu
   laporkan hasil sebenarnya; jangan mengasumsikan validasi acak pasti lebih baik.

Audit seluruh label train ini bersifat eksploratif. Periode evaluasi yang
dipilih sesudah audit tidak boleh disebut holdout yang sama sekali belum
pernah dilihat. Label test dosen tetap tidak tersedia.

## Latihan penjelasan lisan

"Kami memeriksa data sebelum modeling. Nilai kosong tidak langsung membuat
baris dibuang, dan nilai ekstrem tidak otomatis salah. Statistik preprocessing
akan dipelajari dari train setiap fold agar validasi tidak membocorkan informasi.
Fitur hanya boleh digunakan jika nilainya sudah tersedia saat pickup."

## Bantuan AI

Codex membantu menyusun skrip audit dan catatan ini serta menjalankan skrip.
Anggota perlu memeriksa hasil, memahami kode, dan mencatat bantuan ini pada
kontribusi.md bersama kelompok. Pernyataan ini bukan klaim persetujuan anggota.
