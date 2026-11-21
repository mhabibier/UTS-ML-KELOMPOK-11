"""Bangun laporan.pdf dari tabel/grafik eksperimen; tidak melatih atau memilih model."""
from pathlib import Path
import hashlib
import json
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / 'kode/hasil'
WIDTH = A4[0] - 88
NAVY = colors.HexColor('#17344D')
TEAL = colors.HexColor('#0D777C')
S = getSampleStyleSheet()
S.add(ParagraphStyle(name='Body', fontName='Helvetica', fontSize=10, leading=14, spaceAfter=9))
S.add(ParagraphStyle(name='SmallNote', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor('#536170'), spaceAfter=7))
S.add(ParagraphStyle(name='Cell', fontName='Helvetica', fontSize=8, leading=10))
S.add(ParagraphStyle(name='Cover', fontName='Helvetica-Bold', fontSize=26, leading=32, textColor=NAVY, spaceAfter=18))
S['Heading1'].textColor = NAVY
S['Heading1'].fontSize = 19
S['Heading1'].leading = 24
S['Heading1'].spaceAfter = 14
S['Heading2'].textColor = TEAL
S['Heading2'].fontSize = 12
S['Heading2'].leading = 16
S['Heading2'].spaceBefore = 10
S['Heading2'].spaceAfter = 8
story = []
sources = set()

def hash_source(path):
    """Normalisasi akhir baris teks agar manifest konsisten di Windows/Linux."""
    path=Path(path)
    raw=path.read_bytes()
    if path.suffix.lower() in {'.csv','.json','.md','.py','.txt'}:
        raw=raw.replace(b'\r\n',b'\n')
    return hashlib.sha256(raw).hexdigest()

def clean(value):
    return str(value).replace('–', '-').replace('—', '-').replace('≥', '>=').replace('−', '-').replace('×', 'x')

def p(text, style='Body'):
    story.append(Paragraph(escape(clean(text)).replace('\n', '<br/>'), S[style]))

def heading(text): story.append(Paragraph(escape(text), S['Heading2']))

def page(title):
    if story: story.append(PageBreak())
    story.append(Paragraph(escape(title), S['Heading1']))

def csv(name):
    sources.add('kode/hasil/' + name)
    return pd.read_csv(H/name)

def js(name):
    sources.add('kode/hasil/'+name)
    return json.loads((H/name).read_text(encoding='utf8'))

def fmt(v):
    if pd.isna(v): return '-'
    if isinstance(v, float):
        return f'{v:,.0f}' if abs(v)>=1000 or v.is_integer() else f'{v:.4f}'
    return clean(v)

def table(frame, columns, widths=None):
    rows = [[Paragraph(escape(clean(title)), S['Cell']) for title in columns.values()]]
    rows += [[Paragraph(escape(fmt(row[col])),S['Cell']) for col in columns] for _,row in frame.iterrows()]
    w = [WIDTH/len(columns)]*len(columns) if widths is None else [WIDTH*x/sum(widths) for x in widths]
    t=Table(rows,colWidths=w,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCEAED')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.8,TEAL),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F6F8')])]))
    story.extend([t,Spacer(1,9)])

def fig(path, max_height=240):
    sources.add('kode/hasil/'+path)
    im=Image(str(H/path)); scale=min(WIDTH/im.imageWidth,max_height/im.imageHeight)
    im.drawWidth=im.imageWidth*scale; im.drawHeight=im.imageHeight*scale
    story.extend([im,Spacer(1,7)])

def note(*names): p('Sumber artefak: '+ '; '.join(names)+'. Seluruh path relatif terhadap kode/hasil/.','SmallNote')

def footer(canvas, doc):
    canvas.setStrokeColor(TEAL); canvas.line(44,42,A4[0]-44,42)
    canvas.setFont('Helvetica',8); canvas.setFillColor(NAVY)
    canvas.drawString(44,29,'UTS Machine Learning | Kelompok 11 | 2026')
    canvas.drawRightString(A4[0]-44,29,str(doc.page))

def build():
    audit=js('audit_regresi/ringkasan.json')
    final=js('prediksi_final_regresi/konfigurasi.json')
    t3=js('task3_eg/keputusan_eg.json')
    preds=[]
    for file,idcol in [('prediksi_regresi.csv','id_paket'),('prediksi_klasifikasi.csv','id_transaksi'),('cluster.csv','id_pelanggan')]:
        f=pd.read_csv(ROOT/file); sources.add(file)
        assert f[idcol].is_unique and not f[idcol].isna().any()
        preds.append({'berkas':file,'baris':len(f),'kolom':', '.join(f.columns)})

    p('LAPORAN UTS\nMACHINE LEARNING','Cover')
    p('Regresi, klasifikasi, dan segmentasi pelanggan','Heading2')
    p('Kelompok 11\nMuhammad Habibie Rabbani - 101032300005\nAzmi Aziz syah Putra - 101032300216')
    p('Implementasi menggunakan data sintetis resmi dosen. Seluruh fitur prediksi diaudit terhadap waktu penggunaan; label test dan segmen referensi tidak tersedia. Hasil berikut merupakan hasil pengembangan dan evaluasi historis, bukan skor hidden test.')
    heading('Ringkasan keluaran')
    table(pd.DataFrame(preds),{'berkas':'Berkas pengumpulan','baris':'Jumlah baris','kolom':'Kolom'},[2,1,2])
    heading('Pilihan utama')
    p('Task 1 memakai LightGBM quantile untuk P50/P90 dengan koreksi satu sisi dan pemisahan periode fit-kalibrasi. Task 2 memakai LightGBM, fitur riwayat, kalibrasi Platt, serta keputusan berbasis biaya. Task 3 memakai K-Means pada pelanggan aktif, satu kelompok tanpa transaksi dalam 90 hari, dan label anomali -1.')
    p('Pembagian kerja: Habibie mengerjakan Task 1 dan Task 3 a-d; Azmi mengerjakan Task 2 dan Task 3 e-g. Keduanya melakukan review silang melalui pull request. Laporan dan kontribusi gabungan harus dibaca serta disetujui kedua anggota sebelum pengumpulan.')
    p('Repository: github.com/telkom-ml-kelompok-11-2026/UTS-ML-KELOMPOK-11\nSumber angka: CSV/JSON hasil eksperimen dan berkas prediksi yang dibaca langsung oleh kode/susun_laporan_kelompok.py.','SmallNote')
    p('Notasi tabel: titik desimal dan koma ribuan. Coverage, probabilitas, recall, serta porsi ditulis pada skala 0-1 kecuali dinyatakan persen. Pinball loss regresi dalam jam; ARI adalah indeks kesamaan dan dapat bernilai negatif.','SmallNote')

    page('1. Regresi: audit dan waktu prediksi')
    p(f"Target durasi_jam diprediksi ketika paket diambil kurir. Data latih berisi {audit['train']['baris']:,} paket; test berisi {preds[0]['baris']:,} paket pada September-Oktober 2025. P50 adalah median prediktif, P90 adalah batas atas satu sisi dengan sasaran cakupan 90%.")
    heading('Audit kualitas dan ketersediaan fitur')
    p(f"Kolom berat_kg kosong pada {audit['train']['nilai_kosong']['berat_kg']:,} baris train. Nilai kosong tidak diisi menggunakan test: LightGBM menerima missing dan indikator berat_kosong; pipeline Ridge mempelajari median, skala, serta kategori hanya dari periode fit.")
    p('ID paket tidak dipakai sebagai fitur. jumlah_scan dikecualikan karena waktu snapshot tidak jelas; manfaat prediktifnya bukan bukti bahwa informasi tersedia saat pickup. Fitur f01-f50 dipakai karena kamus menyatakan dihitung saat pickup. Hujan adalah prakiraan, sementara antrean hub dan kemacetan dinyatakan tersedia ketika pickup.')
    table(csv('audit_leakage_regresi/dampak_scan.csv'),{'metrik':'Metrik','tanpa_scan':'Tanpa scan','dengan_scan':'Dengan scan','penurunan_persen':'Loss turun (%)'})
    p('Eksperimen scan memakai data dan parameter yang sama. Penurunan loss menunjukkan sensitivitas terhadap kolom tersebut, tetapi tidak membuktikan kapan scan tercatat. Keputusan konservatif adalah mengecualikannya dari model utama.')
    heading('Asumsi label dan nilai ekstrem')
    p('Label dianggap tersedia pada pickup + durasi_jam; timestamp konfirmasi sebenarnya tidak tersedia. Hanya label yang diperkirakan matang sebelum cutoff masuk fit/kalibrasi. Seluruh label periode evaluasi tetap dinilai secara retrospektif, termasuk paket lambat. Penyaringan label matang berpotensi mengurangi representasi pengiriman lambat. Durasi ekstrem tetap dipertahankan; eksperimen cap persentil 99 train tidak memperbaiki kriteria gabungan.')
    note('audit_regresi/ringkasan.json','audit_leakage_regresi/dampak_scan.csv','uji_ketahanan_regresi/metrik_gabungan.csv')

    page('2. Regresi: validasi dan model')
    p('Expanding window melatih pada masa lalu dan mengevaluasi bulan berikutnya. Baseline global dan layanan x pulau memakai kuantil train; kelompok baru memakai fallback global. Parameter dua model lanjutan dipilih pada Mei-Juni dengan rata-rata pinball P50/P90 berbobot jumlah baris.')
    table(csv('perbandingan_validasi/metrik_gabungan.csv'),{'skema':'Skema','jumlah_validasi':'Baris evaluasi','pinball_p50':'Pinball P50','pinball_p90':'Pinball P90','coverage_p90':'Coverage P90'},[1,1.2,1,1,1])
    p('KFold acak menggunakan periode serta ukuran train berbeda dan dapat memakai masa depan. Selisih skor bukan efek murni pengacakan. Validasi waktu dipilih untuk meniru prediksi bulan berikutnya.')
    models=pd.concat([csv('lightgbm_regresi/metrik_gabungan.csv'),csv('linear_regresi/metrik_gabungan.csv').query("model != 'baseline'")],ignore_index=True)
    table(models,{'model':'Model','pinball_p50':'Pinball P50','pinball_p90':'Pinball P90','coverage_p90':'Coverage P90','skor_pemilihan':'Skor pilih'},[1.3,1,1,1,1])
    p('LightGBM menggunakan objective quantile terpisah untuk q=0,5 dan q=0,9; 200 pohon, learning rate 0,05, min_child_samples 100, reg_lambda 1. Num_leaves 15 dan 31 dibandingkan; 31 dipilih. Dua konfigurasi boosting tetap dihitung satu keluarga model. Ridge memakai log1p target dan alpha 1/100; kuantil residual dipelajari dari akhir periode train yang dipisahkan dari fit.')
    p('Kategori LightGBM dipelajari dari train; kategori baru menjadi missing. Ridge menggunakan one-hot dan menggabungkan kategori jarang. Simulasi kurir baru juga mengurangi ukuran train, sehingga hasilnya bukan efek kategori baru semata. Skor pengembangan tidak dianggap evaluasi independen: Agustus sudah pernah diperiksa melalui audit/baseline, dan sebagian label pengembangan baru matang setelah cutoff berikutnya.')
    note('lightgbm_regresi/metrik_gabungan.csv','linear_regresi/metrik_gabungan.csv','perbandingan_validasi/metrik_gabungan.csv')

    page('3. Regresi: kalibrasi dan analisis error')
    p('Model historis dilatih sebelum Juli. Koreksi dari Juli memakai statistik urut ceil((n+1) x 0,9) dari residual y - P90. P90 akhir = max(P90 + koreksi, P50); P50 tidak dikalibrasi. Model tetap untuk evaluasi seluruh Agustus. Ini koreksi bergaya split conformal; drift dan seleksi label matang membatasi jaminan coverage.')
    cal=csv('kalibrasi_regresi/metrik.csv'); cal=cal[cal['model'].isin(['baseline_tanpa_kalibrasi','lgb_sebelum','lgb_sesudah'])]
    if 'periode' in cal: cal=cal[cal.periode.astype(str).str.contains('agustus',case=False)]
    table(cal,{'model':'Model Agustus','pinball_p50':'Pinball P50','pinball_p90':'Pinball P90','coverage_p90':'Coverage P90'},[1.8,1,1,1])
    fig('analisis_error_regresi/coverage_per_kelompok.png',220)
    errors=csv('analisis_error_regresi/error_per_mitra_kurir.csv')
    k07=errors[(errors.model=='lgb_sesudah') & (errors.kelompok=='K07')].iloc[0]
    p(f"Coverage global menyembunyikan kelompok lemah. K07 hanya mencapai {100*k07.coverage_p90:.2f}% pada {int(k07.jumlah):,} paket. Analisis terpisah juga tersedia per layanan dan pulau tujuan. Temuan ini tidak membuktikan penyebab operasional; tidak dilakukan tuning koreksi per kurir menggunakan Agustus.")
    note('kalibrasi_regresi/metrik.csv','analisis_error_regresi/error_per_layanan.csv','analisis_error_regresi/error_per_pulau_tujuan.csv','analisis_error_regresi/error_per_mitra_kurir.csv')

    page('4. Regresi: interpretasi model')
    p('Partial dependence (PDP) dihitung pada sampel train untuk fitur dengan gain tinggi. Prediksi ekspres lebih singkat dan durasi cenderung meningkat terhadap jarak pada rentang grid. Fitur anonim f04/f17 tidak diberi arti bisnis rekaan. Fitur berkorelasi dapat membentuk kombinasi hipotetis yang kurang realistis; PDP bukan sebab-akibat.')
    fig('interpretasi_regresi/partial_dependence.png',275)
    p('TreeSHAP bawaan LightGBM dihitung pada 10.000 sampel evaluasi Agustus. Jumlah kontribusi dan bias diverifikasi sama dengan output mentah model. Ringkasan tersedia global dan per layanan, pulau, serta kurir. SHAP ini menjelaskan output sebelum clipping/kalibrasi, sehingga tidak disamakan dengan kontribusi pipeline akhir.')
    fig('review_tambahan/shap_p90.png',240)
    note('interpretasi_regresi/partial_dependence.csv','review_tambahan/shap_ringkasan.csv')

    page('5. Regresi: horizon dan pilihan submission')
    p('Diagnosis dua bulan memakai model sebelum Juni, kalibrasi Juni matang sebelum Juli, lalu model dan koreksi tetap pada Juli-Agustus. Ini eksploratif karena parameter/periode pernah diperiksa. Hasilnya dilaporkan tanpa memilih ulang koreksi dari periode evaluasi.')
    horizon=csv('validasi_dua_bulan/metrik.csv'); horizon=horizon[horizon.model.isin(['lgb_sebelum','lgb_sesudah'])]
    table(horizon,{'periode':'Periode','model':'Model','pinball_p90':'Pinball P90','coverage_p90':'Coverage P90'},[1,1.4,1,1])
    p('Koreksi Juni negatif menurunkan batas atas; cakupan horizon dua bulan justru memburuk setelah koreksi. Kalibrasi historis tidak otomatis memperbaiki masa depan. Model menua dan bias label matang tetap menjadi risiko.')
    heading('Perbandingan refit eksploratif di Agustus')
    table(csv('review_tambahan/perbandingan_refit_agustus.csv'),{'model':'Pipeline','pinball_p50':'P50','pinball_p90':'P90','coverage_p90':'Coverage P90'},[2,1,1,1])
    p(f"Submission yang dipertahankan adalah prediksi_regresi.csv dari model sebelum Agustus, dengan {final['pembagian']['train_dipakai']:,} baris fit dan {final['pembagian']['kalibrasi_agustus']:,} label Agustus untuk kalibrasi. Koreksi P90 final {final['koreksi_p90_jam']:+.6f} jam. Model dan koreksi tetap untuk September-Oktober.")
    p('Kandidat refit seluruh train disimpan sebagai eksperimen terpisah. Refit mengubah model; koreksi dari model lama tidak otomatis memiliki jaminan coverage pada model baru. Menggunakan seluruh label file juga berbeda dari asumsi label tersedia sebelum September. Ketentuan utama meminta P50/P90; saran refit seluruh train pada panduan diperlakukan sebagai tradeoff yang diuji dan dijelaskan. Skor test tidak diketahui.')
    note('validasi_dua_bulan/metrik.csv','review_tambahan/perbandingan_refit_agustus.csv','prediksi_final_regresi/konfigurasi.json')

    page('6. Fraud: audit, waktu, dan fitur')
    p('Peluang is_fraud diprediksi sebelum transaksi diproses. Review transaksi yang ditandai selalu berbiaya Rp 25.000. Jumlah nominal fraud yang lolos ditambahkan ke biaya. Kolom ID tidak menjadi fitur. flag_review_manual, umur_merchant_hari dan volume_merchant_30h dikecualikan secara konservatif karena ketersediaan/snapshot bermasalah; pola korelasi tidak membuktikan waktu pencatatan.')
    table(csv('t2_audit_ringkasan.csv'),{'masalah':'Temuan audit','jumlah_baris':'Jumlah'},[4,1])
    heading('Dampak kolom yang dicurigai bocor')
    bocor=csv('t2_dampak_kebocoran.csv').replace({'fitur_dibuang':{'(tidak ada)':'tidak ada (semua kolom dipakai)'}})
    table(bocor,{'fitur_dibuang':'Kolom dibuang','pr_auc_rata2':'PR-AUC rata-rata','pr_auc_lipatan1':'Fold 1','pr_auc_lipatan4':'Fold 4'},[2.4,1,1,1])
    p('Model dan fold sama; satu-satunya perbedaan adalah ada/tidaknya flag_review_manual. Skor dengan kolom ini "terlalu bagus" karena kamus menyebutnya "pernah ditinjau manual" tanpa waktu tinjauan, dan tingkat fraud 30% bila flag=1 vs 0,13% bila flag=0. Kolom ini diduga kuat dicatat setelah transaksi sehingga dikecualikan; penurunan skor tidak membuktikan waktu pencatatannya.')
    heading('Keterlambatan label')
    fig('t2_fraud_per_minggu.png',165)
    p('Kematangan label tidak memiliki timestamp konfirmasi. Penurunan fraud setelah 10 Agustus diperlakukan sebagai dugaan label belum matang: cutoff 11 Agustus dan jeda 21 hari merupakan asumsi, bukan fakta per transaksi. Data setelah cutoff dikeluarkan dari fit/evaluasi berlabel; fitur perilaku tanpa label dapat memakai transaksi masa lalu yang sudah terjadi.')
    p('Uji pendukung: model yang dilatih sebelum 23 Juni menilai minggu-minggu sesudahnya. Jumlah fraud yang diharapkan model stabil, tetapi fraud yang tercatat turun tepat mulai 11 Agustus, dan presisi 100 skor teratas ikut jatuh. Pola ini konsisten dengan label yang belum terkonfirmasi, bukan dengan perubahan skor model.')
    fig('t2_kematangan_label.png',165)
    km=csv('t2_kematangan_label.csv')
    table(km,{'minggu':'Minggu mulai','fraud_teramati':'Fraud tercatat','fraud_diharapkan':'Diharapkan model','rasio_relatif':'Rasio relatif','presisi_top100':'Presisi top-100'},[1.4,1,1,1,1])
    heading('Lipatan validasi berbasis waktu dan fitur riwayat')
    table(csv('t2_validasi_lipatan.csv'),{'lipatan':'Fold','latih_sampai_sebelum':'Fit sebelum','valid_mulai':'Mulai validasi','valid_sampai_sebelum':'Akhir eksklusif','valid_baris':'Baris'},[.5,1.2,1.2,1.2,.8])
    p('Fitur riwayat pengguna/merchant hanya memakai transaksi dengan waktu lebih awal. Transaksi pada timestamp sama tidak saling menjadi riwayat. Fitur fraud historis memakai label sebelum batas pengetahuan dan jeda kematangan. Kamus kategori hanya dipelajari pada train. Tes mengubah data masa depan untuk memastikan fitur masa lalu tetap sama.')
    note('t2_audit_ringkasan.csv','t2_dampak_kebocoran.csv','t2_fraud_per_minggu.png','t2_kematangan_label.csv','t2_validasi_lipatan.csv','t2_validasi_acak_vs_waktu.csv')

    page('7. Fraud: model, imbalance, dan keputusan')
    table(csv('t2_validasi_model_dasar.csv'),{'model':'Baseline / model awal','pr_auc_mean':'PR-AUC rata-rata','roc_auc_mean':'ROC-AUC'},[2,1,1])
    p('Seleksi maju membandingkan kelompok fitur pada lipatan dan seed yang sama. Merchant terlihat, fraud historis merchant, dan riwayat pengguna diterima; fraud pengguna serta penanda tambahan ditolak. Model terpilih menggunakan 87 fitur. Skor pemilihan adalah skor pengembangan pada fold yang juga digunakan memilih, sehingga dapat optimistis.')
    table(csv('t2_fitur_seleksi.csv'),{'kelompok':'Kelompok fitur','pr_auc_calon':'PR-AUC calon','keputusan':'Keputusan'},[2.5,1,1])
    heading('Perbandingan ketidakseimbangan kelas')
    table(csv('t2_imbalance_ringkas.csv'),{'cara':'Pendekatan','pr_auc':'PR-AUC','biaya_juta':'Biaya (juta Rp)'},[3,1,1])
    p('Minimal tiga pendekatan dibandingkan: bobot kelas, undersampling dengan koreksi prior, dan keputusan/ambang berbasis biaya; bobot nominal juga diuji. Model dasar tanpa reweighting menjadi acuan terpilih. Ambang yang dipilih pada validasi sebagai oracle tidak dilaporkan sebagai strategi yang dapat dicapai.')
    note('t2_validasi_model_dasar.csv','t2_fitur_seleksi.csv','t2_imbalance_ringkas.csv')

    page('8. Fraud: kalibrasi, biaya, dan drift')
    selected=csv('t2_kalibrasi_pilihan.csv').iloc[0]
    costs=csv('t2_kalibrasi_ringkas.csv')
    view=costs[((costs.model=='-')) | ((costs.model=='dasar') & costs.pendekatan.str.startswith('b:') & costs.kalibrasi.isin(['tanpa','Platt','isotonik']) & costs.aturan.eq('prob x nominal > 25.000'))]
    table(view,{'kalibrasi':'Kalibrasi','aturan':'Aturan','biaya_juta':'Biaya juta Rp','pr_auc':'PR-AUC','brier':'Brier','log_loss':'Log loss'},[1,2.2,1,.9,.9,.9])
    p('Brier score dan log loss sebelum/sesudah kalibrasi: baris "tanpa" adalah probabilitas mentah, baris Platt dan isotonik sesudah kalibrasi pada fold yang sama. Platt menurunkan keduanya; isotonik menurunkan Brier tetapi menaikkan log loss, karena tangga isotonik memberi probabilitas sangat kecil pada sebagian fraud. Ini salah satu alasan Platt lebih disukai selain kesederhanaannya.')
    p(f"Pilihan adalah Platt dan aturan prob x nominal > 25.000; biaya pengembangan rata-rata {selected.biaya_juta:.4f} juta rupiah per fold/seed. Flag semua menanggung biaya seluruh transaksi, sedangkan flag tidak ada menanggung semua nominal fraud. Nilai ini berasal dari label validasi, bukan biaya test.")
    p('Isotonik mempunyai biaya rata-rata lebih rendah, tetapi aturan pemilihan kode memilih pipeline lebih sederhana di antara kandidat yang berada dalam satu galat baku berpasangan dari termurah. Ukuran tersebut merupakan heuristik pengembangan atas fold/seed yang dipakai bersama, bukan bukti kesetaraan statistik dari sampel independen.')
    p('Platt dipelajari pada periode kalibrasi setelah fit awal. Pendekatan refit memakai model lebih baru dengan transformasi Platt dari model sebelumnya; transfer ini diuji dalam skema validasi, tetapi kalibrasi probabilitas model final tetap tidak dijamin. Grafik reliabilitas, Brier, log loss, dan ECE digunakan sebagai diagnostik tambahan.')
    fig('t2_reliabilitas.png',245)
    p('Bobot waktu dan jendela train pendek diuji untuk drift tetapi tidak memenuhi aturan pemilihan. Lonjakan fraud QRIS pada awal Agustus menandakan pola baru. Eksperimen jeda lebih panjang menunjukkan kinerja dapat turun; September-Oktober memiliki horizon lebih jauh daripada fold awal. Tidak memakai label test untuk beradaptasi.')
    heading('Adversarial validation dan pentingnya fitur antarperiode')
    adv=csv('t2_adversarial.csv')
    table(adv,{'pembanding':'Data lama vs data uji','susunan':'Fitur','n_fitur':'Jumlah fitur','auc':'AUC pembeda'},[1.6,2.4,.8,.9])
    p('Model dilatih membedakan baris lama dari baris uji tanpa label fraud; AUC 0,5 berarti tidak terbedakan. AUC hampir 1 terutama karena kolom kumulatif (misalnya hari sejak merchant terlihat) yang memang naik seiring waktu, dan tingkat fraud historis merchant yang membeku di data uji. Setelah keduanya dikeluarkan, pembeda yang tersisa terutama v15. Membuang v15 menurunkan PR-AUC, sehingga kolom tetap dipakai dan pergeseran ini dicatat sebagai risiko.')
    imp=csv('t2_pentingnya_antarperiode.csv')
    rho=imp['Jan-Mar'].rank().corr(imp['Apr-Jun'].rank())
    berubah=imp.reindex(imp['selisih'].abs().sort_values(ascending=False).index).head(5)
    table(berubah,{'fitur':'Fitur','Jan-Mar':'Porsi gain Jan-Mar','Apr-Jun':'Porsi gain Apr-Jun','selisih':'Selisih'},[2.2,1,1,1])
    p(f"Pentingnya fitur model Jan-Mar dan Apr-Jun cukup stabil (korelasi peringkat Spearman {rho:.2f}). Perubahan terbesar ada pada tingkat fraud historis merchant, yang makin informatif setelah label terkumpul lebih banyak.")
    note('t2_kalibrasi_ringkas.csv','t2_kalibrasi_pilihan.csv','t2_pergeseran_ringkas.csv','t2_adversarial.csv','t2_pentingnya_antarperiode.csv')

    page('9. Fraud: error dan prediksi final')
    err=csv('t2_error_per_kanal.csv')
    table(err,{'grup':'Kanal','fraud':'Fraud','tertangkap':'Tertangkap','recall':'Recall','biaya_juta':'Biaya juta Rp','porsi_biaya':'Porsi biaya'},[1,1,1,1,1.2,1])
    p('Analisis menggabungkan validasi empat fold, seed 42. Kanal QRIS menjadi titik lemah dengan recall rendah dan kontribusi biaya besar. Tabel lain memecah kesalahan per kategori merchant, rentang nominal, dan pola fraud. Pola seperti rentetan transaksi cepat atau indikasi pengambilalihan akun dapat tumpang tindih; porsi biaya antar pola tidak dijumlahkan.')
    fig('t2_fraud_bulan_kanal.png',220)
    f=csv('t2_final_ringkas.csv').iloc[0]
    p(f"Model final LightGBM dilatih pada transaksi sebelum 11 Agustus. Platt memakai model sebelum 14 Juli dan periode kalibrasi 14 Juli-10 Agustus. Dari {int(f.transaksi):,} transaksi test, {int(f.ditandai):,} ditandai; probabilitas rata-rata {100*f.prob_rata2:.3f}%. Koefisien Platt a={f.platt_a:.4f}, b={f.platt_b:.4f}.")
    p('Perkiraan biaya dari probabilitas merupakan biaya harapan model dan bukan skor biaya sesungguhnya: label test tersembunyi. Fraud bernominal kecil tidak otomatis ditandai bila ekspektasi kerugiannya lebih kecil daripada biaya review. Asumsi probabilitas jujur, drift QRIS, dan keterlambatan label harus disampaikan dalam ujian lisan.')
    note('t2_error_per_kanal.csv','t2_error_per_kategori_merchant.csv','t2_error_per_pola.csv','t2_final_ringkas.csv')

    page('10. Clustering: preprocessing dan metode')
    p('Pelanggan dikelompokkan tanpa label. Pelanggan dengan frekuensi_90h=0 dan keranjang kosong dipisahkan menjadi segmen 0; clustering dilakukan pada pelanggan aktif. Ini mencegah indikator missing menjadi satu-satunya pembeda dua kelompok. Nol transaksi pada jendela 90 hari tidak membuktikan pelanggan belum pernah membeli sepanjang hidupnya.')
    p('Fitur numerik diimputasi median, dibatasi persentil 1-99, ditransformasi log1p bila miring, dan distandardisasi. Komposisi kategori/waktu menggunakan centered log-ratio (CLR) setelah pseudocount 0,0001 dan normalisasi jumlah. Kategori demografi di-one-hot. ID dan fitur CRM anonim tidak masuk representasi utama; fitur numerik dengan korelasi absolut di atas 0,9 disaring.')
    p('Blok perilaku, nilai belanja, demografi, dan komposisi waktu diberi skala berdasarkan energi blok; demografi berbobot 0,25. Ablation tiap kelompok mempertimbangkan silhouette dan stabilitas. Komposisi kategori tetap diuji pada tahap e serta digunakan untuk profil, walau tidak masuk konfigurasi final.')
    heading('Eksplorasi a-d dan penguatan e-g')
    p('MiniBatch K-Means, GMM diagonal, dan HDBSCAN dibandingkan pada sampel bersama. K-Means/GMM menguji k=2-12; silhouette dan Davies-Bouldin dibandingkan, BIC hanya dalam keluarga GMM. Kriteria tidak sepenuhnya sepakat pada tahap awal. Partisi sementara a-d dipertahankan hanya sebagai pembanding historis, bukan submission.')
    ab=csv('task3_ad_revisi/ablation_kelompok_fitur.csv')
    table(ab,{'perubahan':'Ablation awal','silhouette':'Silhouette','stabilitas_ari_3_run':'ARI 3 run'},[3,1,1])
    p('Tiga run awal terlalu sedikit untuk mengunci keputusan kategori. Tahap e memperluas subsampling dan memakai K-Means penuh, sehingga hasilnya tidak dibandingkan sebagai efek kategori saja jika algoritme atau ukuran fit juga berubah. Preprocessing dibekukan pada seluruh data tanpa label; stabilitas yang dilaporkan bersyarat pada representasi tersebut.')
    note('task3_ad_revisi/ablation_kelompok_fitur.csv','task3_ad_revisi/perbandingan_metode_k.csv','task3_eg/keputusan_kategori.csv')

    page('11. Clustering: jumlah segmen dan stabilitas')
    grid=csv('task3_eg/stabilitas_per_k.csv')
    table(grid[grid.representasi=='utama'],{'k':'k aktif','silhouette':'Silhouette','davies_bouldin':'DB','calinski_harabasz':'CH','ari_rata2':'ARI rata-rata'},[.5,1,1,1.1,1])
    p('Grid menggunakan pool 20.000 pelanggan aktif, 20 subsampel 80%, dan prediksi seluruh pelanggan aktif untuk ARI antar-run. k yang stabil disaring terlebih dahulu; silhouette, DB, dan CH masing-masing memberi suara. Kriteria memilih k=2 aktif. Jika tidak ada k stabil, fallback memilih ARI tertinggi; kondisi fallback tidak terjadi pada hasil ini.')
    methods=csv('task3_eg/stabilitas_metode.csv')
    methods['metode']=methods.metode.str.replace(' (min_cluster_size=50, min_samples=5)','',regex=False)
    table(methods,{'metode':'Metode','n_run_valid':'Valid','n_run_gagal':'Gagal','ari_rata2':'ARI mean','ari_sb':'SD ARI'},[2.5,.6,.6,1,1])
    p('Perbandingan metode menggunakan 30 subsampel 80% pelanggan aktif; HDBSCAN + KNN memakai 20 run dengan sampel lebih kecil. Run gagal tidak menjadi partisi palsu dan dicatat terpisah. ARI HDBSCAN dihitung bersyarat pada run valid serta mencakup penyebaran KNN, bukan HDBSCAN murni.')
    cat=csv('task3_eg/keputusan_kategori.csv').iloc[0]
    p(f"Komposisi kategori: kenaikan ARI {cat.selisih_dengan_minus_tanpa:.6f}, SE jackknife berpasangan {cat.galat_baku_jackknife:.6f}. Blok dipakai hanya bila kenaikan >=0,01 dan >2 SE. Ambang praktis ditambahkan setelah review, bukan klaim praregistrasi. Keputusan akhir tetap tanpa blok kategori. ARI mengukur kestabilan, bukan kesesuaian dengan referensi dosen.")
    note('task3_eg/stabilitas_per_k.csv','task3_eg/stabilitas_metode.csv','task3_eg/keputusan_kategori.csv')

    page('12. Clustering: profil, aksi, dan anomali')
    profiles=csv('task3_eg/profil_segmen.csv')
    table(profiles,{'cluster':'Label','nama':'Nama usulan','pelanggan':'Jumlah','median_belanja_90h':'Median belanja Rp','median_persen_voucher':'Voucher','kategori_dominan':'Kategori dominan','jam_dominan':'Jam dominan'},[.5,1.8,.9,1.1,.8,1.2,.8])
    p('Label aktif diurutkan berdasarkan median belanja sehingga nomor tidak bergantung pada permutasi label model. Nama adalah interpretasi kelompok atas profil, bukan label ground truth. Premium berbelanja lebih besar, memakai voucher lebih sedikit dan lebih dominan pagi. Pemburu promo lebih dominan malam dan lebih sering memakai voucher. Kategori khas dihitung dari lift terhadap pelanggan aktif.')
    table(pd.DataFrame([
        {'segmen':'Premium Loyal','aksi':'Program VIP dan akses awal produk; uji biaya/manfaat sebelum subsidi ongkir.'},
        {'segmen':'Pemburu Promo Malam','aksi':'Flash sale/voucher terbatas malam hari; uji respons dan margin.'},
        {'segmen':'Tidak bertransaksi 90 hari','aksi':'Penawaran aktivasi atau reaktivasi. Cek riwayat sebelum menyebut kupon pembelian pertama.'},
        {'segmen':'Anomali','aksi':'Tinjau kualitas data/perilaku sebelum memasukkan kampanye massal.'}]),{'segmen':'Segmen','aksi':'Aksi yang diusulkan'},[1.4,3.6])
    an=csv('task3_eg/anomali_ringkas.csv').iloc[0]
    p(f"Jarak pelanggan aktif ke centroid di atas persentil 98 diberi -1: {int(an.anomali):,} pelanggan ({an.persen_semua_pelanggan:.2f}% seluruh pelanggan). Irisan dengan 2% log-likelihood GMM terendah adalah {100*an.irisan_dengan_loglik_gmm_terendah:.2f}%. Dari {int(an.rasio_konversi_di_atas_1_aktif):,} pelanggan aktif dengan rasio_konversi >1, {int(an.rasio_konversi_di_atas_1_yang_anomali)} ditandai sebagai anomali jarak.")
    p('Ambang jarak adalah aturan deteksi kandidat anomali, bukan bukti fraud atau kesalahan data. Pelanggan tanpa transaksi dipisahkan sebelum deteksi anomali aktif. Kualitas pemisahan internal masih moderat; beberapa k lain juga stabil. Jumlah segmen referensi dan ARI hidden tidak diketahui.')
    note('task3_eg/profil_segmen.csv','task3_eg/profil_kategori.csv','task3_eg/anomali_ringkas.csv')

    page('13. Reproduksi, kontribusi, dan batasan')
    heading('Satu perintah dari data mentah')
    p('Setelah memasang requirements.txt dan menaruh lima dataset .csv.gz di data/, jalankan:\npython kode/run_all.py --task all\nUrutan: Task 1, Task 2, Task 3 a-d lalu e-g, cek format, kemudian pembentukan laporan.pdf. Run penuh Task 2 memerlukan beberapa jam; perintah ini ditujukan untuk reproduksi penuh, bukan demonstrasi lima menit.')
    p('Untuk ujian lisan: python kode/task2_klasifikasi.py --tahap final mereproduksi prediksi Task 2. Pilihan Task 1: jalankan cell baseline setelah setup/definisi pada notebook. Task 3: tampilkan profil serta jelaskan CLR, pemilihan k, ARI, dan anomali. Ketiga cara harus dipahami kedua anggota, bukan hanya penulis task.')
    heading('Lingkungan dan bukti')
    p('Seed utama 42. Task 2 memakai beberapa seed pada pembanding; Task 3 mengunci empat thread. Versi dependensi dipin di requirements.txt; metadata setiap eksperimen mencatat konfigurasi. Mesin/versi BLAS dapat memengaruhi digit akhir, sehingga perbandingan numerik memakai toleransi yang dijelaskan. Hash input/output serta sumber tabel laporan dicatat pada manifest_laporan.json.')
    p('Pengecekan akhir mencakup tes temporal dan kategori, kalibrasi, biaya, ARI, format output, dan ID terhadap dataset. Hasil tes bukan bukti semua bug mustahil. Riwayat review telah mereproduksi prediksi Task 1/2 dan label final Task 3. Catatan verifikasi integrasi memisahkan run yang benar-benar diulang dari artefak yang berasal dari run terdahulu.')
    heading('Kontribusi dan penggunaan AI')
    p('Habibie: Task 1 (PR #2), Task 3 a-d (PR #6), review/integrasi. Azmi: Task 2 (PR #4, #5, #7-#10), Task 3 e-g (PR #11), review silang. Codex membantu penulisan/debug, audit, eksperimen, tes, dan laporan; bantuan Claude Code dilaporkan pada PR Azmi. Penggunaan AI dan persetujuan anggota dirinci di kontribusi.md; persetujuan baru tidak diasumsikan dari approval PR kode sebelumnya.')
    heading('Referensi implementasi')
    p('Dokumen soal dan kamus_data.md resmi dosen; dokumentasi metode yang dirujuk dalam kode: LightGBM Advanced Topics, scikit-learn (quantile loss, metrics, clustering dan partial dependence), Angelopoulos & Bates, A Gentle Introduction to Conformal Prediction (arXiv:2107.07511). Tidak menggunakan dataset eksternal. Hasil SHAP/PDP dan profil merupakan penjelasan model, bukan hubungan sebab-akibat.','SmallNote')
    p('Laporan ini menggunakan hasil pengembangan yang sudah diperiksa selama pengerjaan. Pemilihan berulang pada periode/fold yang sama membatasi klaim generalisasi. Tidak ada klaim skor hidden test maupun klaim persetujuan anggota yang belum diberikan.','SmallNote')

    dest=ROOT/'laporan.pdf'
    doc=SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=43,bottomMargin=56,title='UTS Machine Learning - Kelompok 11',author='Muhammad Habibie Rabbani; Azmi Aziz syah Putra')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    if doc.page > 18:
        raise ValueError(f'Laporan melebihi batas 18 halaman: {doc.page}')
    manifest={'jumlah_halaman':doc.page,'sumber':{s:hash_source(ROOT/s) for s in sorted(sources)},'normalisasi_hash':'Teks CSV/JSON/MD/PY/TXT memakai LF; gambar dan PDF memakai byte asli.','sha256_pdf':hashlib.sha256(dest.read_bytes()).hexdigest(),'generator':'kode/susun_laporan_kelompok.py','catatan':'Dibangun dari artefak; membangun laporan tidak sama dengan mengulang semua eksperimen.'}
    (H/'manifest_laporan.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(f'Laporan: {dest}')

if __name__=='__main__': build()
