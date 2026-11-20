"""Cek artefak dan kesesuaian laporan; --siap-kumpul juga mewajibkan persetujuan anggota."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cek_format import periksa_semua
from susun_laporan_kelompok import hash_source

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--siap-kumpul',action='store_true')
    args=parser.parse_args()
    code,lines=periksa_semua(ROOT,args.data_dir)
    print('\n'.join(lines))
    if code: raise SystemExit(code)
    for name,cols in [('prediksi_regresi.csv',['p50','p90']),('prediksi_klasifikasi.csv',['prob'])]:
        values=pd.read_csv(ROOT/name,usecols=cols).to_numpy(dtype=float)
        if not np.isfinite(values).all(): raise ValueError(f'{name}: nilai tidak finite')
    for name in ['laporan.pdf','kontribusi.md','kode/README.md']:
        if not (ROOT/name).is_file(): raise FileNotFoundError(name)
    manifest=json.loads((ROOT/'kode/hasil/manifest_laporan.json').read_text(encoding='utf8'))
    assert 0 < manifest['jumlah_halaman'] <= 18
    assert hashlib.sha256((ROOT/'laporan.pdf').read_bytes()).hexdigest()==manifest['sha256_pdf'], 'PDF berubah setelah manifest dibuat'
    for name,digest in manifest['sumber'].items():
        assert hash_source(ROOT/name)==digest, f'Sumber berubah; bangun ulang PDF: {name}'
    print(f"[LOLOS] Laporan {manifest['jumlah_halaman']} halaman; hash PDF dan sumber angka cocok")
    pending='- [ ]' in (ROOT/'kontribusi.md').read_text(encoding='utf8')
    if pending:
        print('[MENUNGGU] Persetujuan laporan/kontribusi kedua anggota belum lengkap')
        if args.siap_kumpul: raise SystemExit(1)
    else:
        print('[LOLOS] Checklist persetujuan anggota lengkap; periksa tautan bukti dan status main sebelum tenggat')

if __name__=='__main__': main()
