"""Peluncur opsional: seluruh logika Task 1 berada di notebook."""
import argparse
import hashlib
import json
import time
import subprocess
import sys
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['1', '2', '3', 'all'], default='1')
    parser.add_argument('--tahap', default='semua', help='Tahap Task 2; final untuk reproduksi singkat setelah PR final digabung.')
    parser.add_argument('--mode', choices=['full', 'tests', 'report'], default='full')
    parser.add_argument('--data-dir', type=Path, default=repo / 'data')
    args = parser.parse_args()
    if args.task == 'all':
        required = ['Task1_Regresi.ipynb', 'task2_klasifikasi.py', 'task3_clustering.py']
        missing = [p for p in required if not (repo/'kode'/p).is_file()]
        if missing:
            parser.error(f'Gabungkan PR terkait dahulu; kode belum tersedia: {missing}')
        for task in ['1', '2', '3']:
            subprocess.run([sys.executable, __file__, '--task', task, '--data-dir', str(args.data_dir.resolve()), '--mode', args.mode, '--tahap', args.tahap], check=True)
        return
    if args.task in ('2', '3'):
        if args.mode != 'full':
            parser.error('--mode tests/report khusus Task 1.')
        script = repo/'kode'/('task2_klasifikasi.py' if args.task=='2' else 'task3_clustering.py')
        if not script.is_file():
            parser.error(f'Gabungkan PR terkait dahulu: {script.name} belum tersedia.')
        options = ['--berkas', str(args.data_dir.resolve()/'klasifikasi_train.csv.gz'), '--berkas-uji', str(args.data_dir.resolve()/'klasifikasi_test.csv.gz'), '--tahap', args.tahap] if args.task=='2' else ['--data-dir', str(args.data_dir.resolve())]
        subprocess.run([sys.executable, str(script), *options], check=True)
        return
    path = repo / 'kode/Task1_Regresi.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    tags = {'setup', 'definitions'}
    if args.mode in ('full', 'tests'):
        tags.update(['test_definitions', 'tests'])
    if args.mode in ('full', 'report'):
        tags.add('report')
    if args.mode == 'full':
        tags.update(['stage', 'preview'])
    scope = {'__name__': '__main__', 'PROJECT_DIR': repo, 'DATA_DIR': args.data_dir.resolve(), 'RUN_MODE': args.mode}
    start = time.monotonic()
    manifest = {'status': 'berjalan', 'mode': args.mode,
                'sumber': path.name, 'sha256_notebook': hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and tags.intersection(cell['metadata'].get('tags', [])):
                print(f"Cell {i}: {cell['metadata']['tags']}", flush=True)
                exec(compile(''.join(cell['source']), f'{path.name}:cell-{i}', 'exec'), scope)
        manifest['status'] = 'sukses'
    except BaseException as exc:
        manifest.update(status='gagal', error=str(exc))
        raise
    finally:
        manifest['durasi_detik'] = round(time.monotonic() - start, 3)
        dest = repo / 'kode/hasil/reproduksi_task1'
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f'manifest_notebook_{args.mode}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
