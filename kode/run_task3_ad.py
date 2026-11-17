"""Reproduksi Task 3 a-d dari notebook yang sama dengan Run All."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/'kode/Task3_Clustering.ipynb').read_text(encoding='utf-8'))
    import os
    os.chdir(root)
    scope={'__name__':'__main__','DATA_DIR':args.data_dir}
    start=time.monotonic()
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            print(f'Cell {i}',flush=True)
            exec(compile(''.join(c['source']),f'Task3:cell-{i}','exec'),scope)
    print(f'Selesai dalam {time.monotonic()-start:.1f} detik',flush=True)

if __name__=='__main__':
    main()
