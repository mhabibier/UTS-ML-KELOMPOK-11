"""Kontrak integrasi: output final mendahului pemeriksaan dan pembuatan laporan."""
from pathlib import Path
from unittest.mock import patch
import subprocess
import pytest
import run_all
import cek_format

def test_task3_berakhir_dengan_eg(tmp_path):
    calls=[]
    with patch('sys.argv',['run_all.py','--task','3','--data-dir',str(tmp_path)]), patch.object(run_all.subprocess,'run',side_effect=lambda cmd,**kw:calls.append(cmd)):
        run_all.main()
    assert [Path(c[1]).name for c in calls]==['task3_clustering.py','task3_lanjutan.py']
    assert all(c[-1]==str(tmp_path.resolve()) for c in calls)

def test_laporan_hanya_dibuat_setelah_semua_output_valid(tmp_path):
    calls=[]
    with patch('sys.argv',['run_all.py','--task','all','--data-dir',str(tmp_path)]), patch.object(run_all.subprocess,'run',side_effect=lambda cmd,**kw:calls.append(cmd)), patch.object(cek_format,'periksa_semua',return_value=(0,['valid'])) as check:
        run_all.main()
    assert [c[c.index('--task')+1] for c in calls[:3]]==['1','2','3']
    assert Path(calls[-1][1]).name=='susun_laporan_kelompok.py'
    assert check.call_args.args[1]==tmp_path.resolve()
    calls.clear()
    with patch('sys.argv',['run_all.py','--task','all']), patch.object(run_all.subprocess,'run',side_effect=lambda cmd,**kw:calls.append(cmd)), patch.object(cek_format,'periksa_semua',return_value=(1,['invalid'])):
        with pytest.raises(SystemExit) as exc: run_all.main()
    assert exc.value.code==1
    assert all(Path(c[1]).name!='susun_laporan_kelompok.py' for c in calls)

def test_kegagalan_task_menghentikan_pipeline():
    calls=[]
    def fail(cmd,**kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1,cmd)
    with patch('sys.argv',['run_all.py','--task','all']), patch.object(run_all.subprocess,'run',side_effect=fail):
        with pytest.raises(subprocess.CalledProcessError): run_all.main()
    assert len(calls)==1
