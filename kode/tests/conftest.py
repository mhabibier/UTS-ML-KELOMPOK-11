"""Supaya modul di folder kode/ (misalnya t2_data) bisa di-import oleh tes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
