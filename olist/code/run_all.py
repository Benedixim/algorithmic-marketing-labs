"""Запустить все расчёты. Генерацию текстов запускают отдельно lab09.py."""
from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
for lab in ['01','02','04','05','06','07','08','09','10']:
    args=[sys.executable,str(HERE/f'lab{lab}.py')]
    if lab=='09':args.append('--prepare')
    subprocess.run(args,check=True)
