"""Пересчитать лабораторные. --generate также заново запускает открытую LLM."""
from pathlib import Path
import subprocess
import sys

CODE=Path(__file__).resolve().parent
for n in ['01','02','04','05','06','07','08','09','10']:
    args=[sys.executable,str(CODE/f'lab{n}.py')]
    if n=='09' and '--generate' not in sys.argv:
        args.append('--prepare')
    print(f'\nЛабораторная {n}',flush=True)
    subprocess.run(args,check=True)
print('\nГотово. JSON и CSV находятся в results/.')
