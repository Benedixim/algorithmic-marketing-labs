"""Run every analysis. Full LLM generation is explicit with --generate."""
from pathlib import Path
import subprocess,sys
code=Path(__file__).resolve().parent
for lab in ['01','02','04','05','06','07','08','09','10']:
    args=[sys.executable,str(code/('lab'+lab+'.py'))]
    if lab=='09' and '--generate' not in sys.argv:args.append('--prepare')
    subprocess.run(args,check=True)
