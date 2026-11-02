"""Prepare the user-downloaded Kaggle ZIP and verify analysed CSV versions.
Usage: python code/prepare_data.py --archive /path/to/archive.zip
"""
from pathlib import Path
from zipfile import ZipFile
import argparse,json,hashlib,shutil

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
args=parser.parse_args()
manifest=json.loads((root/'sources.json').read_text())['journey_files']
(root/'data').mkdir(exist_ok=True)
with ZipFile(args.archive) as source:
    by_name={Path(n).name:n for n in source.namelist()}
    for name,meta in manifest.items():
        target=root/'data'/name
        with source.open(by_name[name]) as a,target.open('wb') as b:shutil.copyfileobj(a,b)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=meta['sha256']:
            raise ValueError('Unexpected file version: '+name)
print('Verified',len(manifest),'CSV files from the two-year Kaggle release')
