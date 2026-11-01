"""Скачать открытые данные из sources.json и проверить SHA-256 каждого файла."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    sources = json.loads((ROOT / 'sources.json').read_text(encoding='utf-8'))
    for item in sources['data']:
        target = ROOT / item['file']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']:
            print(f'Уже загружен: {target.name}')
            continue
        print(f'Загрузка: {target.name}', flush=True)
        request = urllib.request.Request(item['source'], headers={'User-Agent': 'MarketingLabs/1.0'})
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
        if zipfile.is_zipfile(io.BytesIO(payload)):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                candidates = [n for n in archive.namelist() if Path(n).name == target.name]
                if len(candidates) != 1:
                    raise ValueError(f'Не удалось однозначно найти {target.name} в архиве')
                payload = archive.read(candidates[0])
        if hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError(f'Контрольная сумма не совпала: {target.name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        print(f'Сохранён: {target.name}')

if __name__ == '__main__':
    main()
