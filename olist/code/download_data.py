"""Загрузка двух связанных публичных выпусков Olist с проверкой файлов.
Геолокацию не загружаем в рабочие таблицы, она для заданий не нужна.
"""
from common import *
import hashlib,io,urllib.request,zipfile
sources=json.loads((ROOT/'sources.json').read_text())
DATA.mkdir(exist_ok=True)
for source in sources['releases']:
    with urllib.request.urlopen(source['download']) as response:blob=response.read()
    if hashlib.sha256(blob).hexdigest()!=source['archive_sha256']:
        raise ValueError('Архив изменился. Сверьте версию с sources.json перед запуском.')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for filename in source['files']:
            content=z.read(filename)
            if hashlib.sha256(content).hexdigest()!=source['files'][filename]['sha256']:raise ValueError(filename)
            (DATA/filename).write_bytes(content)
    print('Downloaded',source['title'])
