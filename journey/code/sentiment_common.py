"""Read the original pseudo-XML without extracting or publishing reviewer identities."""
from pathlib import Path
import re
import html
import tarfile
import json
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
DOMAINS = {'books': 'Книги', 'dvd': 'Фильмы на DVD',
           'electronics': 'Электроника', 'kitchen_&_housewares': 'Товары для кухни'}

def save(name, value):
    (RESULTS / (name + '.json')).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def reviews():
    archive = ROOT / 'data/domain_sentiment_data.tar.gz'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected = 'e65d4dae3f223316a5db426b2c3f7f3667e9b165f03c024cb4875e11af4cd941'
    if digest != expected:
        raise ValueError('Unexpected archive checksum')
    rows = []
    file_counts = {}
    fields = ['asin', 'product_name', 'rating', 'title', 'date', 'review_text', 'helpful']
    with tarfile.open(archive, 'r:gz') as source:
        for member in source.getmembers():
            if not member.isfile() or not member.name.endswith('.review'):
                continue
            domain = member.name.split('/')[-2]
            subset = Path(member.name).stem
            text = source.extractfile(member).read().decode('utf8', errors='replace')
            records = re.findall(r'<review>(.*?)</review>', text, re.S)
            file_counts[member.name] = len(records)
            for record in records:
                row = {'domain': domain, 'subset': subset}
                for field in fields:
                    match = re.search(r'<' + field + r'>(.*?)</' + field + r'>', record, re.S)
                    row[field] = html.unescape(match.group(1).strip()) if match else ''
                rows.append(row)
    data = pd.DataFrame(rows)
    data['rating'] = pd.to_numeric(data.rating, errors='coerce')
    data['date'] = pd.to_datetime(data.date, format='mixed', errors='coerce')
    data['norm'] = data.review_text.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    save('archive_audit', {
        'sha256': digest, 'raw_records': len(data), 'files': file_counts,
        'domains': {k: int(v) for k, v in data.domain.value_counts().items()},
        'missing': {k: int(data[k].isna().sum() + data[k].eq('').sum()) for k in fields},
        'duplicate_text_rows': int(data.norm.duplicated().sum()),
        'rating_values': {str(k): int(v) for k, v in data.rating.value_counts(dropna=False).items()},
        'fields_retained': fields,
        'excluded_identity_fields': ['reviewer', 'reviewer_location', 'unique_id'],
    })
    return data
