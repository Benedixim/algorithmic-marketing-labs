"""Download the original four-domain release from the authors' official page."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz'
SHA256 = 'e65d4dae3f223316a5db426b2c3f7f3667e9b165f03c024cb4875e11af4cd941'

def main():
    target = ROOT / 'data' / 'domain_sentiment_data.tar.gz'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix('.download')
        with urlopen(URL, timeout=90) as source, temporary.open('wb') as out:
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(target)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError('Archive checksum differs from the analysed release')
    print('Official archive verified', target.stat().st_size, 'bytes')

if __name__ == '__main__':
    main()
