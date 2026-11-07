"""Download the exact UCI bank-full.csv and verify the frozen SHA-256."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip'


def main():
    target = ROOT / 'data/raw/bank-full.csv'
    expected = json.loads((ROOT / 'results/data_audit.json').read_text(encoding='utf-8'))['sha256']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError('Existing data differs from frozen checksum; refusing overwrite')
        print('Existing bank-full.csv checksum PASS')
        return
    with urllib.request.urlopen(URL, timeout=90) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        nested = next(n for n in archive.namelist() if n.split('/')[-1] == 'bank.zip')
        with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as bank:
            name = next(n for n in bank.namelist() if n.split('/')[-1] == 'bank-full.csv')
            data = bank.read(name)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Downloaded dataset checksum differs; no data written')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f'Downloaded bank-full.csv; SHA-256 PASS: {expected}')


if __name__ == '__main__':
    main()
