"""Verify every preserved historical artifact against the pre-work snapshot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    reference = ROOT / 'docs/review/v1_sha256_before.json'
    before = json.loads(reference.read_text(encoding='utf-8-sig'))
    after, missing, changed = {}, [], []
    for name, digest in before.items():
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        after[name] = actual
        if actual != digest:
            changed.append(name)
    report = {'status': 'FAIL' if changed else ('INCOMPLETE' if missing else 'PASS'),
              'checked': len(after), 'missing_not_checked': missing, 'changed': changed}
    (ROOT / 'docs/review/v1_sha256_after.json').write_text(json.dumps(after, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'docs/review/v1_integrity.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    return 1 if changed else 0


if __name__ == '__main__':
    raise SystemExit(main())
