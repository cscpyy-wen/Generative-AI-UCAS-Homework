"""Verify every published checkpoint and reference result without downloading data."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'results/manifest.json').read_text())
    count = 0
    for name, record in manifest['reference_files'].items():
        path = root / name
        assert path.is_file(), name
        assert path.stat().st_size == record['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], name
        count += 1
    for name, record in manifest['checkpoints'].items():
        path = root / 'checkpoints' / (name + '.pt')
        assert path.stat().st_size == record['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], name
        count += 1
    summary = json.loads((root / 'results/summary.json').read_text())
    assert len(summary['experiments']) == 7
    assert sum(r['completed_epochs'] for r in summary['experiments']) == 442
    assert summary['selected'] == 'f02'
    print('Reference files and seven checkpoints verified:', count)


if __name__ == '__main__':main()
