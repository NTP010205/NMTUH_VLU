"""Read experiment progress without importing training libraries or changing DBs."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / 'results/final'
    cfg = json.loads((root / 'protocol_snapshot.json').read_text(encoding='utf-8'))
    total_complete = total_failed = total_running = 0
    print('Seed  Sampler  COMPLETE  FAIL  RUNNING  Budget')
    for seed in cfg['search_seeds']:
        for sampler in cfg['samplers']:
            db = root / f'seed_{seed}' / sampler / 'study.db'
            counts = {}
            if db.exists():
                with closing(sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)) as connection:
                    counts = dict(connection.execute('SELECT state,COUNT(*) FROM trials GROUP BY state'))
            complete, failed, running = [counts.get(k, 0) for k in ['COMPLETE', 'FAIL', 'RUNNING']]
            total_complete += complete; total_failed += failed; total_running += running
            print(f'{seed:<5} {sampler:<8} {complete:<9} {failed:<5} {running:<8} {cfg["trials_per_sampler"]}')
    budget = len(cfg['search_seeds']) * len(cfg['samplers']) * cfg['trials_per_sampler']
    print(f'Finished attempted: {total_complete + total_failed}/{budget}; COMPLETE={total_complete}; FAIL={total_failed}; RUNNING={total_running}')
    print('Running trial may be live or interrupted; confirm worker state before starting another runner.')
    for name in ['selected_candidate.json', 'baseline_full_train.json']:
        print(f'{name}: {"present" if (root / name).exists() else "pending"}')
    print('DOCX:', 'present; inspect render QA evidence' if (ROOT / 'report/Bao_cao_NMTUH.docx').exists() else 'pending')


if __name__ == '__main__':
    main()
