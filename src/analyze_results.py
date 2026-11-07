"""Audit CSV/JSON exports and derive v2 tables without data, models or SQLite."""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import sha256_file
from .pareto import pareto_front_ap_model_size
from .paths import resolve_project_path
from .run_experiments import PARAMETERS, atomic_json, canonical_hash, read_json

AP = 'average_precision_cv_mean'
SIZE = 'model_size_max_mb'
STD = 'average_precision_cv_std'


def eligible(frame):
    return frame.loc[frame.status.eq('COMPLETE') & frame.feasible.eq(True)].copy()


def rank_candidates(frame, tolerance):
    candidates = eligible(frame)
    if candidates.empty:
        raise ValueError('No complete feasible candidate')
    candidates = candidates.loc[candidates[AP] >= candidates[AP].max() - tolerance]
    return candidates.sort_values(
        ['model_size_max_bytes', AP, STD, 'config_hash', 'sampler', 'trial', 'search_seed'],
        ascending=[True, False, True, True, True, True, True],
    ).drop_duplicates('config_hash')


def pareto(frame):
    return pareto_front_ap_model_size(frame.rename(columns={'status': 'state'})).drop_duplicates('config_hash')


def validate_aggregate(record, folds, expected_folds):
    if len(folds) != expected_folds or sorted(folds['fold']) != list(range(1, expected_folds + 1)):
        raise ValueError('Missing/duplicate folds')
    ap = folds.average_precision.to_numpy()
    sizes = folds.model_size_bytes.to_numpy()
    if not np.all(np.isfinite(ap)) or not np.all((ap >= 0) & (ap <= 1)) or not np.all(sizes > 0):
        raise ValueError('Invalid fold metrics')
    expected = {AP: ap.mean(), STD: ap.std(ddof=1), SIZE: sizes.max() / 1e6,
                'model_size_max_bytes': sizes.max()}
    for key, value in expected.items():
        if not np.isclose(float(record[key]), value, rtol=1e-12, atol=1e-12):
            raise ValueError(f'Fold aggregate mismatch: {key}')


def load_audited(root, allow_partial=False):
    config = read_json(root / 'protocol_snapshot.json')
    all_rows, audit = [], []
    fingerprints = set()
    for seed in config['search_seeds']:
        for sampler in config['samplers']:
            directory = root / f'seed_{seed}' / sampler
            if not (directory / 'manifest.json').exists():
                if allow_partial:
                    continue
                raise ValueError(f'Missing study: seed {seed} {sampler}')
            manifest = read_json(directory / 'manifest.json')
            if manifest['mode'] != 'full' or read_json(directory / 'protocol_snapshot.json') != config:
                raise ValueError('Wrong run/protocol, or smoke mixed into final')
            if manifest['budget'] != config['trials_per_sampler']:
                raise ValueError('Budget differs from registered protocol')
            fingerprints.add(manifest['fingerprint'])
            for name, expected in manifest['files'].items():
                if sha256_file(directory / name) != expected:
                    raise ValueError(f'Export hash mismatch: {directory / name}')
            rows = read_json(directory / 'trials.json')
            csv = pd.read_csv(directory / 'trials.csv', float_precision='round_trip')
            frame = pd.DataFrame(rows)
            pd.testing.assert_frame_equal(frame, csv[frame.columns], check_dtype=False,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
            folds = pd.read_csv(directory / 'folds.csv', float_precision='round_trip')
            if len(rows) != manifest['attempted'] or len(rows) > config['trials_per_sampler']:
                raise ValueError('Attempted count mismatch')
            if not allow_partial and len(rows) != config['trials_per_sampler']:
                raise ValueError('Full budget not complete')
            if sorted(frame.trial) != list(range(len(frame))) or not frame.status.isin(['COMPLETE', 'FAIL']).all():
                raise ValueError('Trial sequence is incomplete')
            expected_id = f"{config['protocol']}_full_seed{seed}_{sampler}"
            for row in rows:
                if row['run_id'] != expected_id or row['search_seed'] != seed or row['sampler'] != sampler:
                    raise ValueError('Run identity mismatch')
                if row['model_seed'] != 42 or row['class_weight'] is not None:
                    raise ValueError('Fixed parameter mismatch')
                params = {p: row[p] for p in PARAMETERS}
                if canonical_hash(params) != row['config_hash']:
                    raise ValueError('Configuration hash mismatch')
                for p, v in params.items():
                    spec = config['search_space'][p]
                    if not spec['low'] <= v <= spec['high']:
                        raise ValueError('Out-of-domain parameter')
                    if spec['type'] == 'int' and (v - spec['low']) % spec.get('step', 1):
                        raise ValueError('Invalid integer step')
                for key, value in manifest['hashes'].items():
                    if row[key] != value:
                        raise ValueError('Input hash mismatch')
                if row['status'] == 'COMPLETE':
                    subset = folds.loc[folds.trial.eq(row['trial'])]
                    validate_aggregate(row, subset, config['split']['cv_folds'])
                    if not np.isclose(row['g'], row[SIZE] - config['size_cap_mb'], atol=1e-12):
                        raise ValueError('Constraint mismatch')
                    if row['feasible'] != (row['g'] <= 0):
                        raise ValueError('Feasibility mismatch')
                elif not row['error']:
                    raise ValueError('Failed trial requires an error reason')
            sqlite_status = 'not checked (database absent)'
            db = directory / 'study.db'
            if db.exists():
                with closing(sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)) as connection:
                    stored = connection.execute('SELECT number, state FROM trials ORDER BY number').fetchall()
                    if stored != [(r['trial'], r['status']) for r in rows]:
                        raise ValueError('SQLite state mismatch')
                    values = connection.execute('SELECT t.number, v.objective, v.value FROM trial_values v JOIN trials t ON t.trial_id=v.trial_id ORDER BY t.number, v.objective').fetchall()
                    expected = [(r['trial'], i, r[k]) for r in rows if r['status'] == 'COMPLETE' for i, k in enumerate([AP, SIZE])]
                    if len(values) != len(expected) or not np.allclose(values, expected, rtol=1e-12, atol=1e-12):
                        raise ValueError('SQLite objective mismatch')
                sqlite_status = 'PASS (trial states and objectives)'
            audit.append({'search_seed': seed, 'sampler': sampler, 'csv_json_folds': 'PASS',
                          'sqlite': sqlite_status, 'manifest_sha256': sha256_file(directory / 'manifest.json'),
                          'wall_seconds': sum(s['wall_seconds'] for s in manifest['sessions'])})
            all_rows.extend(rows)
    if len(fingerprints) != 1:
        raise ValueError('Mixed fingerprints or no runs')
    baseline = read_json(root / 'baseline.json')
    if baseline['fingerprint'] not in fingerprints:
        raise ValueError('Baseline and search fingerprints differ')
    validate_aggregate(baseline, pd.DataFrame(baseline['folds']), config['split']['cv_folds'])
    return config, pd.DataFrame(all_rows), audit


def summarize(frame, config, audits):
    summary, budgets, convergence, startup = [], [], [], []
    for (seed, sampler), group in frame.groupby(['search_seed', 'sampler']):
        good = eligible(group)
        ranked = rank_candidates(group, config['selection_tolerance']) if len(good) else None
        selected = ranked.iloc[0] if ranked is not None else None
        summary.append({'search_seed': seed, 'sampler': sampler, 'attempted': len(group),
                        'complete': int(group.status.eq('COMPLETE').sum()), 'failed': int(group.status.eq('FAIL').sum()),
                        'best_ap': good[AP].max(), 'selected_ap': selected[AP] if selected is not None else None,
                        'selected_size_mb': selected[SIZE] if selected is not None else None,
                        'selected_config_hash': selected.config_hash if selected is not None else None,
                        'pareto_distinct': len(pareto(group)), 'total_trial_seconds': group.runtime_seconds.sum(),
                        'wall_seconds': next(a['wall_seconds'] for a in audits if a['search_seed'] == seed and a['sampler'] == sampler)})
        for budget in range(1, config['trials_per_sampler'] + 1):
            if budget > len(group):
                break
            best = good.loc[good.trial.lt(budget), AP].max()
            convergence.append({'search_seed': seed, 'sampler': sampler, 'budget': budget, 'best_ap': best})
            if budget in config['comparison_budgets']:
                budgets.append(convergence[-1])
    for seed, group in frame.groupby('search_seed'):
        r = group.loc[group.sampler.eq('random')].sort_values('trial').head(config['tpe_startup_trials'])
        t = group.loc[group.sampler.eq('tpe')].sort_values('trial').head(config['tpe_startup_trials'])
        matched = sum(a == b for a, b in zip(r.config_hash, t.config_hash))
        startup.append({'search_seed': seed, 'compared': min(len(r), len(t)), 'identical_startup_configs': matched})
    summary = pd.DataFrame(summary)
    aggregate = summary.groupby('sampler').agg(
        runs=('best_ap', 'count'), best_ap_mean=('best_ap', 'mean'), best_ap_std=('best_ap', 'std'),
        selected_ap_mean=('selected_ap', 'mean'), selected_ap_std=('selected_ap', 'std'),
        selected_size_mean=('selected_size_mb', 'mean'), selected_size_std=('selected_size_mb', 'std'),
        trial_seconds_mean=('total_trial_seconds', 'mean'), trial_seconds_std=('total_trial_seconds', 'std'),
    ).reset_index()
    paired = summary.pivot(index='search_seed', columns='sampler', values='best_ap').reindex(columns=['random', 'tpe'])
    paired['tpe_minus_random'] = paired.tpe - paired.random
    return {'run_summary': summary, 'sampler_summary': aggregate, 'paired_differences': paired.reset_index(),
            'budget_summary': pd.DataFrame(budgets), 'convergence': pd.DataFrame(convergence),
            'startup_duplicates': pd.DataFrame(startup), 'pareto': pareto(frame),
            'ranked_candidates': rank_candidates(frame, config['selection_tolerance'])}


def figures(tables, frame, baseline, directory, selected_hash=None):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sampler, group in tables['convergence'].groupby('sampler'):
        stats = group.groupby('budget').best_ap.agg(['mean', 'std'])
        x = stats.index.to_numpy()
        ax.plot(x, stats['mean'], label=sampler.upper())
        ax.fill_between(x, stats['mean'] - stats['std'], stats['mean'] + stats['std'], alpha=.18)
    ax.axhline(baseline[AP], color='gray', ls='--', label='Baseline CV')
    ax.set(xlabel='Attempted trial budget', ylabel='Best feasible CV AP', title='Mean and sample SD across search seeds')
    ax.legend(); fig.tight_layout(); fig.savefig(directory / 'convergence.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sampler, group in eligible(frame).groupby('sampler'):
        ax.scatter(group[SIZE], group[AP], s=13, alpha=.45, label=sampler.upper())
    front = tables['pareto'].sort_values(SIZE)
    ax.plot(front[SIZE], front[AP], color='black', lw=1, label='Distinct pooled Pareto')
    selected = tables['ranked_candidates'].iloc[0] if selected_hash is None else frame.loc[frame.config_hash.eq(selected_hash)].iloc[0]
    ax.scatter([selected[SIZE]], [selected[AP]], marker='*', s=160, color='red', label='Selected by CV rule')
    ax.scatter([baseline[SIZE]], [baseline[AP]], marker='D', color='gray', label='Baseline CV')
    ax.set(xlabel='Maximum fold pipeline size (decimal MB)', ylabel='Mean 3-fold CV AP')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(directory / 'pareto.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    paired = tables['paired_differences']
    for _, row in paired.iterrows():
        ax.plot([0, 1], [row['random'], row['tpe']], marker='o', label=f"Seed {int(row.search_seed)}")
    ax.set_xticks([0, 1], ['Random', 'TPE']); ax.set_ylabel('Best feasible CV AP at full budget')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(directory / 'seed_distribution.png'); plt.close(fig)


def analyze(root, output, figure_dir, allow_partial=False):
    config, frame, audit = load_audited(root, allow_partial)
    tables = summarize(frame, config, audit)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    selected_path = root / 'selected_candidate.json'
    selected_hash = read_json(selected_path)['candidate']['config_hash'] if selected_path.exists() else None
    figures(tables, frame, read_json(root / 'baseline.json'), figure_dir, selected_hash)
    atomic_json({'status': 'PARTIAL' if allow_partial else 'COMPLETE', 'source_root': str(root),
                 'protocol': config['protocol'], 'runs': audit, 'attempted': len(frame),
                 'complete': int(frame.status.eq('COMPLETE').sum()), 'failed': int(frame.status.eq('FAIL').sum()),
                 'baseline_sha256': sha256_file(root / 'baseline.json'),
                 'tables': {p.name: sha256_file(p) for p in output.glob('*.csv')},
                 'figures': {p.name: sha256_file(p) for p in figure_dir.glob('*.png')}}, output / 'analysis_manifest.json')
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/final')
    parser.add_argument('--output', default='results/analysis/final')
    parser.add_argument('--figures', default='figures/final')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    analyze(resolve_project_path(args.input), resolve_project_path(args.output), resolve_project_path(args.figures), args.allow_partial)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
