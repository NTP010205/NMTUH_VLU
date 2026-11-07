"""Lock pooled selection, audit full-train size, and run local sensitivity."""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .analyze_results import AP, SIZE, analyze, load_audited, rank_candidates
from .paths import resolve_project_path
from .run_experiments import (PARAMETERS, atomic_json, canonical_hash, evaluate, fingerprint,
                              load_context, read_json, refit, sha256_file)


def parameters(row):
    return {p: int(row[p]) if p != 'max_features' else float(row[p]) for p in PARAMETERS}


def local_values(params, space):
    result = {}
    for name in PARAMETERS:
        spec = space[name]
        value = params[name]
        delta = spec.get('step', 1) if spec['type'] == 'int' else .1
        values = [max(spec['low'], value - delta), value, min(spec['high'], value + delta)]
        result[name] = sorted(set(values))
    return result


def select_with_refit(ranked, cfg, context, root, refitter=refit):
    audits = []
    for _, row in ranked.iterrows():
        params = parameters(row)
        result = refitter(params, cfg, context, root / 'selected_model.joblib')
        audits.append({'config_hash': row.config_hash, **result})
        if result['feasible']:
            return {'candidate': row.to_dict(), 'parameters': params, 'full_train': result,
                    'refit_attempts': audits, 'pooled_budget': int(cfg['trials_per_sampler'] * len(cfg['samplers']) * len(cfg['search_seeds'])),
                    'tolerance': cfg['selection_tolerance'], 'selection_scope': 'pooled complete feasible trials',
                    'holdout_used': False}
    atomic_json({'status': 'NO_FEASIBLE_FULL_TRAIN_MODEL', 'refit_attempts': audits}, root / 'refit_failure.json')
    raise ValueError('No near-best candidate passes full-train size cap; tolerance is unchanged')


def finalize(root, sensitivity_dir, figure_dir):
    cfg, frame, _ = load_audited(root)
    context = load_context(cfg)
    fp = fingerprint(cfg, context['hashes'])
    if read_json(root / 'baseline.json')['fingerprint'] != fp:
        raise ValueError('Finalization environment/code differs from experiment')
    ranked = rank_candidates(frame, cfg['selection_tolerance'])
    selection_path = root / 'selected_candidate.json'
    sources = {str(p.relative_to(root)): sha256_file(p) for p in root.glob('seed_*/*/manifest.json')}
    if selection_path.exists():
        selected = read_json(selection_path)
        if selected['source_manifests'] != sources or selected['fingerprint'] != fp:
            raise ValueError('Locked selection source mismatch')
    else:
        selected = select_with_refit(ranked, cfg, context, root)
        selected.update(source_manifests=sources, fingerprint=fp)
        atomic_json(selected, selection_path)
    base_refit = root / 'baseline_full_train.json'
    if not base_refit.exists():
        params = {p: cfg['baseline'][p] for p in PARAMETERS}
        atomic_json({'fingerprint': fp, **refit(params, cfg, context, root / 'baseline_model.joblib')}, base_refit)
    elif read_json(base_refit)['fingerprint'] != fp:
        raise ValueError('Baseline refit fingerprint mismatch')
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    plan = {'selected_hash': selected['candidate']['config_hash'], 'fingerprint': fp,
            'values': local_values(selected['parameters'], cfg['search_space']),
            'selection_must_not_change': True}
    plan_path = sensitivity_dir / 'protocol_snapshot.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Sensitivity plan mismatch')
    atomic_json(plan, plan_path)
    rows = []
    for name, values in plan['values'].items():
        for value in values:
            params = {**selected['parameters'], name: value}
            key = canonical_hash(params)
            result_path = sensitivity_dir / f'{key}.json'
            if result_path.exists():
                result = read_json(result_path)
                if result['fingerprint'] != fp or result['parameters'] != params:
                    raise ValueError('Sensitivity result fingerprint mismatch')
            else:
                started = time.perf_counter()
                metrics = evaluate(params, cfg, context, sensitivity_dir, key)
                result = {'parameters': params, 'fingerprint': fp, 'config_hash': key, **metrics,
                          'runtime_seconds': time.perf_counter() - started}
                atomic_json(result, result_path)
            rows.append({'variable': name, 'value': value, 'config_hash': key,
                         'is_selected': value == selected['parameters'][name],
                         **{k: result[k] for k in [AP, SIZE, 'average_precision_cv_std', 'runtime_seconds']}})
    table = pd.DataFrame(rows)
    table.to_csv(sensitivity_dir / 'summary.csv', index=False)
    atomic_json({'fingerprint': fp, 'selected_hash': selected['candidate']['config_hash'],
                 'unique_evaluations': table.config_hash.nunique(),
                 'files': {p.name: sha256_file(p) for p in sensitivity_dir.glob('*.json') if p.name != 'manifest.json'},
                 'summary_sha256': sha256_file(sensitivity_dir / 'summary.csv')}, sensitivity_dir / 'manifest.json')
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for (name, group), ax in zip(table.groupby('variable', sort=False), axes.flat):
        ax.errorbar(group.value, group[AP], yerr=group.average_precision_cv_std, marker='o', capsize=3)
        ax.axvline(selected['parameters'][name], color='gray', ls='--')
        ax.set(xlabel=name, ylabel='CV AP mean and fold SD')
    fig.tight_layout(); fig.savefig(figure_dir / 'sensitivity.png', dpi=170); plt.close(fig)
    analyze(root, resolve_project_path('results/analysis/final'), figure_dir)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/final')
    parser.add_argument('--sensitivity', default='results/sensitivity')
    parser.add_argument('--figures', default='figures/final')
    args = parser.parse_args(argv)
    finalize(resolve_project_path(args.input), resolve_project_path(args.sensitivity), resolve_project_path(args.figures))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
