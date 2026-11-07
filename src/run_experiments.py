"""Four-parameter v2 experiments. Historical files are read-only inputs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import sha256_file, write_json
from .paths import PROJECT_ROOT, resolve_project_path
from .pipeline import build_hpo_pipeline
from .optimize_ap_model_size import (
    build_fixed_folds, canonical_hash, evaluate_configuration_cv,
    load_frozen_training_rows, load_yaml_mapping, prepare_training_features,
    validate_frozen_folds,
)

PARAMETERS = ('n_estimators', 'max_depth', 'min_samples_leaf', 'max_features')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    write_json(value, temp)
    os.replace(temp, path)


def environment():
    return {
        'python': sys.version, 'os': platform.platform(),
        'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
        'packages': {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()},
    }


def load_context(config):
    audit = read_json(resolve_project_path(config['data_audit_reference']))
    split_path = resolve_project_path(config['split_indices_reference'])
    with np.load(split_path) as stored:
        train = stored['train_indices']
        test = stored['test_indices']
    if len(train) != 36168 or len(test) != 9043 or np.intersect1d(train, test).size:
        raise ValueError('Frozen split is invalid')
    frame = load_frozen_training_rows(resolve_project_path(config['data_path']), train, audit)
    X, y = prepare_training_features(frame)
    folds = build_fixed_folds(X, y, config)
    fold_audit = validate_frozen_folds(resolve_project_path(config['cv_folds_reference']), folds, train)
    hashes = {'dataset_sha256': audit['sha256'], 'split_sha256': sha256_file(split_path),
              'folds_sha256': fold_audit['sha256']}
    unknown = {c: int(frame[c].eq('unknown').sum()) for c in frame.select_dtypes('object')}
    return {'X': X, 'y': y, 'folds': folds, 'fold_metadata': fold_audit['folds'],
            'hashes': hashes, 'train_unknown_counts': unknown}


def suggest_parameters(trial, space):
    if set(space) != set(PARAMETERS):
        raise ValueError('Exactly four search parameters required')
    params = {}
    for name in PARAMETERS:
        spec = space[name]
        if spec['type'] == 'int':
            params[name] = trial.suggest_int(name, spec['low'], spec['high'], step=spec.get('step', 1))
        else:
            params[name] = trial.suggest_float(name, spec['low'], spec['high'])
    return params


def evaluate(params, config, context, directory, prefix, repeat=False, on_fold=None):
    return evaluate_configuration_cv(
        X_train=context['X'], y_train=context['y'], folds=context['folds'],
        fold_metadata=context['fold_metadata'], base_config=config, params=params,
        artifact_dir=directory / 'temporary_models', artifact_prefix=prefix,
        compression=config['compression'], thread_limit=config['thread_limit'],
        verify_repeat_serialization=repeat, on_fold_complete=on_fold,
    )


def fingerprint(config, hashes):
    # Include implementation and environment: changing either requires a fresh output.
    code = {p: sha256_file(PROJECT_ROOT / p) for p in
            ['src/run_experiments.py', 'src/pipeline.py', 'src/optimize_ap_model_size.py', 'src/data.py']}
    return canonical_hash({'config': config, 'inputs': hashes, 'code': code, 'environment': environment()})


def constraints(trial):
    return (trial.user_attrs.get('g', float('inf')),)


def save_sampler(sampler, path, count, fp):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        pickle.dump({'sampler': sampler, 'count': count, 'fingerprint': fp}, handle)
    os.replace(temp, path)


def records_from_study(study, config, context, seed, sampler_name, run_id):
    rows, folds = [], []
    for trial in study.trials:
        attrs = trial.user_attrs
        row = {'run_id': run_id, 'protocol': config['protocol'], 'search_seed': seed,
               'model_seed': config['baseline']['random_state'], 'sampler': sampler_name,
               'trial': trial.number, 'status': trial.state.name, 'class_weight': None,
               'config_hash': attrs.get('config_hash'), **context['hashes'],
               **trial.params, 'error': attrs.get('error'),
               'runtime_seconds': attrs.get('runtime_seconds'),
               'g': attrs.get('g'), 'feasible': attrs.get('g', float('inf')) <= 0}
        result = attrs.get('result', {})
        row.update({k: v for k, v in result.items() if k != 'folds'})
        rows.append(row)
        for fold in result.get('folds', attrs.get('partial_folds', [])):
            folds.append({'run_id': run_id, 'search_seed': seed, 'sampler': sampler_name,
                          'trial': trial.number, 'config_hash': attrs.get('config_hash'), **fold})
    return rows, folds


def export_study(study, config, context, seed, sampler_name, directory, manifest):
    rows, folds = records_from_study(study, config, context, seed, sampler_name, manifest['run_id'])
    atomic_json(rows, directory / 'trials.json')
    pd.DataFrame(rows).to_csv(directory / 'trials.csv', index=False)
    pd.DataFrame(folds).to_csv(directory / 'folds.csv', index=False)
    manifest['attempted'] = len(rows)
    manifest['complete'] = sum(r['status'] == 'COMPLETE' for r in rows)
    manifest['failed'] = sum(r['status'] == 'FAIL' for r in rows)
    manifest['total_trial_seconds'] = sum(r['runtime_seconds'] or 0 for r in rows)
    manifest['files'] = {n: sha256_file(directory / n) for n in ['trials.json', 'trials.csv', 'folds.csv']}
    atomic_json(manifest, directory / 'manifest.json')


def run_study(config, context, root, seed, sampler_name, budget, mode, evaluator=evaluate):
    directory = root / f'seed_{seed}' / sampler_name
    directory.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(config, context['hashes'])
    run_id = f"{config['protocol']}_{mode}_seed{seed}_{sampler_name}"
    db = directory / 'study.db'
    checkpoint = directory / 'sampler.pkl'
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest['fingerprint'], manifest['budget'], manifest['run_id']) != (fp, budget, run_id):
            raise ValueError('Resume fingerprint/protocol/budget mismatch')
        if not db.exists():
            raise ValueError('Resume requires the local SQLite DB; exported logs support analysis only')
    else:
        if db.exists():
            raise ValueError('Unregistered database; refusing resume')
        manifest = {'run_id': run_id, 'fingerprint': fp, 'budget': budget, 'mode': mode,
                    'environment': environment(), 'hashes': context['hashes'],
                    'started_utc': datetime.now(timezone.utc).isoformat(), 'sessions': []}
        atomic_json(manifest, manifest_path)
        atomic_json(config, directory / 'protocol_snapshot.json')
    sampler = (optuna.samplers.RandomSampler(seed=seed) if sampler_name == 'random' else
               optuna.samplers.TPESampler(seed=seed, n_startup_trials=config['tpe_startup_trials'],
                                          constraints_func=constraints))
    study = optuna.create_study(study_name=run_id, storage='sqlite:///' + db.as_posix(),
                                sampler=sampler, directions=config['directions'], load_if_exists=True)
    if study.user_attrs.get('fingerprint', fp) != fp:
        raise ValueError('SQLite fingerprint mismatch')
    study.set_user_attr('fingerprint', fp)
    if len(study.trials) > budget:
        raise ValueError('Database exceeds attempted-trial budget')
    if len(study.trials) == budget and all(t.state.is_finished() for t in study.trials):
        export_study(study, config, context, seed, sampler_name, directory, manifest)
        return
    if study.trials:
        if not checkpoint.exists():
            raise ValueError('Missing sampler checkpoint; cannot safely resume RNG')
        # Local checkpoint created by this runner only. Never load untrusted pickles.
        with checkpoint.open('rb') as handle:
            saved = pickle.load(handle)
        if saved['count'] != len(study.trials) or saved['fingerprint'] != fp:
            raise ValueError('Sampler checkpoint mismatch; retain evidence and use a fresh run')
        sampler = saved['sampler']
        study.sampler = sampler
        for t in study.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                live = optuna.Trial(study, t._trial_id)
                live.set_user_attr('error', 'Interrupted process; counted as attempted, not retried')
                study.tell(live, state=optuna.trial.TrialState.FAIL)
    started = time.perf_counter()
    session = {'started_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': 0.0}
    manifest['sessions'].append(session)
    try:
        while len(study.trials) < budget:
            trial = study.ask()
            params = suggest_parameters(trial, config['search_space'])
            trial.set_user_attr('config_hash', canonical_hash(params))
            save_sampler(sampler, checkpoint, len(study.trials), fp)
            trial_started = time.perf_counter()
            try:
                result = evaluator(params, config, context, directory, f'trial_{trial.number}', mode == 'smoke',
                                   lambda records: trial.set_user_attr('partial_folds', records))
                trial.set_user_attr('result', result)
                trial.set_user_attr('g', result['model_size_max_mb'] - config['size_cap_mb'])
                trial.set_user_attr('runtime_seconds', time.perf_counter() - trial_started)
                study.tell(trial, [result['average_precision_cv_mean'], result['model_size_max_mb']])
            except Exception as exc:
                trial.set_user_attr('error', f'{type(exc).__name__}: {exc}')
                trial.set_user_attr('runtime_seconds', time.perf_counter() - trial_started)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            save_sampler(sampler, checkpoint, len(study.trials), fp)
            session['wall_seconds'] = time.perf_counter() - started
            export_study(study, config, context, seed, sampler_name, directory, manifest)
            print(f'{run_id}: {len(study.trials)}/{budget}', flush=True)
    finally:
        session['wall_seconds'] = time.perf_counter() - started
        export_study(study, config, context, seed, sampler_name, directory, manifest)


def baseline(config, context, root):
    path = root / 'baseline.json'
    fp = fingerprint(config, context['hashes'])
    if path.exists():
        value = read_json(path)
        if value['fingerprint'] != fp:
            raise ValueError('Baseline fingerprint mismatch')
        return value
    params = {p: config['baseline'][p] for p in PARAMETERS}
    started = time.perf_counter()
    result = evaluate(params, config, context, root, 'baseline')
    value = {'fingerprint': fp, 'parameters': params, 'fixed_parameters': config['fixed_parameters'],
             **result, 'runtime_seconds': time.perf_counter() - started}
    atomic_json(value, path)
    return value


def refit(params, config, context, artifact):
    model = build_hpo_pipeline(context['X'], config, params)
    started = time.perf_counter()
    with threadpool_limits(limits=config['thread_limit']):
        model.fit(context['X'], context['y'])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact, compress=config['compression'])
    size = artifact.stat().st_size
    return {'full_train_size_bytes': size, 'full_train_size_mb': size / 1e6,
            'feasible': size / 1e6 <= config['size_cap_mb'], 'sha256': sha256_file(artifact),
            'runtime_seconds': time.perf_counter() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/final.yaml')
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--output')
    parser.add_argument('--seed', type=int, help='Run one registered seed, preserving the full protocol')
    args = parser.parse_args(argv)
    config = load_yaml_mapping(args.config)
    if config['fixed_parameters'] != {'class_weight': None, 'random_state': 42, 'n_jobs': 1}:
        raise ValueError('Unexpected fixed parameters')
    context = load_context(config)
    root = resolve_project_path(args.output or ('results/smoke_v2' if args.mode == 'smoke' else 'results/final'))
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / 'protocol_snapshot.json'
    if snapshot.exists() and read_json(snapshot) != config:
        raise ValueError('Output protocol mismatch')
    atomic_json(config, snapshot)
    atomic_json({'hashes': context['hashes'], 'train_unknown_counts': context['train_unknown_counts'],
                 'train_rows': len(context['X']), 'train_positive': int(context['y'].sum()),
                 'environment': environment()}, root / 'context.json')
    baseline(config, context, root)
    seeds = config['search_seeds'] if args.mode == 'full' else config['search_seeds'][:1]
    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError('Seed not in registered protocol')
        seeds = [args.seed]
    budget = config['trials_per_sampler'] if args.mode == 'full' else config['smoke_trials_per_sampler']
    for seed in seeds:
        for sampler_name in config['samplers']:
            run_study(config, context, root, seed, sampler_name, budget, args.mode)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
