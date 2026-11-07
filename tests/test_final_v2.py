"""Behavioral tests; small synthetic fixtures are never experimental evidence."""
import copy
import json
import shutil

import numpy as np
import optuna
import pandas as pd
import pytest

from src import run_experiments as runner
from src.analyze_results import AP, SIZE, STD, load_audited, pareto, rank_candidates, validate_aggregate
from src.pipeline import build_hpo_pipeline


def config():
    return runner.load_yaml_mapping('configs/final.yaml')


def fake_evaluate(params, cfg, context, directory, prefix, repeat=False, on_fold=None):
    folds = [{'fold': i, 'average_precision': .3 + i * .01, 'model_size_bytes': i * 100000,
              'fit_seconds': .01, 'predict_seconds': .001, 'serialization_seconds': .001}
             for i in [1, 2, 3]]
    return {'folds': folds, AP: .32, STD: .01, SIZE: .3, 'model_size_max_bytes': 300000}


def fake_context():
    return {'hashes': {'dataset_sha256': 'test', 'split_sha256': 'test', 'folds_sha256': 'test'}}


def test_exactly_four_parameters_and_fixed_none(synthetic_bank_frame):
    cfg = config()
    study = optuna.create_study()
    trial = study.ask()
    params = runner.suggest_parameters(trial, cfg['search_space'])
    assert set(trial.params) == set(runner.PARAMETERS)
    X, _ = runner.prepare_training_features(synthetic_bank_frame)
    model = build_hpo_pipeline(X, cfg, {**params, 'class_weight': 'balanced'})
    assert model['model'].class_weight is None
    assert model['model'].random_state == 42 and model['model'].n_jobs == 1
    with pytest.raises(ValueError, match='string'):
        build_hpo_pipeline(X, cfg, params, {'class_weight': 'None'})


def test_seed_isolation_idempotence_and_protocol_mismatch(tmp_path):
    cfg, context = config(), fake_context()
    for seed in [42, 52]:
        runner.run_study(cfg, context, tmp_path, seed, 'random', 2, 'smoke', fake_evaluate)
    paths = [tmp_path / f'seed_{s}/random/trials.json' for s in [42, 52]]
    before = [p.read_bytes() for p in paths]
    runner.run_study(cfg, context, tmp_path, 42, 'random', 2, 'smoke', fake_evaluate)
    assert before == [p.read_bytes() for p in paths]
    assert json.loads(before[0])[0]['config_hash'] != json.loads(before[1])[0]['config_hash']
    cfg['selection_tolerance'] = .1
    with pytest.raises(ValueError, match='mismatch'):
        runner.run_study(cfg, context, tmp_path, 42, 'random', 2, 'smoke', fake_evaluate)


def test_interruption_resume_preserves_sampler_sequence(tmp_path):
    cfg, context = config(), fake_context()
    calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return fake_evaluate(*args)
    for sampler in ['random', 'tpe']:
        calls = 0
        partial = tmp_path / ('partial_' + sampler)
        full = tmp_path / ('full_' + sampler)
        with pytest.raises(KeyboardInterrupt):
            runner.run_study(cfg, context, partial, 42, sampler, 3, 'smoke', interrupted)
        runner.run_study(cfg, context, partial, 42, sampler, 3, 'smoke', fake_evaluate)
        runner.run_study(cfg, context, full, 42, sampler, 3, 'smoke', fake_evaluate)
        p = runner.read_json(partial / f'seed_42/{sampler}/trials.json')
        f = runner.read_json(full / f'seed_42/{sampler}/trials.json')
        assert [r['config_hash'] for r in p] == [r['config_hash'] for r in f]
        assert [r['status'] for r in p] == ['COMPLETE', 'FAIL', 'COMPLETE']
        assert 'Interrupted' in p[1]['error']


def test_failures_consume_budget(tmp_path):
    def fail(*args):
        raise RuntimeError('test failure')
    runner.run_study(config(), fake_context(), tmp_path, 42, 'random', 2, 'smoke', fail)
    rows = runner.read_json(tmp_path / 'seed_42/random/trials.json')
    assert len(rows) == 2 and all(r['status'] == 'FAIL' for r in rows)
    assert all('test failure' in r['error'] for r in rows)


def test_fold_aggregates_are_recomputed():
    record = fake_evaluate(None, None, None, None, None)
    folds = pd.DataFrame(record.pop('folds'))
    validate_aggregate(record, folds, 3)
    record[SIZE] = .2
    with pytest.raises(ValueError, match='aggregate'):
        validate_aggregate(record, folds, 3)


def candidate(ap, size, key, **kwargs):
    return {AP: ap, SIZE: size, STD: .01, 'model_size_max_bytes': int(size * 1e6),
            'status': 'COMPLETE', 'feasible': True, 'config_hash': key, 'sampler': 'random',
            'trial': 0, 'search_seed': 42, **kwargs}


def test_pareto_unique_and_locked_tie_break():
    frame = pd.DataFrame([candidate(.44, 5, 'a'), candidate(.437, 1, 'b'),
                          candidate(.437, 1, 'b'), candidate(.43, 2, 'c'),
                          candidate(.9, .1, 'bad', feasible=False),
                          candidate(.437, 1, 'd', **{STD: .001})])
    assert set(pareto(frame).config_hash) == {'a', 'b', 'd'}
    ranked = rank_candidates(frame, .005)
    assert list(ranked.config_hash) == ['d', 'b', 'a']
    assert rank_candidates(frame, 0).iloc[0].config_hash == 'a'


def test_train_only_loader_skips_holdout(tmp_path, synthetic_bank_frame):
    frame = synthetic_bank_frame.copy()
    frame.loc[89, 'job'] = 'HOLDOUT_SENTINEL'
    path = tmp_path / 'data.csv'
    frame.to_csv(path, sep=';', index=False)
    audit = {'sha256': runner.sha256_file(path), 'shape': list(frame.shape)}
    loaded = runner.load_frozen_training_rows(path, np.arange(89), audit)
    assert len(loaded) == 89 and not loaded.job.eq('HOLDOUT_SENTINEL').any()


def test_refit_enforces_full_train_size(tmp_path, synthetic_bank_frame):
    cfg = config()
    cfg['size_cap_mb'] = .000001
    X, y = runner.prepare_training_features(synthetic_bank_frame)
    params = dict(n_estimators=2, max_depth=4, min_samples_leaf=1, max_features=.5)
    result = runner.refit(params, cfg, {'X': X, 'y': y}, tmp_path / 'model.joblib')
    assert not result['feasible'] and result['full_train_size_bytes'] > 1


def test_full_train_fallback_never_expands_near_best_set(tmp_path):
    from src.finalize_experiments import select_with_refit, local_values
    params = dict(n_estimators=100, max_depth=10, min_samples_leaf=5, max_features=.7)
    frame = pd.DataFrame([{**candidate(.44, 1, 'a'), **params},
                          {**candidate(.438, 2, 'b'), **params, 'n_estimators': 125},
                          {**candidate(.42, .5, 'outside'), **params}])
    called = []
    def refitter(p, cfg, ctx, artifact):
        called.append(p['n_estimators'])
        return {'feasible': len(called) == 2, 'full_train_size_mb': 60 if len(called) == 1 else 3}
    selected = select_with_refit(rank_candidates(frame, .005), config(), {}, tmp_path, refitter)
    assert selected['candidate']['config_hash'] == 'b' and called == [100, 125]
    values = local_values(params, config()['search_space'])
    assert all(len(v) <= 3 and params[k] in v for k, v in values.items())


def test_export_analysis_without_database_and_reject_tampering(tmp_path):
    cfg = config()
    cfg.update(search_seeds=[42], trials_per_sampler=2)
    context = fake_context()
    runner.atomic_json(cfg, tmp_path / 'protocol_snapshot.json')
    base = fake_evaluate(None, None, None, None, None)
    base['fingerprint'] = runner.fingerprint(cfg, context['hashes'])
    runner.atomic_json(base, tmp_path / 'baseline.json')
    for sampler in cfg['samplers']:
        runner.run_study(cfg, context, tmp_path, 42, sampler, 2, 'full', fake_evaluate)
    _, rows, audit = load_audited(tmp_path)
    assert len(rows) == 4 and all(a['sqlite'].startswith('PASS') for a in audit)
    clone = tmp_path / 'export_only'
    shutil.copytree(tmp_path, clone, ignore=shutil.ignore_patterns('*.db', '*.pkl', 'export_only'))
    _, _, audit = load_audited(clone)
    assert all(a['sqlite'].startswith('not checked') for a in audit)
    csv = tmp_path / 'seed_42/random/trials.csv'
    csv.write_text(csv.read_text() + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_audited(tmp_path)


def test_report_rejects_partial_runs_and_changed_figures(tmp_path):
    from scripts.build_final_report import audited_sources, digest
    cfg = config()
    cfg.update(search_seeds=[42], samplers=['random'], trials_per_sampler=2)
    root = tmp_path
    final = root / 'results/final'
    analysis = root / 'results/analysis/final'
    runner.atomic_json(cfg, final / 'protocol_snapshot.json')
    manifest = {'status': 'PARTIAL', 'attempted': 2}
    runner.atomic_json(manifest, analysis / 'analysis_manifest.json')
    with pytest.raises(ValueError, match='complete registered'):
        audited_sources(root)
    study_path = final / 'seed_42/random/manifest.json'
    runner.atomic_json({'files': {}}, study_path)
    figure_path = root / 'figures/final/pareto.png'
    figure_path.parent.mkdir(parents=True)
    figure_path.write_bytes(b'test-only corrupted image')
    manifest.update(status='COMPLETE', protocol=cfg['protocol'],
                    runs=[{'search_seed': 42, 'sampler': 'random', 'manifest_sha256': digest(study_path)}],
                    tables={}, figures={'pareto.png': 'wrong-hash'})
    runner.atomic_json(manifest, analysis / 'analysis_manifest.json')
    with pytest.raises(ValueError, match='Figure hash mismatch'):
        audited_sources(root)
