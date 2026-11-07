from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from src.data import sha256_file
from src.week4_analysis import (
    PROJECT_ROOT,
    analysis_tables,
    frozen_train_eda_tables,
    load_frozen_training_frame,
    load_sources,
    validate_sources,
)


@pytest.mark.skipif(
    not all((PROJECT_ROOT / 'results/hpo/ap_model_size_v1/full_run_01' /
             f'{sampler}_seed42_ap_model_size_v1.db').is_file() for sampler in ['random', 'tpe']),
    reason='Historical SQLite audit not checked: DBs are local-only; obtain the v1 evidence package',
)
def test_week4_official_sources_are_consistent() -> None:
    report = validate_sources(load_sources())
    assert report["status"] == "PASS"
    assert report["sqlite_checks"]["random"]["trial_count"] == 40
    assert report["sqlite_checks"]["tpe"]["trial_count"] == 40
    assert report["official_pareto_hash_count"] == 13


@pytest.mark.skipif(
    not (PROJECT_ROOT / 'data/raw/bank-full.csv').is_file(),
    reason='Historical train EDA requires bank-full.csv; run scripts/download_data.py',
)
def test_week4_eda_materializes_only_frozen_train_without_duration() -> None:
    frame = load_frozen_training_frame()
    summary, numeric, categorical = frozen_train_eda_tables(frame)
    assert len(frame) == 36_168
    assert "duration" not in frame.columns
    assert summary["scope"] == "frozen_training_rows_only"
    assert summary["target_counts"] == {"no": 31_937, "yes": 4_231}
    assert not numeric.empty
    assert not categorical.empty


def test_week4_tables_distinguish_best_and_selected() -> None:
    sources = load_sources()
    tables = analysis_tables(sources)
    key = tables["key_models"].set_index("model")
    assert key.loc["Best Tpe", "config_hash"] == "cd972e72ea0a"
    assert key.loc["Selected TPE", "config_hash"] == "4cdd6dd35f1d"
    assert key.loc["Best Tpe", "cv_ap_mean"] > key.loc["Selected TPE", "cv_ap_mean"]
    assert key.loc["Selected TPE", "cv_max_fold_size_mb"] < key.loc["Best Tpe", "cv_max_fold_size_mb"]
    assert key.loc["Selected TPE", "full_train_size_mb"] == 1.28934


def test_week4_dominance_and_near_best_counts() -> None:
    tables = analysis_tables(load_sources())
    assert int(tables["dominance"]["dominated"].sum()) == 67
    assert tables["dominance"]["config_hash"].nunique() == 70
    assert tables["near_best"]["config_hash"].nunique() == 18
    assert tables["near_best"].iloc[0]["config_hash"] == "4cdd6dd35f1d"


def test_week4_figures_are_png_and_at_least_300_dpi() -> None:
    figure_dir = PROJECT_ROOT / "figures/week4"
    paths = sorted(figure_dir.glob("*.png"))
    assert len(paths) == 7
    for path in paths:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            assert dpi[0] >= 299 and dpi[1] >= 299


def test_week4_eda_figures_use_locked_train_and_match_manifest() -> None:
    manifest_path = PROJECT_ROOT / "results/analysis/week4/eda_figure_manifest.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["scope"] == "frozen_training_rows_only"
    assert report["training_samples"] == 36_168
    assert report["statistics"]["target_counts"] == {"no": 31_937, "yes": 4_231}
    assert report["dataset_sha256"] == "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"
    assert report["split_artifact_sha256"] == "583acb9dbb4ee92852b1d0a9b584cffb2f7b1fc6409dfdb495a46df4cb3d7d1a"
    assert report["train_indices_sha256"] == "b078ac348228f8e49c30af463dc1198fb5e8e11d6379e5d842caf580fc728469"
    assert report["duration_materialized"] is False
    assert report["holdout_materialized"] is False
    assert report["training_performed"] is False
    assert report["hpo_performed"] is False

    expected = {
        "01_target_distribution_train.png": ["y"],
        "02_age_distribution_train.png": ["age"],
        "03_balance_distribution_train.png": ["balance"],
    }
    assert set(report["figures"]) == set(expected)
    for name, source_columns in expected.items():
        path = PROJECT_ROOT / "figures/week4/eda" / name
        metadata = report["figures"][name]
        assert path.is_file()
        assert metadata["sha256"] == sha256_file(path)
        assert metadata["source_columns"] == source_columns
        assert metadata["source_scope"] == "frozen_training_rows_only"
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            assert dpi[0] >= 319 and dpi[1] >= 319


def test_dataset_eda_notebook_references_reproducible_figures() -> None:
    notebook_path = PROJECT_ROOT / "notebooks/01_Dataset_EDA.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(str(cell.get("source", "")) for cell in payload["cells"])
    assert "build_week4_eda_outputs" in source
    assert "assert len(train_only) == 36_168" in source
    for filename in (
        "01_target_distribution_train.png",
        "02_age_distribution_train.png",
        "03_balance_distribution_train.png",
    ):
        assert f"../figures/week4/eda/{filename}" in source


def test_week4_notebooks_are_valid_json_with_safety_metadata() -> None:
    notebook_dir = PROJECT_ROOT / "notebooks"
    paths = sorted(notebook_dir.glob("0[1-5]_*.ipynb"))
    assert len(paths) == 5
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["metadata"]["week4_policy"] == {
            "read_only_analysis": True,
            "holdout_used": False,
            "training_performed": False,
            "hpo_performed": False,
        }
        assert any(cell["cell_type"] == "markdown" for cell in payload["cells"])
        assert any(cell["cell_type"] == "code" for cell in payload["cells"])


def test_week4_notebook_execution_report_passes() -> None:
    path = PROJECT_ROOT / "results/analysis/week4/notebook_validation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["executed"] is True
    assert len(report["notebooks"]) == 5
    assert all(item["status"] == "PASS" for item in report["notebooks"])
    assert report["holdout_used"] is False

