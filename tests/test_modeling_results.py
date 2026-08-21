"""The machine-readable experiment record, and the holdout audit.

The audit tests are cheap and blunt on purpose: they read the Phase 5 source and
fail if any of it so much as mentions the holdout. A guarantee that depends on
nobody adding a line later is not a guarantee.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from churn.config import PROJECT_ROOT, get_config
from churn.modeling.evaluation import N_SPLITS, build_splitter, evaluate_models
from churn.modeling.metrics import METRIC_NAMES
from churn.modeling.models import LOGISTIC_REGRESSION, build_baselines
from churn.modeling.results import (
    SCHEMA_VERSION,
    BaselineResults,
    build_results,
    load_results,
    write_results,
)
from churn.preprocessing.contracts import FEATURE_COLUMNS, build_feature_matrix
from churn.preprocessing.target import encode_target

_FAKE_DIGEST = "0" * 64


@pytest.fixture
def experiment(contract_frame: pd.DataFrame) -> BaselineResults:
    """A complete record built from the synthetic frame, not the real dataset."""
    features = build_feature_matrix(contract_frame)
    target = encode_target(contract_frame["Churn"])
    models = build_baselines()
    evaluations = evaluate_models(models, features, target, build_splitter())
    return build_results(
        evaluations,
        models,
        np.asarray(target),
        raw_sha256=_FAKE_DIGEST,
        training_ids_sha256=_FAKE_DIGEST,
    )


def test_record_carries_the_provenance_chain(experiment: BaselineResults) -> None:
    assert experiment.schema_version == SCHEMA_VERSION
    assert experiment.raw_sha256 == _FAKE_DIGEST
    assert experiment.training_ids_sha256 == _FAKE_DIGEST
    assert experiment.random_seed == get_config().seed
    assert experiment.scikit_learn_version


def test_record_documents_the_cross_validation(experiment: BaselineResults) -> None:
    cv = experiment.cross_validation

    assert cv.strategy == "StratifiedKFold"
    assert cv.n_splits == N_SPLITS
    assert cv.shuffle is True
    assert cv.random_state == get_config().seed
    assert cv.shared_across_models is True


def test_metric_protocol_puts_average_precision_first(experiment: BaselineResults) -> None:
    """Minority positive class: AP leads, ROC-AUC is the secondary view."""
    protocol = experiment.metric_protocol

    assert protocol.primary_development_metric == "average_precision"
    assert protocol.secondary_ranking_metric == "roc_auc"
    assert protocol.diagnostic_metrics == ["precision", "recall", "f1"]
    assert protocol.auxiliary_metrics == ["accuracy"]
    assert protocol.threshold_optimised is False


def test_metric_protocol_names_only_known_metrics(experiment: BaselineResults) -> None:
    protocol = experiment.metric_protocol
    named = {
        protocol.primary_development_metric,
        protocol.secondary_ranking_metric,
        *protocol.diagnostic_metrics,
        *protocol.auxiliary_metrics,
    }

    assert named <= set(METRIC_NAMES)


def test_metric_protocol_prescribes_paired_per_fold_comparison(
    experiment: BaselineResults,
) -> None:
    """Later phases compare deltas on the same folds, not against a baseline SD."""
    method = experiment.metric_protocol.comparison_method.lower()

    assert "paired" in method
    assert "per-fold" in method or "per fold" in method
    assert "standard deviation" in method


def test_record_warns_that_fold_spread_is_not_a_comparison_threshold(
    experiment: BaselineResults,
) -> None:
    notes = " ".join(experiment.methodological_notes).lower()

    assert "not a margin of error" in notes


def test_record_declares_no_engineered_features(experiment: BaselineResults) -> None:
    assert experiment.feature_set.engineered == []
    assert experiment.feature_set.n_features == len(FEATURE_COLUMNS)


def test_record_holds_every_model_with_every_fold(experiment: BaselineResults) -> None:
    assert len(experiment.models) == 3
    for model in experiment.models:
        assert len(model.folds) == N_SPLITS
        assert set(model.mean) == set(METRIC_NAMES)
        assert set(model.std) == set(METRIC_NAMES)
        assert set(model.out_of_fold) == set(METRIC_NAMES)


def test_record_captures_effective_hyperparameters(experiment: BaselineResults) -> None:
    logistic = next(m for m in experiment.models if m.name == LOGISTIC_REGRESSION)

    assert logistic.hyperparameters["C"] == 1.0
    assert logistic.hyperparameters["class_weight"] is None


def test_confusion_counts_sum_to_the_training_rows(experiment: BaselineResults) -> None:
    for model in experiment.models:
        assert sum(model.out_of_fold_confusion.values()) == experiment.n_training_rows


def test_record_carries_no_timestamp(experiment: BaselineResults) -> None:
    """A timestamp would break the determinism check without adding traceability."""
    serialized = json.dumps(experiment.model_dump())

    assert "timestamp" not in serialized
    assert "generated_at" not in serialized


def test_roundtrip_validates_against_the_schema(
    experiment: BaselineResults, tmp_path: Path
) -> None:
    destination = write_results(experiment, tmp_path / "baseline_results.json")

    assert load_results(destination) == experiment


def test_serialization_is_stable(experiment: BaselineResults, tmp_path: Path) -> None:
    first = write_results(experiment, tmp_path / "first.json").read_text(encoding="utf-8")
    second = write_results(experiment, tmp_path / "second.json").read_text(encoding="utf-8")

    assert first == second


def test_missing_record_points_at_the_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_baselines.py"):
        load_results(tmp_path / "absent.json")


def test_malformed_record_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_results(path)


# --- holdout audit ------------------------------------------------------------

_PHASE5_SOURCES = [
    *sorted((PROJECT_ROOT / "src" / "churn" / "modeling").glob("*.py")),
    PROJECT_ROOT / "scripts" / "run_baselines.py",
]


def _identifiers(source: Path) -> set[str]:
    """Every name the module actually *uses*, ignoring strings and comments.

    A substring search would flag the report template, which discusses the
    holdout in prose. The AST sees only code, which is what the audit is about.
    """
    used: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.alias):
            used.add(node.name.rsplit(".", 1)[-1])
            if node.asname:
                used.add(node.asname)
    return used


#: Modules authorised to name the holdout partition, and why.
#:
#: This audit was written in Phase 5, when no module under ``modeling/`` was
#: allowed to touch the holdout at all. Phase 9D is the authorised exception: it
#: is the final evaluation, and it necessarily names the partition it measures.
#: The exemption is a **declared list rather than a relaxed rule**, so a module
#: that starts naming the holdout without being on it still fails — and even the
#: exempt modules may not import the loader, which stays forbidden everywhere
#: under ``modeling/``. Only the orchestration script opens the partition.
_HOLDOUT_CONSUMERS: frozenset[str] = frozenset(
    {
        "holdout.py",  # Phase 9D: computes the final metrics on arrays handed to it
        "holdout_results.py",  # Phase 9D: records the final evaluation
    }
)


@pytest.mark.parametrize("source", _PHASE5_SOURCES, ids=lambda path: path.name)
def test_phase5_code_never_reaches_the_holdout(source: Path) -> None:
    used = _identifiers(source)

    assert "load_holdout" not in used, f"{source.name} imports or calls the holdout loader"
    if source.name in _HOLDOUT_CONSUMERS:
        return
    assert "holdout" not in used, f"{source.name} reads a holdout partition"


def test_only_the_declared_phase9d_modules_name_the_holdout() -> None:
    """The exemption list is exactly the set of modules that use the name."""
    naming = {
        source.name
        for source in _PHASE5_SOURCES
        if source.parent.name == "modeling" and "holdout" in _identifiers(source)
    }

    assert naming == set(_HOLDOUT_CONSUMERS)


def test_evaluation_module_cannot_load_data() -> None:
    """Evaluation receives X and y; it has no data access of its own."""
    used = _identifiers(PROJECT_ROOT / "src" / "churn" / "modeling" / "evaluation.py")

    assert "load_training_pool" not in used
    assert "load_raw_typed" not in used
    assert "load_split" not in used


def test_baseline_script_reads_only_the_training_pool() -> None:
    used = _identifiers(PROJECT_ROOT / "scripts" / "run_baselines.py")

    assert "load_training_pool" in used
    assert "load_split" not in used
    assert "load_raw_typed" not in used
