"""The Phase 6 record, and the baseline-reproduction guard.

The reproduction test against the real dataset is the one that matters: if E0
stops reproducing the frozen Phase 5 numbers, every delta in the phase becomes
uninterpretable, because the difference could be an artefact of the pipeline
rather than of the feature. It is skipped on a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from churn.data.loader import raw_csv_path
from churn.features.ablation import PRIMARY_METRIC, SECONDARY_METRIC, run_experiment
from churn.features.groups import ABLATION_ORDER, FEATURE_GROUPS
from churn.features.results import (
    SCHEMA_VERSION,
    AblationResults,
    BaselineReproductionError,
    build_results,
    load_results,
    verify_baseline_reproduction,
    write_results,
)
from churn.modeling.evaluation import build_splitter
from churn.modeling.results import load_results as load_baseline_results
from churn.preprocessing.contracts import build_feature_matrix
from churn.preprocessing.manifest import MANIFEST_PATH
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target

_FAKE_DIGEST = "0" * 64


@pytest.fixture
def synthetic_experiments(contract_frame: pd.DataFrame):
    features = build_feature_matrix(contract_frame)
    target = encode_target(contract_frame["Churn"])
    splitter = build_splitter()
    baseline = run_experiment("E0", "baseline", (), features, target, splitter)
    candidate = run_experiment(
        "E1",
        "Original + protective_services",
        ("protective_services",),
        features,
        target,
        splitter,
        baseline.evaluation,
    )
    return {"E0": baseline, "E1": candidate}, np.asarray(target)


def _record(experiments, target, reference) -> AblationResults:
    from churn.features.results import BaselineReproduction

    return build_results(
        experiments,
        reference,
        BaselineReproduction(
            reference_artefact="reports/experiments/baseline_results.json",
            reference_schema_version=reference.schema_version,
            reference_model="logistic_regression",
            tolerance=1e-9,
            max_absolute_difference=0.0,
            reproduced=True,
        ),
        target,
        ("E0",),
    )


@pytest.fixture
def reference():
    if not MANIFEST_PATH.is_file() or not raw_csv_path().is_file():
        pytest.skip("Phase 5 artefacts require the acquired dataset")
    return load_baseline_results()


def test_record_declares_every_candidate(synthetic_experiments, reference) -> None:
    experiments, target = synthetic_experiments

    results = _record(experiments, target, reference)

    assert {candidate.group for candidate in results.candidates} == set(FEATURE_GROUPS)
    assert results.schema_version == SCHEMA_VERSION


def test_record_marks_only_charge_intensity_as_stateful(synthetic_experiments, reference) -> None:
    experiments, target = synthetic_experiments

    results = _record(experiments, target, reference)
    stateful = {candidate.group for candidate in results.candidates if candidate.stateful}

    assert stateful == {"charge_intensity"}


def test_record_keeps_paired_deltas_for_candidates_only(synthetic_experiments, reference) -> None:
    experiments, target = synthetic_experiments

    results = _record(experiments, target, reference)
    by_id = {record.experiment: record for record in results.experiments}

    assert by_id["E0"].paired_deltas == {}
    assert set(by_id["E1"].paired_deltas) == {PRIMARY_METRIC, SECONDARY_METRIC}
    assert by_id["E0"].classification is None
    assert by_id["E1"].classification is not None


def test_record_states_the_folds_are_identical_to_the_baseline(
    synthetic_experiments, reference
) -> None:
    experiments, target = synthetic_experiments

    results = _record(experiments, target, reference)

    assert results.cross_validation["identical_to_baseline"] is True
    assert results.cross_validation["shared_across_experiments"] is True


def test_record_carries_no_timestamp(synthetic_experiments, reference) -> None:
    experiments, target = synthetic_experiments

    serialized = json.dumps(_record(experiments, target, reference).model_dump())

    assert "timestamp" not in serialized
    assert "generated_at" not in serialized


def test_roundtrip_validates_against_the_schema(
    synthetic_experiments, reference, tmp_path: Path
) -> None:
    experiments, target = synthetic_experiments
    results = _record(experiments, target, reference)

    destination = write_results(results, tmp_path / "ablation.json")

    assert load_results(destination) == results


def test_serialization_is_stable(synthetic_experiments, reference, tmp_path: Path) -> None:
    experiments, target = synthetic_experiments
    results = _record(experiments, target, reference)

    first = write_results(results, tmp_path / "a.json").read_text(encoding="utf-8")
    second = write_results(results, tmp_path / "b.json").read_text(encoding="utf-8")

    assert first == second


def test_missing_record_points_at_the_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_feature_engineering.py"):
        load_results(tmp_path / "absent.json")


def test_malformed_record_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_results(path)


# --- baseline reproduction against the real dataset ---------------------------

pytestmark_real = pytest.mark.skipif(
    not raw_csv_path().is_file() or not MANIFEST_PATH.is_file(),
    reason="raw dataset not acquired yet (see data/README.md)",
)


@pytestmark_real
def test_e0_reproduces_the_frozen_phase5_baseline() -> None:
    """The control of the whole phase. If this drifts, no delta is interpretable."""
    training = load_training_pool()
    features = build_feature_matrix(training)
    target = encode_target(training["Churn"])

    e0 = run_experiment("E0", "baseline", (), features, target, build_splitter())
    reproduction = verify_baseline_reproduction(e0, load_baseline_results())

    assert reproduction.reproduced
    assert reproduction.max_absolute_difference <= reproduction.tolerance


@pytestmark_real
def test_reproduction_guard_fires_on_a_perturbed_reference() -> None:
    """The guard must actually be able to fail, not just pass."""
    training = load_training_pool()
    features = build_feature_matrix(training)
    target = encode_target(training["Churn"])
    e0 = run_experiment("E0", "baseline", (), features, target, build_splitter())

    reference = load_baseline_results()
    perturbed = reference.model_dump()
    logistic = next(m for m in perturbed["models"] if m["name"] == "logistic_regression")
    logistic["folds"][0]["metrics"][PRIMARY_METRIC] += 0.01

    with pytest.raises(BaselineReproductionError, match="does not reproduce"):
        verify_baseline_reproduction(e0, type(reference).model_validate(perturbed))


@pytestmark_real
def test_every_registered_group_appears_in_the_ablation_order() -> None:
    assert set(ABLATION_ORDER) == set(FEATURE_GROUPS)
    assert len(ABLATION_ORDER) == len(FEATURE_GROUPS)
