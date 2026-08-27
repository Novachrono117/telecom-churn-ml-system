"""The Phase 12 machine-readable record: reproducible, and honest about its limits."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from churn.monitoring.reference import load_reference_profile, profile_digest
from churn.monitoring.results import (
    COMPONENT,
    PHASE,
    RECORD_PATH,
    SERVING_VERIFIED_BY,
    VERIFIED_BY,
    build_record,
    canonical_json,
    read_record,
    record_invariants,
    write_record,
)
from churn.monitoring.settings import load_monitoring_policy

ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2})?")


def _keys(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    return build_record(load_reference_profile(), load_monitoring_policy())


def test_every_invariant_holds(record: dict[str, Any]) -> None:
    failures = [(name, detail) for name, holds, detail in record_invariants(record) if not holds]

    assert not failures


def test_the_record_is_a_pure_function_of_the_system(record: dict[str, Any]) -> None:
    again = build_record(load_reference_profile(), load_monitoring_policy())

    assert json.dumps(record, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_the_record_carries_no_wall_clock(record: dict[str, Any]) -> None:
    forbidden = {"generated_at", "timestamp", "created_at", "run_at", "hostname", "run_id"}

    assert not forbidden & set(_keys(record))
    values = [value for value in json.dumps(record).split('"') if ISO_TIMESTAMP.search(value)]
    assert not values


def test_the_record_states_the_required_phase_facts(record: dict[str, Any]) -> None:
    assert record["phase"] == PHASE
    assert record["component"] == COMPONENT
    assert record["model_changed"] is False
    assert record["fit_calls"] == 0
    assert record["holdout_loaded"] is False
    assert record["reference"]["reference_population"] == "training_pool"
    assert record["reference"]["reference_n"] == 5634
    assert record["reference"]["target_used_for_reference"] is False
    assert record["metrics"]["numeric_drift_metric"] == "population_stability_index"
    assert record["metrics"]["categorical_drift_metric"] == "total_variation_distance"
    assert record["metrics"]["score_drift_metric"] == "population_stability_index"
    assert record["monitoring_scope"]["structural_checks"] == 7
    assert record["monitoring_scope"]["unseen_category_monitoring"] is True
    assert record["monitoring_scope"]["prediction_drift_monitoring"] is True
    assert record["monitoring_scope"]["performance_monitoring_without_labels"] is False
    assert record["privacy"]["payloads_persisted"] is False
    assert record["privacy"]["identifiers_persisted"] is False
    assert record["privacy"]["row_level_predictions_persisted"] is False
    assert record["serving_integration"]["monitoring_changes_prediction"] is False


def test_the_record_names_the_reference_it_describes(record: dict[str, Any]) -> None:
    assert record["reference"]["reference_profile_sha256"] == profile_digest(
        load_reference_profile()
    )


def test_the_record_carries_the_window_policy(record: dict[str, Any]) -> None:
    policy = load_monitoring_policy()

    assert record["window"]["minimum_window_size"] == policy.minimum_window_size
    assert record["window"]["counts_reported_below_minimum"] is True
    assert record["window"]["drift_claimed_below_minimum"] is False


def test_the_alert_policy_is_labelled_operational(record: dict[str, Any]) -> None:
    """A cutoff presented as a significance level would be the phase's worst failure."""
    alert = record["alert_policy"]

    assert alert["classification"] == "OPERATIONAL_MONITORING_POLICY"
    assert alert["is_statistical_significance"] is False
    assert alert["estimated_from_data"] is False
    assert record["metrics"]["is_statistical_test"] is False
    assert record["metrics"]["p_values_computed"] is False


def test_the_record_declares_no_new_dependency(record: dict[str, Any]) -> None:
    dependencies = record["dependencies"]

    assert dependencies["new_runtime_dependencies"] == []
    assert dependencies["root_pyproject_modified"] is False
    assert dependencies["root_uv_lock_modified"] is False
    assert dependencies["configs_base_toml_modified"] is False
    assert dependencies["new_environment_created"] is False


def test_the_record_does_not_claim_the_phase_11_record_was_rewritten(
    record: dict[str, Any],
) -> None:
    integration = record["serving_integration"]

    assert integration["phase11_serving_record_rewritten"] is False
    assert integration["phase11_route_list_unchanged_by_default"] is True
    assert integration["opt_in"] is True
    assert integration["default_enabled"] is False
    assert integration["http_reset_endpoint"] is False


def test_readiness_and_drift_stay_separate(record: dict[str, Any]) -> None:
    integration = record["serving_integration"]

    assert integration["drift_can_make_service_not_ready"] is False
    assert integration["artifact_corruption_makes_service_not_ready"] is True


def test_every_prohibition_names_a_root_test_that_exists(record: dict[str, Any]) -> None:
    tests_dir = Path(__file__).parent

    for claim, reference in VERIFIED_BY.items():
        path, _, test_name = reference.partition("::")
        module = tests_dir / Path(path).name
        assert module.is_file(), f"{claim} points at a missing module: {path}"
        assert f"def {test_name}(" in module.read_text(encoding="utf-8"), (
            f"{claim} points at a missing test: {reference}"
        )


def test_every_serving_claim_names_a_serving_test_that_exists(record: dict[str, Any]) -> None:
    root = Path(__file__).parents[1]

    for claim, reference in SERVING_VERIFIED_BY.items():
        path, _, test_name = reference.partition("::")
        module = root / path
        assert module.is_file(), f"{claim} points at a missing module: {path}"
        assert f"def {test_name}(" in module.read_text(encoding="utf-8"), (
            f"{claim} points at a missing test: {reference}"
        )


def test_the_committed_record_matches_the_running_system(record: dict[str, Any]) -> None:
    assert RECORD_PATH.is_file()
    assert read_record() == record


def test_writing_the_record_is_idempotent(record: dict[str, Any], tmp_path: Path) -> None:
    destination = tmp_path / "monitoring_results.json"

    first = write_record(record, destination).read_bytes()
    second = write_record(record, destination).read_bytes()

    assert first == second
    assert first.endswith(b"\n")
    assert b"\r\n" not in first
    assert canonical_json(record).encode("utf-8") == first
