"""The machine-readable record: built from the system, and reproducible."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from churn.serving.artifacts import FREEZE_COMMIT
from churn.serving.record import (
    COMPONENT,
    PHASE,
    RECORD_PATH,
    VERIFIED_BY,
    build_record,
    read_record,
    record_invariants,
    write_record,
)
from churn.serving.service import ChurnInferenceService

#: Anything shaped like an ISO-8601 date or datetime.
ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2})?")


@pytest.fixture(scope="module")
def record(service: ChurnInferenceService) -> dict[str, Any]:
    return build_record(service)


def test_every_invariant_holds(record: dict[str, Any]) -> None:
    failures = [(name, detail) for name, holds, detail in record_invariants(record) if not holds]

    assert not failures


def test_the_record_is_a_pure_function_of_the_system(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """No clock, no run id: two builds are byte-identical or the record is not evidence."""
    again = build_record(service)

    assert json.dumps(record, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_the_record_carries_no_wall_clock_field(record: dict[str, Any]) -> None:
    """Checked on keys and values, not on a substring of the whole file.

    A naive substring scan would flag `timestamp_in_response: false`, which is the
    record *declaring the absence* of a clock — the opposite of the problem.
    """
    forbidden = {
        "generated_at",
        "timestamp",
        "created_at",
        "run_at",
        "duration_seconds",
        "hostname",
        "run_id",
        "elapsed",
        "date",
    }

    def keys(node: Any) -> Iterator[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    assert not forbidden & set(keys(record))

    def values(node: Any) -> Iterator[str]:
        if isinstance(node, dict):
            for value in node.values():
                yield from values(value)
        elif isinstance(node, list):
            for item in node:
                yield from values(item)
        elif isinstance(node, str):
            yield node

    assert not [text for text in values(record) if ISO_TIMESTAMP.search(text)]


def test_the_record_names_the_frozen_model_not_this_release(record: dict[str, Any]) -> None:
    frozen = record["frozen_model"]

    assert frozen["freeze_commit"] == FREEZE_COMMIT
    assert record["serving_version"] != frozen["freeze_commit"]
    assert record["model_changed"] is False


def test_the_record_states_the_required_phase_facts(record: dict[str, Any]) -> None:
    """The fields the phase specification asks for, read straight out of the record."""
    assert record["phase"] == PHASE
    assert record["component"] == COMPONENT
    assert record["fit_calls"] == 0
    assert record["holdout_loaded"] is False
    assert record["training_data_loaded"] is False
    assert record["inference_chain"]["prepare_features_required"] is True
    assert record["inference_chain"]["predict_proba_used"] is True
    assert record["inference_chain"]["pipeline_predict_used"] is False
    assert record["startup"]["artifact_validation_on_startup"] is True
    assert record["startup"]["fail_closed"] is True
    assert record["privacy"]["payloads_persisted"] is False
    assert record["privacy"]["identifiers_persisted"] is False


def test_the_endpoints_are_read_from_the_application(record: dict[str, Any]) -> None:
    """Not from a list kept by hand, which is how such a list goes stale."""
    api = record["api"]

    assert api["single_endpoint"] in api["published_paths"]
    assert api["batch_endpoint"] in api["published_paths"]
    assert set(api["health_endpoints"]) <= set(api["published_paths"])


def test_the_gates_listed_are_the_gates_that_ran(
    record: dict[str, Any], service: ChurnInferenceService
) -> None:
    assert record["startup"]["gates"] == [check.name for check in service.artifacts.checks]
    assert record["startup"]["n_gates"] == len(service.artifacts.checks)


def test_every_prohibition_names_a_test_that_exists(record: dict[str, Any]) -> None:
    """A claim with no evidence behind it is a claim the record should not make."""
    tests_dir = Path(__file__).parent

    for claim, reference in VERIFIED_BY.items():
        path, _, test_name = reference.partition("::")
        module = tests_dir / Path(path).name
        assert module.is_file(), f"{claim} points at a missing module: {path}"
        assert f"def {test_name}(" in module.read_text(encoding="utf-8"), (
            f"{claim} points at a missing test: {reference}"
        )


def test_the_committed_record_matches_the_running_boundary(record: dict[str, Any]) -> None:
    """The same check `--verify` runs, so a stale record fails the suite too."""
    assert RECORD_PATH.is_file()
    assert read_record() == record


def test_writing_the_record_is_idempotent(record: dict[str, Any], tmp_path: Path) -> None:
    destination = tmp_path / "serving_results.json"

    first = write_record(record, destination).read_bytes()
    second = write_record(record, destination).read_bytes()

    assert first == second
    assert first.endswith(b"\n")
    assert b"\r\n" not in first
