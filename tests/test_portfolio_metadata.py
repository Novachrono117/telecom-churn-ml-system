"""The metrics the product displays must be Phase 9D's, copied and never recomputed.

A portfolio page is the place where an evaluation quietly becomes a marketing number:
accuracy gets promoted because it is the largest, a caveat gets dropped because it is
inconvenient, and a metric gets recomputed because it was easier than reading the
artefact. These tests exist so none of that can happen without a failure.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from churn.portfolio.metadata import (
    ANALYST_EXPOSURE_CAVEAT,
    EXPECTED_PORTFOLIO_METADATA_SHA256,
    HOLDOUT_RESULTS_RELATIVE_PATH,
    PORTFOLIO_METADATA_TRUST_ANCHOR,
    PortfolioMetadata,
    PortfolioMetadataError,
    build_metadata,
    canonical_json,
    default_metadata_path,
    load_portfolio_metadata,
    metadata_digest,
    raise_on_failure,
    verify_portfolio_metadata,
)
from churn.preprocessing import splitting as splitting_module


@pytest.fixture(scope="module")
def metadata():
    return build_metadata()


@pytest.fixture(scope="module")
def holdout_record():
    from churn.config import PROJECT_ROOT

    path = PROJECT_ROOT / HOLDOUT_RESULTS_RELATIVE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Copied, not recomputed.
# --------------------------------------------------------------------------- #


def test_every_metric_equals_the_phase_9d_artefact(metadata, holdout_record) -> None:
    """Value for value against the committed record, not to within a rounding."""
    metrics = holdout_record["metrics"]
    flat = {}
    for section in metrics.values():
        if isinstance(section, dict):
            flat.update(section)

    for entry in [*metadata.evaluation.headline, *metadata.evaluation.auxiliary]:
        assert entry.value == flat[entry.key], entry.key


def test_every_interval_equals_the_phase_9d_artefact(metadata, holdout_record) -> None:
    intervals = holdout_record["confidence_intervals"]

    for entry in metadata.evaluation.headline:
        assert entry.ci_lower == intervals[entry.key]["ci_lower"], entry.key
        assert entry.ci_upper == intervals[entry.key]["ci_upper"], entry.key
        assert entry.confidence_level == intervals[entry.key]["confidence_level"]


def test_the_metadata_reopens_no_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every loader armed while the summary is built. It reads JSON and nothing else."""

    def explode(*_: object, **__: object) -> None:
        raise AssertionError("a dataset loader was called while building the metadata.")

    for name in ("load_holdout", "load_training_pool", "load_split", "split_dataset"):
        monkeypatch.setattr(splitting_module, name, explode, raising=False)
    monkeypatch.setattr(pd, "read_csv", explode)

    metadata = build_metadata()

    assert metadata.provenance.holdout_reopened is False
    assert metadata.provenance.metrics_recomputed is False
    assert metadata.evaluation.evaluations_performed == 1


def test_the_metadata_scores_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No pipeline is loaded, so no probability can be produced while building it."""
    import joblib

    def explode(*_: object, **__: object) -> None:
        raise AssertionError("an artefact was loaded while building the metadata.")

    monkeypatch.setattr(joblib, "load", explode)

    metadata = build_metadata()

    assert metadata.policy.threshold == 0.3272694566222328


# --------------------------------------------------------------------------- #
# What may be highlighted.
# --------------------------------------------------------------------------- #


def test_accuracy_is_never_a_headline_metric(metadata) -> None:
    """On a 73.5 % negative population, predicting nobody churns scores near it."""
    headline = {entry.key for entry in metadata.evaluation.headline}
    auxiliary = {entry.key for entry in metadata.evaluation.auxiliary}

    assert "accuracy" not in headline
    assert "accuracy" in auxiliary
    assert metadata.evaluation.headline[0].key == "average_precision"
    assert metadata.evaluation.headline[1].key == "roc_auc"


def test_the_ranking_metrics_are_named_as_ranking_quality(metadata) -> None:
    """Not accuracy, not certainty. ROC-AUC 0.842 is neither."""
    note = metadata.evaluation.metric_reading_note

    assert "ranking quality" in note
    assert "not accuracy" in note
    assert "not certainty" in note


def test_the_no_skill_baselines_travel_with_the_metrics(metadata, holdout_record) -> None:
    """0.63 average precision means little without the 0.27 a coin flip would get."""
    baselines = holdout_record["metrics"]["reference_baselines"]

    assert metadata.evaluation.no_skill_average_precision == baselines["no_skill_average_precision"]
    assert metadata.evaluation.no_skill_roc_auc == baselines["no_skill_roc_auc"]
    assert metadata.evaluation.majority_class_accuracy == baselines["majority_class_accuracy"]


def test_the_analyst_exposure_caveat_travels_with_the_metrics(metadata) -> None:
    """A field of the artefact, so the numbers cannot be rendered without it."""
    caveat = metadata.evaluation.caveat

    assert caveat == ANALYST_EXPOSURE_CAVEAT
    assert "optimistic bias" in caveat
    assert "cannot be quantified" in caveat


def test_no_calibrated_probability_is_claimed(metadata) -> None:
    assert metadata.policy.calibration_policy == "NONE"


# --------------------------------------------------------------------------- #
# Determinism and verification.
# --------------------------------------------------------------------------- #


def test_the_metadata_is_deterministic() -> None:
    """Two builds on an unchanged repository are byte-identical."""
    first = canonical_json(build_metadata())
    second = canonical_json(build_metadata())

    assert first == second
    assert "timestamp" not in first
    assert "generated_at" not in first


def test_the_committed_file_matches_a_fresh_build() -> None:
    committed = load_portfolio_metadata()
    rebuilt = build_metadata()

    assert committed == rebuilt
    assert canonical_json(committed) == default_metadata_path().read_text(encoding="utf-8")


def test_every_invariant_holds_on_the_committed_metadata() -> None:
    checks = verify_portfolio_metadata(load_portfolio_metadata())

    failed = [name for name, passed, _ in checks if not passed]

    assert not failed, failed
    assert len(checks) >= 10


def test_a_tampered_summary_fails_verification(metadata) -> None:
    """Promoting accuracy to the headline is exactly the failure to catch."""
    payload = metadata.model_dump()
    payload["evaluation"]["headline"].append(
        {
            "key": "accuracy",
            "label": "Accuracy",
            "value": 0.761533,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": None,
        }
    )
    tampered = type(metadata).model_validate(payload)

    with pytest.raises(PortfolioMetadataError) as error:
        raise_on_failure(verify_portfolio_metadata(tampered))

    assert "accuracy_is_not_headline" in str(error.value)


def test_a_summary_claiming_a_reopened_holdout_fails(metadata) -> None:
    payload = metadata.model_dump()
    payload["provenance"]["holdout_reopened"] = True
    tampered = type(metadata).model_validate(payload)

    with pytest.raises(PortfolioMetadataError) as error:
        raise_on_failure(verify_portfolio_metadata(tampered))

    assert "holdout_not_reopened" in str(error.value)


def test_the_provenance_names_the_artefact_it_copied(metadata) -> None:
    """So a reader can confirm which evaluation the numbers came from."""
    import hashlib

    from churn.config import PROJECT_ROOT

    path = PROJECT_ROOT / HOLDOUT_RESULTS_RELATIVE_PATH
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert metadata.provenance.holdout_results_sha256 == digest
    assert metadata.provenance.holdout_results_path == HOLDOUT_RESULTS_RELATIVE_PATH
    assert (
        metadata.provenance.model_fingerprint_sha256
        == "a57568e18ec0333e1c85b590f98e75da93ebbfafff063860ec5daf4e460d939f"
    )
    assert (
        metadata.provenance.pipeline_sha256
        == "574fde36c6e2e991de7c3dfdab21981dc41eae2810d003ecbfb179dd504dc3d8"
    )


def test_the_confusion_matrix_is_internally_consistent(metadata) -> None:
    matrix = metadata.evaluation.confusion_matrix

    assert sum(matrix.values()) == metadata.evaluation.n_samples
    assert matrix["true_positives"] + matrix["false_negatives"] == metadata.evaluation.n_positive
    assert matrix["true_negatives"] + matrix["false_positives"] == metadata.evaluation.n_negative


def test_the_metadata_carries_no_row_and_no_identifier(metadata) -> None:
    """Aggregates only. Nothing that could be traced to a customer."""
    rendered = canonical_json(metadata)

    assert "customerID" not in rendered
    for forbidden in ("probabilities", "scores", "predictions", "row_", "customer_id"):
        assert forbidden not in rendered


# --------------------------------------------------------------------------- #
# The trust anchor.
#
# Every check above this line describes STRUCTURE, and structure is exactly what a
# tampered file keeps. A summary whose average precision was edited still reports one
# evaluation, still keeps accuracy out of the headline, still carries a caveat
# containing the words "optimistic bias", and still passes every invariant — while
# displaying a number nobody measured. Only a digest compared against an expectation
# recorded OUTSIDE the file separates the summary from a file shaped like it.
# --------------------------------------------------------------------------- #


def _tampered(metadata: PortfolioMetadata, mutate) -> PortfolioMetadata:
    """Return a valid summary with one presentation field changed.

    Round-tripped through the model so the result is a genuine
    :class:`PortfolioMetadata` — the point of these tests is that a tampered file is
    *not* malformed, and building a malformed one would prove nothing.
    """
    payload = json.loads(canonical_json(metadata))
    mutate(payload)
    return PortfolioMetadata.model_validate(payload)


def _inflate_average_precision(payload: dict) -> None:
    for entry in payload["evaluation"]["headline"]:
        if entry["key"] == "average_precision":
            entry["value"] = 0.97
            entry["ci_lower"] = 0.96
            entry["ci_upper"] = 0.98


def _inflate_roc_auc(payload: dict) -> None:
    for entry in payload["evaluation"]["headline"]:
        if entry["key"] == "roc_auc":
            entry["value"] = 0.99


def _soften_the_caveat(payload: dict) -> None:
    """Keep the words the invariant looks for; drop the sentence that stings."""
    payload["evaluation"]["caveat"] = (
        "The estimate may carry optimistic bias."  # the exposure explanation is gone
    )


def _relabel_a_metric(payload: dict) -> None:
    payload["evaluation"]["headline"][0]["label"] = "Accuracy on unseen customers"


TAMPERINGS = {
    "average_precision": _inflate_average_precision,
    "roc_auc": _inflate_roc_auc,
    "analyst_caveat": _soften_the_caveat,
    "metric_label": _relabel_a_metric,
}


def test_the_committed_summary_matches_the_pinned_digest(metadata) -> None:
    """The artefact and the constant move together, in one commit, or this fails."""
    committed = load_portfolio_metadata(default_metadata_path())

    assert metadata_digest(committed) == EXPECTED_PORTFOLIO_METADATA_SHA256
    assert metadata_digest(metadata) == EXPECTED_PORTFOLIO_METADATA_SHA256


def test_the_digest_is_a_property_of_the_value_not_of_the_bytes(metadata) -> None:
    """A summary built in memory and one read back from disk hash identically."""
    committed = load_portfolio_metadata(default_metadata_path())

    assert metadata_digest(metadata) == metadata_digest(committed)
    assert len(metadata_digest(metadata)) == 64


def test_the_expectation_is_not_stored_in_the_file_it_checks(metadata) -> None:
    """A digest a file carries about itself is not an expectation, it is an echo."""
    rendered = canonical_json(metadata)

    assert EXPECTED_PORTFOLIO_METADATA_SHA256 not in rendered
    assert PORTFOLIO_METADATA_TRUST_ANCHOR.startswith("churn.portfolio.metadata.")


def test_the_expectation_is_not_stored_in_the_decision_policy() -> None:
    """The frozen Phase 9C provenance is not where a Phase 13 pin belongs."""
    from churn.config import PROJECT_ROOT

    policy = (PROJECT_ROOT / "reports" / "decision_policy.json").read_text(encoding="utf-8")

    assert EXPECTED_PORTFOLIO_METADATA_SHA256 not in policy


@pytest.mark.parametrize("name", sorted(TAMPERINGS))
def test_a_tampered_summary_still_passes_every_structural_invariant(name, metadata) -> None:
    """The premise of the gate: structure alone cannot detect this tampering."""
    tampered = _tampered(metadata, TAMPERINGS[name])

    assert tampered != metadata
    for check, holds, detail in verify_portfolio_metadata(tampered):
        assert holds, f"{check} failed on the tampered summary ({detail})"


@pytest.mark.parametrize("name", sorted(TAMPERINGS))
def test_a_tampered_summary_is_caught_by_the_digest(name, metadata) -> None:
    """...and the digest is what catches it."""
    tampered = _tampered(metadata, TAMPERINGS[name])
    checks = verify_portfolio_metadata(tampered, EXPECTED_PORTFOLIO_METADATA_SHA256)
    failed = [check for check, holds, _ in checks if not holds]

    assert metadata_digest(tampered) != EXPECTED_PORTFOLIO_METADATA_SHA256
    assert failed == ["metadata_digest_matches"]

    with pytest.raises(PortfolioMetadataError, match="metadata_digest_matches"):
        raise_on_failure(checks)


def test_the_intact_summary_passes_the_digest_check(metadata) -> None:
    checks = verify_portfolio_metadata(metadata, EXPECTED_PORTFOLIO_METADATA_SHA256)

    assert dict((check, holds) for check, holds, _ in checks)["metadata_digest_matches"]
    assert raise_on_failure(checks) is checks


def test_the_digest_check_is_opt_in(metadata) -> None:
    """Offline builders check structure before a pin exists; startup checks both."""
    without = [check for check, _, _ in verify_portfolio_metadata(metadata)]
    with_pin = [
        check
        for check, _, _ in verify_portfolio_metadata(metadata, EXPECTED_PORTFOLIO_METADATA_SHA256)
    ]

    assert "metadata_digest_matches" not in without
    assert with_pin == [*without, "metadata_digest_matches"]
