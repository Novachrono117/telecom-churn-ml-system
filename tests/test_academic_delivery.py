"""Phase 14A: gates on the academic package.

These tests guard two different things, and mixing them up would weaken both.

**That the deliverable is complete.** Every required section is present, every frozen
anchor is pinned, the notebook can start on a clean runtime.

**That the deliverable is honest.** No Colab execution is claimed, no link was
invented, no number contradicts the artefact it came from, and the caveats the
project has carried since Phase 3 are still there. These are the tests that matter,
because a rubric rewards completeness and nothing rewards the caveat — so the caveat
is what quietly goes missing.

Nothing here re-runs an experiment. Phase 14A takes decisions already made and turns
them into a deliverable; a test that fitted a model would be doing the one thing the
phase forbids.
"""

from __future__ import annotations

import json
import re

import pytest

from churn.academic.results import (
    EXTERNAL_INPUT_MARKER,
    REQUIRED_NOTEBOOK_SECTIONS,
    REQUIRED_REPORT_SECTIONS,
    build_record,
    canonical_json,
    read_record,
    record_invariants,
)
from churn.config import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/02_academic_delivery.ipynb"
REPORT_PATH = PROJECT_ROOT / "reports/academic/academic_report.md"
RECORD_PATH = PROJECT_ROOT / "reports/experiments/academic_delivery_results.json"
CHECKLIST_PATH = PROJECT_ROOT / "reports/academic/submission_checklist.md"

FROZEN_THRESHOLD = 0.3272694566222328
RAW_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def notebook_source(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def report_prose(report: str) -> str:
    """The report with block-quote markers stripped, wrapping removed, lower-cased."""
    return " ".join(
        " ".join(line.lstrip().lstrip(">").strip() for line in report.splitlines()).split()
    ).lower()


@pytest.fixture(scope="module")
def record() -> dict:
    return read_record()


@pytest.fixture(scope="module")
def holdout() -> dict:
    path = PROJECT_ROOT / "reports/experiments/holdout_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


# -- 1. the report is structurally a submission ---------------------------------


def test_the_report_carries_every_required_section(report: str) -> None:
    """The eleven sections the assignment names, as level-1 headings."""
    headings = {line[2:].strip() for line in report.splitlines() if line.startswith("# ")}
    missing = [section for section in REQUIRED_REPORT_SECTIONS if section not in headings]
    assert not missing, f"missing required report sections: {missing}"


# -- 2. the notebook tells the whole academic story ------------------------------


def test_the_notebook_carries_every_academic_section(notebook_source: str) -> None:
    missing = [s for s in REQUIRED_NOTEBOOK_SECTIONS if s not in notebook_source]
    assert not missing, f"missing notebook sections: {missing}"


# -- 3 to 5. the notebook can start on a clean Colab runtime ---------------------


def test_the_notebook_has_no_absolute_path(notebook_source: str) -> None:
    """A drive letter or home directory would tie the notebook to one machine.

    URLs are removed first: ``https://`` ends in ``s:/``, and the notebook is supposed
    to contain a URL — that is how it avoids needing a local file at all.
    """
    without_urls = re.sub(r"https?://\S*", "", notebook_source)
    found = re.findall(
        r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"']*|/home/[a-z]\S*|/Users/\S*", without_urls
    )
    assert not found, f"absolute paths in the notebook: {found[:5]}"


def test_the_notebook_requires_no_local_upload_or_mounted_drive(notebook_source: str) -> None:
    """A notebook that needs a file uploaded by hand is not reproducible by a reader."""
    forbidden = ("files.upload", "drive.mount", "google.colab", "kaggle.json", "../data")
    found = [token for token in forbidden if token in notebook_source]
    assert not found, f"local dependencies in the notebook: {found}"


def test_the_notebook_carries_no_credential(notebook_source: str) -> None:
    pattern = re.compile(r"api_key\s*=|secret\s*=|password\s*=|Bearer\s|token\s*=\s*['\"]")
    assert not pattern.search(notebook_source), "the notebook appears to embed a credential"


def test_the_notebook_does_not_import_the_local_package(notebook_source: str) -> None:
    """Option B: the academic notebook reproduces the protocol, it does not import it.

    The configured remote is not publicly reachable and the current commit is not
    published on it, so a notebook that installed the repository could not run in a
    clean Colab session.
    """
    assert "from churn" not in notebook_source
    assert "import churn" not in notebook_source


# -- 6 to 9. the notebook pins the frozen protocol -------------------------------


def test_the_notebook_pins_the_expected_dataset_digest(notebook_source: str) -> None:
    assert RAW_SHA256 in notebook_source
    assert "ABORTADO" in notebook_source, "a digest mismatch must abort, not warn"


def test_the_notebook_pins_the_frozen_split(notebook_source: str) -> None:
    manifest = json.loads((PROJECT_ROOT / "reports/split_manifest.json").read_text("utf-8"))
    assert manifest["training_ids_sha256"] in notebook_source
    assert manifest["holdout_ids_sha256"] in notebook_source
    assert "TEST_SIZE = 0.20" in notebook_source
    assert "RANDOM_SEED = 42" in notebook_source
    assert str(manifest["n_rows_training"]) in notebook_source
    assert str(manifest["n_rows_holdout"]) in notebook_source


def test_the_notebook_pins_the_frozen_threshold(notebook_source: str) -> None:
    policy = json.loads((PROJECT_ROOT / "reports/decision_policy.json").read_text("utf-8"))
    assert policy["threshold"]["final_threshold"] == FROZEN_THRESHOLD
    assert repr(FROZEN_THRESHOLD) in notebook_source


def test_the_notebook_states_the_calibration_policy_is_none(notebook_source: str) -> None:
    policy = json.loads((PROJECT_ROOT / "reports/decision_policy.json").read_text("utf-8"))
    assert policy["calibration"]["calibration_policy"] == "NONE"
    assert "calibration_policy = NONE" in notebook_source


# -- 10 to 11. the reported numbers are the measured ones ------------------------


def test_the_report_metrics_match_the_holdout_artefact(report: str, holdout: dict) -> None:
    """Every headline figure in the report must appear in the Phase 9D record.

    Formatted to four decimals with a comma, as Portuguese prose writes them.
    """
    metrics = {
        **holdout["metrics"]["discrimination_primary"],
        **holdout["metrics"]["operating_point"],
        **holdout["metrics"]["auxiliary"],
    }
    for key in ("average_precision", "roc_auc", "recall", "precision", "f1", "accuracy"):
        rendered = f"{metrics[key]:.4f}".replace(".", ",")
        assert rendered in report, f"{key} ({rendered}) is not stated in the report"

    matrix = holdout["confusion_matrix"]
    for cell in ("true_negatives", "false_positives", "false_negatives", "true_positives"):
        assert str(matrix[cell]) in report, f"confusion cell {cell} missing from the report"


def test_the_report_never_makes_accuracy_the_headline(report_prose: str) -> None:
    """Accuracy is reported, and explicitly demoted every time it is."""
    assert "auxiliar" in report_prose
    assert "nunca como manchete" in report_prose
    assert 'não é "84 % de acerto"' in report_prose, "ROC-AUC must not be sold as accuracy"


# -- 12 to 15. the caveats the project has carried are still carried -------------


def test_the_report_keeps_the_analyst_exposure_caveat(report_prose: str) -> None:
    """Carried since Phase 3, and an academic deliverable is where it goes missing."""
    assert "viés otimista" in report_prose
    assert "7043" in report_prose


def test_the_report_declares_explanations_non_causal(report_prose: str) -> None:
    assert "não são efeitos causais" in report_prose


def test_the_report_does_not_call_psi_a_statistical_test(report_prose: str) -> None:
    assert "não são testes estatísticos" in report_prose
    assert "operational_monitoring_policy" in report_prose
    assert "p-valor" in report_prose, "the absence of a p-value must be stated, not implied"


def test_the_report_refuses_an_arbitrary_retraining_schedule(report_prose: str) -> None:
    assert 'não prescreve "retreinar todo mês"' in report_prose
    assert "investigação" in report_prose


def test_the_report_claims_no_reduction_in_churn(report_prose: str) -> None:
    """Prediction was demonstrated; intervention was not, and is not claimed."""
    assert "não demonstra" in report_prose
    assert "experimento de intervenção" in report_prose


# -- 16 to 18. nothing external is fabricated ------------------------------------


def test_no_colab_link_was_invented(record: dict, report: str) -> None:
    assert record["colab_shared_link"] is None
    assert record["colab_shared_link_verified"] is False
    assert record["actual_google_colab_execution"] is False
    assert not re.search(r"https?://colab\.research\.google\.com/\S+", report)


def test_no_video_link_was_invented(record: dict, report: str) -> None:
    assert record["video_link"] is None
    assert record["video_link_verified"] is False
    for host in ("youtube.com/watch", "youtu.be/", "vimeo.com/", "drive.google.com/file"):
        assert host not in report, f"an unverified {host} link is present"


def test_the_record_reports_the_remaining_external_actions(record: dict) -> None:
    assert record["academic_submission_ready"] is False
    assert record["external_actions_remaining"], "pending work must be enumerated"
    assert record["actual_colab_export_pdf_present"] is False
    # A local nbconvert run is not a Colab run, and the record must say so.
    assert record["academic_notebook_local_execution_passed"] is True
    assert record["local_execution_is_not_colab"] is True


def test_the_checklist_separates_done_from_external(record: dict) -> None:
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    assert "CONCLUÍDO NO REPOSITÓRIO" in text
    assert "EXTERNAL ACTION REQUIRED" in text
    assert "academic_submission_ready = false" in text
    unchecked = text.count("- [ ]")
    assert unchecked == len(
        [line for line in text.splitlines() if line.strip().startswith("- [ ]")]
    )
    assert unchecked > 0, "the external section must still have open items"


def test_the_draft_pdf_is_not_named_final(record: dict) -> None:
    """While a cover field is unfilled, the PDF must not be called the submission."""
    if record["report"]["n_external_input_markers"] > 0:
        assert record["report_pdf_is_draft_only"] is True
        assert record["cover_metadata_complete"] is False
        assert not (PROJECT_ROOT / "reports/academic/academic_report.pdf").exists()


def test_the_report_marks_its_pending_fields(report: str) -> None:
    assert EXTERNAL_INPUT_MARKER in report
    assert "RASCUNHO" in report or "DRAFT" in report


# -- 19 to 20. the phase changed nothing -----------------------------------------


def test_the_phase_changed_no_modelling_decision(record: dict) -> None:
    assert record["model_changed"] is False
    assert record["fit_calls_for_selection"] == 0
    assert record["selection_after_academic_reproduction"] is False
    assert record["threshold_changed"] is False
    assert record["calibration_changed"] is False
    assert record["academic_reproduction_only"] is True
    assert record["evaluated_configurations"] == 1


def test_the_frozen_artefacts_are_untouched() -> None:
    """The anchors this delivery describes must still be the ones on disk."""
    import hashlib

    raw = next((PROJECT_ROOT / "data/raw").glob("*.csv"))
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == RAW_SHA256

    pipeline = PROJECT_ROOT / "artifacts/model/churn_pipeline.joblib"
    assert (
        hashlib.sha256(pipeline.read_bytes()).hexdigest()
        == "574fde36c6e2e991de7c3dfdab21981dc41eae2810d003ecbfb179dd504dc3d8"
    )

    policy = json.loads((PROJECT_ROOT / "reports/decision_policy.json").read_text("utf-8"))
    assert policy["threshold"]["final_threshold"] == FROZEN_THRESHOLD
    assert policy["calibration"]["calibration_policy"] == "NONE"
    assert policy["decision_rule"]["comparison"] == ">="


# -- the record itself -----------------------------------------------------------


def test_the_record_is_reproducible_and_canonical(record: dict) -> None:
    rebuilt = build_record()
    assert rebuilt == record, "the record no longer matches the repository"
    assert canonical_json(record) == RECORD_PATH.read_text(encoding="utf-8")


def test_every_record_invariant_holds(record: dict) -> None:
    failed = [(name, detail) for name, holds, detail in record_invariants(record) if not holds]
    assert not failed, f"record invariants failed: {failed}"


def test_the_notebook_is_versioned_without_outputs(notebook: dict) -> None:
    """Repository convention: notebooks are committed clean, as 01_eda.ipynb is."""
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(not cell["outputs"] for cell in code_cells)
    assert all(cell["execution_count"] is None for cell in code_cells)


def test_the_academic_package_documents_exist() -> None:
    for relative in (
        "academic/README.md",
        "academic/colab_instructions.md",
        "academic/video_script.md",
        "academic/presentation_outline.md",
        "reports/academic/academic_report.md",
        "reports/academic/academic_report.html",
        "reports/academic/submission_checklist.md",
        "reports/academic/rubric_self_audit.md",
    ):
        assert (PROJECT_ROOT / relative).is_file(), f"missing deliverable: {relative}"


def test_the_report_figures_reference_existing_artefacts(report: str) -> None:
    """Figures are referenced, never duplicated, so a broken link is a real break."""
    base = REPORT_PATH.parent
    references = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    assert references, "the report should carry figures"
    for reference in references:
        assert (base / reference).resolve().is_file(), f"missing figure: {reference}"


def test_the_rubric_audit_does_not_invent_points() -> None:
    text = (PROJECT_ROOT / "reports/academic/rubric_self_audit.md").read_text(encoding="utf-8")
    assert "points_not_reliably_extracted_from_source = true" in text
    for proven in ("**8 pontos**", "**5 pontos**", "**7 pontos**"):
        assert proven in text


def test_the_notebook_uses_a_public_no_auth_source(notebook_source: str, record: dict) -> None:
    url = record["notebook"]["dataset_source_url"]
    # The notebook wraps the URL across two adjacent string literals to stay inside the
    # line length, so adjacent-literal concatenation is collapsed before matching.
    joined = re.sub(r'"\s*\n\s*"', "", notebook_source)
    assert url in joined
    assert record["dataset_provenance"]["dataset_source_public"] is True
    assert record["dataset_provenance"]["dataset_source_requires_credentials"] is False
    assert url.startswith("https://")
    assert record["notebook"]["dataset_source_requires_authentication"] is False
    assert "kaggle" not in url.lower(), "the source must not require a Kaggle account"


def test_the_notebook_uses_no_sampling_explainer(notebook_source: str) -> None:
    """The decomposition is exact; a sampling estimator would be a worse answer."""
    lowered = notebook_source.lower()
    assert "import shap" not in lowered
    assert "import lime" not in lowered
    assert "EXACT" in notebook_source or "exata" in lowered


# -- dataset provenance: two identities, kept apart ------------------------------


def test_both_dataset_digests_exist_under_distinct_names(
    record: dict, notebook_source: str
) -> None:
    """The mirror's bytes and the project's canonical bytes are different things.

    Recording one digest and calling it "the raw dataset SHA" would assert a byte
    identity that does not hold: the mirror serves LF, the repository froze CRLF.
    """
    provenance = record["dataset_provenance"]

    assert provenance["downloaded_bytes_sha256"]
    assert provenance["canonicalized_dataset_sha256"]
    assert provenance["downloaded_bytes_sha256"] != provenance["canonicalized_dataset_sha256"], (
        "the two digests must differ; if they were equal no canonicalisation happened"
    )

    # The notebook computes both, under names a reader cannot confuse.
    assert "downloaded_bytes_sha256" in notebook_source
    assert "canonicalized_dataset_sha256" in notebook_source


def test_the_frozen_expectation_applies_to_the_canonicalized_digest(
    record: dict, notebook_source: str
) -> None:
    """The pin is compared against the canonical digest, never the download."""
    provenance = record["dataset_provenance"]

    assert provenance["canonicalized_dataset_sha256"] == RAW_SHA256
    assert provenance["canonicalized_dataset_matches_frozen_raw"] is True
    assert provenance["downloaded_bytes_sha256"] != RAW_SHA256
    assert provenance["downloaded_bytes_sha256_is_not_the_frozen_raw_digest"] is True

    assert "canonicalized_dataset_sha256 != FROZEN_CANONICAL_SHA256" in notebook_source


def test_the_canonicalization_touches_only_newlines(record: dict, notebook_source: str) -> None:
    """Scope is declared, and the notebook proves it rather than asserting it."""
    provenance = record["dataset_provenance"]

    assert provenance["canonicalization_applied"] is True
    assert provenance["canonicalization_scope"] == "newline representation only"
    assert provenance["canonicalization_verified_content_preserving"] is True

    # The proof: with every line terminator removed, the two streams are equal.
    assert "without_line_terminators" in notebook_source
    assert "newline_only" in notebook_source

    forbidden = (".strip()", "sort_values", "sort_index", "reindex", "round(")
    acquisition = notebook_source.split("## 4.")[0]
    found = [token for token in forbidden if token in acquisition]
    assert not found, f"the acquisition step appears to do more than reserialise: {found}"


def test_the_notebook_claims_no_byte_identity_with_the_mirror(
    notebook_source: str, report_prose: str
) -> None:
    """Neither document may imply the download is byte-identical to the frozen file."""
    assert "nao precisa coincidir" in notebook_source or "não precisa coincidir" in notebook_source
    # Probed without the surrounding emphasis markers, which markdown puts around
    # "não" and which are a typographic choice rather than part of the claim.
    assert "entrega os mesmos bytes" in report_prose
    assert "representação" in report_prose
    assert "downloaded_bytes_sha256" in report_prose
    assert "canonicalized_dataset_sha256" in report_prose


def test_the_shape_and_schema_are_verified_after_download(
    record: dict, notebook_source: str
) -> None:
    """A digest proves bytes; this proves the table is the expected table."""
    provenance = record["dataset_provenance"]

    assert provenance["expected_n_rows"] == 7043
    assert len(provenance["expected_columns"]) == 21
    assert provenance["expected_columns"][0] == "customerID"
    assert provenance["expected_columns"][-1] == "Churn"
    assert provenance["shape_and_schema_verified_after_download"] is True

    assert "EXPECTED_N_ROWS" in notebook_source
    assert "EXPECTED_COLUMNS" in notebook_source


def test_the_pdf_determinism_is_recorded_as_engineering_not_a_requirement(
    record: dict,
) -> None:
    assert record["pdf_metadata_normalized_for_determinism"] is True
    assert record["pdf_content_changed_by_normalization"] is False
    assert record["pdf_normalization_scope"] == "/CreationDate and /ModDate only"
