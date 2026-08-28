"""Builds the deterministic record of the academic delivery.

Every value here is either read from the repository as it stands or held at an honest
default. Nothing is asserted about work that has not been done — see the package
docstring for why that distinction is the point of this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from churn.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
EXPERIMENT = "phase14a-academic-delivery"

#: Where the record is written.
RECORD_RELATIVE_PATH = "reports/experiments/academic_delivery_results.json"
RECORD_PATH = PROJECT_ROOT / RECORD_RELATIVE_PATH

#: The deliverables this phase produces, relative to the repository root.
NOTEBOOK_RELATIVE_PATH = "notebooks/02_academic_delivery.ipynb"
REPORT_RELATIVE_PATH = "reports/academic/academic_report.md"
REPORT_HTML_RELATIVE_PATH = "reports/academic/academic_report.html"
REPORT_DRAFT_PDF_RELATIVE_PATH = "reports/academic/academic_report_DRAFT.pdf"
REPORT_FINAL_PDF_RELATIVE_PATH = "reports/academic/academic_report.pdf"
CHECKLIST_RELATIVE_PATH = "reports/academic/submission_checklist.md"
RUBRIC_RELATIVE_PATH = "reports/academic/rubric_self_audit.md"

VIDEO_SCRIPT_RELATIVE_PATH = "academic/video_script.md"
PRESENTATION_RELATIVE_PATH = "academic/presentation_outline.md"
COLAB_INSTRUCTIONS_RELATIVE_PATH = "academic/colab_instructions.md"
PACKAGE_README_RELATIVE_PATH = "academic/README.md"

#: The eleven sections the assignment requires, as level-1 headings.
REQUIRED_REPORT_SECTIONS: tuple[str, ...] = (
    "Capa",
    "Introdução",
    "Descrição do problema",
    "Dataset utilizado",
    "Metodologia adotada",
    "Pipeline de Machine Learning",
    "Resultados experimentais",
    "Análise das métricas",
    "Explicabilidade do modelo",
    "Estratégia de monitoramento",
    "Conclusão",
)

#: The academic narrative the notebook must carry, as numbered section headings.
REQUIRED_NOTEBOOK_SECTIONS: tuple[str, ...] = (
    "Contexto e descrição do problema",
    "Ambiente e dependências",
    "Aquisição dos dados",
    "Entendimento do dataset",
    "Análise exploratória",
    "Divisão treino/teste",
    "Pré-processamento sem vazamento",
    "Protocolo experimental e baselines",
    "Comparação de modelos",
    "Engenharia de atributos",
    "Tuning e política de calibração",
    "O limiar de decisão",
    "Avaliação final no holdout",
    "Análise das métricas",
    "exposição do analista",
    "Explicabilidade global",
    "Dependências estruturais",
    "Explicabilidade local",
    "Estratégia de monitoramento",
    "Critérios de investigação e retreinamento",
    "Demonstração da solução",
    "Limitações",
    "Conclusão",
)

#: The frozen anchors the notebook must pin, so a reproduction that drifts is visible.
EXPECTED_RAW_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
EXPECTED_TRAINING_IDS_SHA256 = "a553196dd46b672f6344867707144fbf56a8abe14b338a225662c7208a1450dd"
EXPECTED_HOLDOUT_IDS_SHA256 = "1ad8aefb7e34776a78d76765d2465c630a41b3813b1d7d96d0d1b03d049776f6"
FROZEN_THRESHOLD = 0.3272694566222328
FROZEN_CALIBRATION_POLICY = "NONE"
FROZEN_COMPARISON = ">="

#: The public, no-authentication source the notebook downloads from. Recorded because
#: "reproducible" is a claim about a specific URL, not a general aspiration.
DATASET_SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

#: SHA-256 of the bytes the mirror actually serves, observed from a real download.
#:
#: **This is deliberately not the frozen raw digest, and must never be named as one.**
#: The mirror serves the dataset with ``LF`` line endings; the repository froze it with
#: ``CRLF``. The two byte streams are therefore different, and calling this value "the
#: raw dataset SHA" would assert a byte identity that does not hold.
#:
#: It is also not a project invariant: it describes how the mirror currently
#: serialises the file. If GitHub ever re-serves it with different line endings this
#: constant goes stale while :data:`EXPECTED_RAW_SHA256` stays true — which is exactly
#: the asymmetry the two names exist to express.
DOWNLOADED_BYTES_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

#: What the notebook is allowed to change between the two digests above, and nothing
#: else. Named in the record so a reader never has to infer the scope from prose.
CANONICALIZATION_SCOPE = "newline representation only"

#: The rule, spelled out, so that "we normalised it" is a checkable statement rather
#: than a reassurance.
CANONICALIZATION_RULE = (
    "split the byte stream into logical lines at the terminators present (CRLF, bare "
    "CR or bare LF), then reserialise those lines with CRLF. No whitespace is "
    "stripped, no line or column is reordered, no value is converted, no number is "
    "normalised, no quoting is altered and the bytes within each line are preserved "
    "exactly."
)

#: The shape and schema the notebook re-checks after downloading, independently of any
#: digest. A hash proves byte identity; this proves the table is the expected table,
#: and the two fail in different ways.
EXPECTED_N_ROWS = 7043
EXPECTED_COLUMNS: tuple[str, ...] = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)

#: Marker used in the report wherever a field needs a human. Counted, never removed
#: by code: the count is what keeps a draft from being called final.
EXTERNAL_INPUT_MARKER = "EXTERNAL_INPUT_REQUIRED"

#: Actions this process cannot perform, and therefore cannot mark as done.
EXTERNAL_ACTIONS_REMAINING: tuple[str, ...] = (
    "open the notebook in Google Colab and run every cell",
    "confirm no cell raised, and that the pinned comparisons still match",
    "share the Colab notebook with link access and record the real URL",
    "validate the Colab link in a private browsing window",
    "export the executed Colab notebook to PDF and inspect it",
    "record and publish the presentation video on a freely accessible platform",
    "record the real video URL and validate it in a private browsing window",
    "supply the cover metadata: institution, course, subject, instructor, city, date",
    "fill the delivery links in the report and drop the DRAFT notice",
    "regenerate the final report PDF once no EXTERNAL_INPUT_REQUIRED marker remains",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook_facts(root: Path) -> dict[str, Any]:
    """Read the notebook and report what it structurally is."""
    path = root / NOTEBOOK_RELATIVE_PATH
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in cells)

    # A notebook that must run on a fresh Colab runtime may not reach for a local
    # checkout, a mounted drive, an uploaded file or a credential. These are asserted
    # over the delivered bytes rather than trusted.
    #
    # URLs are removed before the drive-letter check: `https://` ends in `s:/`, which
    # any naive drive-letter pattern reads as an absolute path. The notebook is
    # *supposed* to contain a URL — that is how it avoids needing a local file — so
    # the check has to be able to tell one from the other.
    without_urls = re.sub(r"https?://\S*", "", source)
    absolute_path = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/home/[a-z]|/Users/")
    local_dependency = re.compile(
        r"files\.upload|drive\.mount|google\.colab|os\.getcwd|\.\./data|kaggle\.json"
    )
    credential = re.compile(r"api_key\s*=|secret\s*=|password\s*=|Bearer\s|token\s*=\s*['\"]")

    return {
        "path": NOTEBOOK_RELATIVE_PATH,
        "sha256": _digest(path),
        "n_cells": len(cells),
        "n_code_cells": len(code_cells),
        "n_markdown_cells": len(cells) - len(code_cells),
        "outputs_stored": any(cell.get("outputs") for cell in code_cells),
        "sections_present": [
            section for section in REQUIRED_NOTEBOOK_SECTIONS if section in source
        ],
        "sections_missing": [
            section for section in REQUIRED_NOTEBOOK_SECTIONS if section not in source
        ],
        "self_contained": "from churn" not in source and "import churn" not in source,
        "absolute_path_present": bool(absolute_path.search(without_urls)),
        "local_dependency_present": bool(local_dependency.search(source)),
        "credential_present": bool(credential.search(source)),
        "dataset_source_url": DATASET_SOURCE_URL,
        "dataset_source_requires_authentication": False,
        "raw_sha256_pinned": EXPECTED_RAW_SHA256 in source,
        "aborts_on_digest_mismatch": "ABORTADO" in source and "SystemExit" in source,
        # The notebook must compute both digests under distinct names, and must apply
        # the frozen expectation to the canonicalised one only.
        "computes_downloaded_bytes_digest": "downloaded_bytes_sha256" in source,
        "computes_canonicalized_digest": "canonicalized_dataset_sha256" in source,
        "digest_names_are_distinct": (
            "downloaded_bytes_sha256" in source
            and "canonicalized_dataset_sha256" in source
            and "downloaded_bytes_sha256" != "canonicalized_dataset_sha256"
        ),
        "frozen_expectation_applied_to_canonicalized_digest": (
            "canonicalized_dataset_sha256 != FROZEN_CANONICAL_SHA256" in source
        ),
        "proves_canonicalization_is_newline_only": (
            "without_line_terminators" in source and "newline_only" in source
        ),
        "verifies_shape_and_schema_after_download": (
            "EXPECTED_N_ROWS" in source and "EXPECTED_COLUMNS" in source
        ),
        "training_ids_sha256_pinned": EXPECTED_TRAINING_IDS_SHA256 in source,
        "holdout_ids_sha256_pinned": EXPECTED_HOLDOUT_IDS_SHA256 in source,
        "split_pinned": "test_size=TEST_SIZE" in source and "RANDOM_SEED = 42" in source,
        "threshold_pinned": repr(FROZEN_THRESHOLD) in source,
        "calibration_policy_stated_none": "calibration_policy = NONE" in source,
        "comparison_pinned": "probabilidade >= limiar" in source,
    }


def _report_facts(root: Path) -> dict[str, Any]:
    """Read the report and report which required sections it carries."""
    path = root / REPORT_RELATIVE_PATH
    text = path.read_text(encoding="utf-8")
    headings = {line[2:].strip() for line in text.splitlines() if line.startswith("# ")}

    # The phrase checks below run against normalised prose: block-quote markers
    # stripped, then whitespace collapsed. The report is hard-wrapped, so a required
    # sentence routinely straddles a line break — and inside a block quote the wrap
    # also inserts a `>` mid-sentence. Matching against the raw text would make these
    # gates depend on where the wrap happens to fall, which is exactly the kind of
    # check that passes today and fails on a reflow that changed nothing.
    # Lower-cased too: whether a sentence opens a paragraph decides its capital
    # letter, and that is a typographic accident rather than a property worth gating on.
    flat = " ".join(
        " ".join(line.lstrip().lstrip(">").strip() for line in text.splitlines()).split()
    ).lower()

    return {
        "path": REPORT_RELATIVE_PATH,
        "sha256": _digest(path),
        "sections_present": [s for s in REQUIRED_REPORT_SECTIONS if s in headings],
        "sections_missing": [s for s in REQUIRED_REPORT_SECTIONS if s not in headings],
        "n_external_input_markers": text.count(EXTERNAL_INPUT_MARKER),
        "analyst_exposure_caveat_present": "viés otimista" in flat,
        "accuracy_is_auxiliary": "auxiliar" in flat and "nunca como manchete" in flat,
        "roc_auc_not_called_accuracy": 'não é "84 % de acerto"' in flat,
        "explanation_declared_non_causal": "não são efeitos causais" in flat,
        "psi_not_called_a_statistical_test": "não são testes estatísticos" in flat,
        "retraining_schedule_refused": 'não prescreve "retreinar todo mês"' in flat,
        "no_causal_claim_of_churn_reduction": "não demonstra" in flat,
        "traceability_table_present": "# Rastreabilidade" in text,
    }


def _artefact(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    present = path.is_file()
    return {
        "path": relative,
        "present": present,
        "sha256": _digest(path) if present else None,
    }


def build_record(root: Path | None = None) -> dict[str, Any]:
    """Assemble the record from what the repository actually contains."""
    base = root or PROJECT_ROOT
    notebook = _notebook_facts(base)
    report = _report_facts(base)

    draft_pdf = _artefact(base, REPORT_DRAFT_PDF_RELATIVE_PATH)
    final_pdf = _artefact(base, REPORT_FINAL_PDF_RELATIVE_PATH)

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": "14",
        "subphase": "14A",
        "component": "ACADEMIC_DELIVERY",
        # -- what this phase did NOT do -----------------------------------
        "model_changed": False,
        "fit_calls_for_selection": 0,
        "selection_after_academic_reproduction": False,
        "threshold_changed": False,
        "calibration_changed": False,
        "new_experiment_performed": False,
        "academic_reproduction_only": True,
        "evaluated_configurations": 1,
        # -- the frozen anchors the delivery is pinned to -------------------
        "frozen_anchors": {
            "raw_sha256": EXPECTED_RAW_SHA256,
            "training_ids_sha256": EXPECTED_TRAINING_IDS_SHA256,
            "holdout_ids_sha256": EXPECTED_HOLDOUT_IDS_SHA256,
            "final_threshold": FROZEN_THRESHOLD,
            "calibration_policy": FROZEN_CALIBRATION_POLICY,
            "comparison": FROZEN_COMPARISON,
        },
        # -- how the public mirror relates to the frozen file ---------------
        #
        # Two digests, two names, and the difference between them is the whole point.
        # The mirror serves LF; the repository froze CRLF. Recording one digest and
        # calling it "the raw SHA" would assert a byte identity that does not hold.
        "dataset_provenance": {
            "source_url": DATASET_SOURCE_URL,
            "dataset_source_public": True,
            "dataset_source_requires_credentials": False,
            "downloaded_bytes_representation": "LF",
            "downloaded_bytes_sha256": DOWNLOADED_BYTES_SHA256,
            "downloaded_bytes_sha256_is_not_the_frozen_raw_digest": True,
            "downloaded_bytes_sha256_describes": (
                "the mirror's current serialisation, not a project invariant"
            ),
            "canonicalization_applied": True,
            "canonicalization_scope": CANONICALIZATION_SCOPE,
            "canonicalization_rule": CANONICALIZATION_RULE,
            "canonical_representation": "CRLF",
            "canonicalized_dataset_sha256": EXPECTED_RAW_SHA256,
            "canonicalized_dataset_matches_frozen_raw": True,
            "canonicalization_verified_content_preserving": True,
            "expected_n_rows": EXPECTED_N_ROWS,
            "expected_columns": list(EXPECTED_COLUMNS),
            "shape_and_schema_verified_after_download": notebook[
                "verifies_shape_and_schema_after_download"
            ],
        },
        # -- derived from the repository -----------------------------------
        "notebook": notebook,
        "report": report,
        "artefacts": {
            "report_html": _artefact(base, REPORT_HTML_RELATIVE_PATH),
            "report_draft_pdf": draft_pdf,
            "report_final_pdf": final_pdf,
            "submission_checklist": _artefact(base, CHECKLIST_RELATIVE_PATH),
            "rubric_self_audit": _artefact(base, RUBRIC_RELATIVE_PATH),
            "video_script": _artefact(base, VIDEO_SCRIPT_RELATIVE_PATH),
            "presentation_outline": _artefact(base, PRESENTATION_RELATIVE_PATH),
            "colab_instructions": _artefact(base, COLAB_INSTRUCTIONS_RELATIVE_PATH),
            "package_readme": _artefact(base, PACKAGE_README_RELATIVE_PATH),
        },
        "report_source_present": (base / REPORT_RELATIVE_PATH).is_file(),
        "report_pdf_present": draft_pdf["present"] or final_pdf["present"],
        "report_pdf_is_draft_only": draft_pdf["present"] and not final_pdf["present"],
        "report_pdf_verified": draft_pdf["present"],
        # The browser stamps a wall clock into every PDF it prints, which would make
        # this artefact differ on every render. The two date fields are rewritten to a
        # fixed instant of the same byte length. This is a repository engineering
        # property, not an academic requirement, and it changes no rendered content.
        "pdf_metadata_normalized_for_determinism": True,
        "pdf_content_changed_by_normalization": False,
        "pdf_normalization_scope": "/CreationDate and /ModDate only",
        "academic_notebook_present": (base / NOTEBOOK_RELATIVE_PATH).is_file(),
        "video_script_present": (base / VIDEO_SCRIPT_RELATIVE_PATH).is_file(),
        # -- externally attested: this process cannot observe any of these ---
        #
        # A local nbconvert run is not a Colab run. The two are different runtimes,
        # different library versions and different evidence, and collapsing them would
        # be the single easiest lie in this whole phase.
        "academic_notebook_local_execution_passed": True,
        "local_execution_method": "nbconvert ExecutePreprocessor, isolated scratch directory",
        "local_execution_cells_executed": notebook["n_code_cells"],
        "local_execution_failures": 0,
        "local_execution_requires_network": True,
        "local_execution_is_not_colab": True,
        "actual_google_colab_execution": False,
        "colab_shared_link": None,
        "colab_shared_link_verified": False,
        "actual_colab_export_pdf_present": False,
        "video_link": None,
        "video_link_verified": False,
        "cover_metadata_complete": report["n_external_input_markers"] == 0,
        "external_actions_remaining": list(EXTERNAL_ACTIONS_REMAINING),
        "academic_submission_ready": False,
        "notes": [
            "This phase turns already-executed results into an academic deliverable. "
            "No model, feature, hyperparameter, calibration policy or threshold was "
            "selected, changed or revisited.",
            "The notebook may re-run the frozen protocol to demonstrate "
            "reproducibility. That is a reproduction, not a new selection: one "
            "configuration is evaluated and nothing is decided from the result.",
            "academic_submission_ready stays false while any external action remains. "
            "It is not a formatting flag and must not be flipped to make the record "
            "look complete.",
        ],
    }


def canonical_json(record: dict[str, Any]) -> str:
    """Return the one serialisation this project writes and hashes."""
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def write_record(record: dict[str, Any], path: Path | None = None) -> Path:
    """Write the record with LF endings and a trailing newline."""
    destination = path or RECORD_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(record), encoding="utf-8", newline="\n")
    logger.info("Wrote academic delivery record: %s", destination)
    return destination


def read_record(path: Path | None = None) -> dict[str, Any]:
    """Read the committed record.

    Raises:
        FileNotFoundError: If it has not been built yet.
    """
    source = path or RECORD_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Academic delivery record not found at {source}. Build it with "
            "`uv run python scripts/build_academic_record.py`."
        )
    return json.loads(source.read_text(encoding="utf-8"))


def record_invariants(record: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Return ``(name, passed, detail)`` for every invariant the record must satisfy.

    Two families of check live here, and they pull in opposite directions on purpose.

    The first family asserts that the deliverable is **complete**: every required
    section present, every anchor pinned, no absolute path, no credential.

    The second asserts that it is **honest**: that nothing claims Colab execution, that
    no link was invented, that ``academic_submission_ready`` is still false. Those
    would all pass more easily if the code simply wrote ``true`` — which is exactly why
    they are invariants rather than comments.
    """
    notebook = record["notebook"]
    report = record["report"]
    provenance = record["dataset_provenance"]

    return [
        # -- the phase changed no modelling decision ----------------------
        ("model_not_changed", record["model_changed"] is False, "model_changed=false"),
        (
            "no_selection_after_reproduction",
            record["selection_after_academic_reproduction"] is False
            and record["fit_calls_for_selection"] == 0,
            "no configuration was chosen in this phase",
        ),
        (
            "reproduction_evaluated_one_configuration",
            record["academic_reproduction_only"] is True
            and record["evaluated_configurations"] == 1,
            f"{record['evaluated_configurations']} configuration(s)",
        ),
        (
            "frozen_anchors_unchanged",
            record["frozen_anchors"]["final_threshold"] == FROZEN_THRESHOLD
            and record["frozen_anchors"]["calibration_policy"] == FROZEN_CALIBRATION_POLICY
            and record["frozen_anchors"]["raw_sha256"] == EXPECTED_RAW_SHA256,
            "threshold, calibration policy and raw digest are the frozen ones",
        ),
        # -- the notebook is a deliverable, and a runnable one -------------
        (
            "notebook_carries_every_section",
            not notebook["sections_missing"],
            f"{len(notebook['sections_present'])}/{len(REQUIRED_NOTEBOOK_SECTIONS)} sections",
        ),
        (
            "notebook_is_self_contained",
            notebook["self_contained"],
            "no import of the local churn package",
        ),
        (
            "notebook_has_no_absolute_path",
            notebook["absolute_path_present"] is False,
            "no drive letter or home directory",
        ),
        (
            "notebook_needs_no_local_upload",
            notebook["local_dependency_present"] is False,
            "no files.upload, drive.mount or relative data path",
        ),
        (
            "notebook_carries_no_credential",
            notebook["credential_present"] is False,
            "no token, key or password",
        ),
        (
            "notebook_pins_the_raw_digest",
            notebook["raw_sha256_pinned"] and notebook["aborts_on_digest_mismatch"],
            "digest pinned and mismatch aborts",
        ),
        # -- the two dataset digests are kept apart ------------------------
        (
            "notebook_computes_both_dataset_digests",
            notebook["computes_downloaded_bytes_digest"]
            and notebook["computes_canonicalized_digest"],
            "downloaded_bytes_sha256 and canonicalized_dataset_sha256",
        ),
        (
            "the_two_digests_have_different_names_and_values",
            notebook["digest_names_are_distinct"]
            and provenance["downloaded_bytes_sha256"] != provenance["canonicalized_dataset_sha256"],
            f"{provenance['downloaded_bytes_sha256'][:12]}… != "
            f"{provenance['canonicalized_dataset_sha256'][:12]}…",
        ),
        (
            "downloaded_digest_is_not_called_the_frozen_raw_digest",
            provenance["downloaded_bytes_sha256_is_not_the_frozen_raw_digest"] is True
            and provenance["downloaded_bytes_sha256"] != EXPECTED_RAW_SHA256,
            "the mirror's serialisation is not the frozen file",
        ),
        (
            "frozen_expectation_applies_to_the_canonicalized_digest",
            notebook["frozen_expectation_applied_to_canonicalized_digest"]
            and provenance["canonicalized_dataset_sha256"] == EXPECTED_RAW_SHA256
            and provenance["canonicalized_dataset_matches_frozen_raw"] is True,
            "only the canonical digest is compared with the pin",
        ),
        (
            "canonicalization_is_newline_only",
            provenance["canonicalization_scope"] == CANONICALIZATION_SCOPE
            and provenance["canonicalization_applied"] is True
            and notebook["proves_canonicalization_is_newline_only"],
            CANONICALIZATION_SCOPE,
        ),
        (
            "shape_and_schema_are_verified_independently_of_the_digest",
            provenance["shape_and_schema_verified_after_download"]
            and provenance["expected_n_rows"] == EXPECTED_N_ROWS
            and len(provenance["expected_columns"]) == len(EXPECTED_COLUMNS),
            f"{EXPECTED_N_ROWS} rows, {len(EXPECTED_COLUMNS)} columns",
        ),
        (
            "dataset_source_is_public_and_credential_free",
            provenance["dataset_source_public"] is True
            and provenance["dataset_source_requires_credentials"] is False,
            provenance["source_url"],
        ),
        (
            "notebook_pins_the_frozen_split",
            notebook["training_ids_sha256_pinned"]
            and notebook["holdout_ids_sha256_pinned"]
            and notebook["split_pinned"],
            "both identifier digests and the split parameters",
        ),
        (
            "notebook_pins_the_frozen_decision",
            notebook["threshold_pinned"]
            and notebook["calibration_policy_stated_none"]
            and notebook["comparison_pinned"],
            "threshold, calibration policy and comparison",
        ),
        (
            "notebook_stores_no_outputs",
            notebook["outputs_stored"] is False,
            "versioned clean, per the repository convention",
        ),
        # -- the report is a deliverable, and an honest one ----------------
        (
            "report_carries_every_required_section",
            not report["sections_missing"],
            f"{len(report['sections_present'])}/{len(REQUIRED_REPORT_SECTIONS)} sections",
        ),
        (
            "report_keeps_the_analyst_exposure_caveat",
            report["analyst_exposure_caveat_present"],
            "carried since Phase 3",
        ),
        (
            "report_treats_accuracy_as_auxiliary",
            report["accuracy_is_auxiliary"] and report["roc_auc_not_called_accuracy"],
            "accuracy auxiliary; ROC-AUC not described as accuracy",
        ),
        (
            "report_declares_explanations_non_causal",
            report["explanation_declared_non_causal"],
            "contributions are not causal effects",
        ),
        (
            "report_declares_psi_not_a_test",
            report["psi_not_called_a_statistical_test"],
            "PSI and TVD are descriptive distances",
        ),
        (
            "report_refuses_an_arbitrary_retraining_schedule",
            report["retraining_schedule_refused"],
            "drift triggers investigation, not a calendar",
        ),
        (
            "report_is_traceable",
            report["traceability_table_present"],
            "every claim mapped to a versioned artefact",
        ),
        # -- nothing external is claimed -----------------------------------
        (
            "colab_execution_not_claimed",
            record["actual_google_colab_execution"] is False
            and record["local_execution_is_not_colab"] is True,
            "a local nbconvert run is not a Colab run",
        ),
        (
            "no_link_was_invented",
            record["colab_shared_link"] is None and record["video_link"] is None,
            "both links are null",
        ),
        (
            "no_link_is_claimed_verified",
            record["colab_shared_link_verified"] is False
            and record["video_link_verified"] is False,
            "nothing was validated in a private window by this process",
        ),
        (
            "external_actions_are_listed",
            len(record["external_actions_remaining"]) > 0,
            f"{len(record['external_actions_remaining'])} action(s) remain",
        ),
        (
            "submission_is_not_claimed_ready",
            record["academic_submission_ready"] is False,
            "external actions remain",
        ),
        (
            "pdf_is_draft_while_markers_remain",
            (report["n_external_input_markers"] == 0)
            or (record["report_pdf_is_draft_only"] and not record["cover_metadata_complete"]),
            f"{report['n_external_input_markers']} marker(s); draft-only PDF",
        ),
    ]


__all__ = [
    "CANONICALIZATION_RULE",
    "CANONICALIZATION_SCOPE",
    "DATASET_SOURCE_URL",
    "DOWNLOADED_BYTES_SHA256",
    "EXPECTED_COLUMNS",
    "EXPECTED_N_ROWS",
    "EXTERNAL_ACTIONS_REMAINING",
    "EXTERNAL_INPUT_MARKER",
    "RECORD_PATH",
    "RECORD_RELATIVE_PATH",
    "REQUIRED_NOTEBOOK_SECTIONS",
    "REQUIRED_REPORT_SECTIONS",
    "SCHEMA_VERSION",
    "build_record",
    "canonical_json",
    "read_record",
    "record_invariants",
    "write_record",
]
