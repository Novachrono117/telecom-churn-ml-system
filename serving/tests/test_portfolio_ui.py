"""The demo page: opt-in, self-contained, and never a second source of truth.

No browser automation. The properties worth guarding here are static ones — which
routes exist, what the HTML declares, what the script is allowed to know — and a
headless browser would add a dependency and a flake surface without checking any of
them better.

The two claims that matter most:

**Turning the demo off leaves Phase 11 exactly as it was.** The recorded route list
is verified byte for byte elsewhere; here it is verified behaviourally.

**The frontend presents and does not decide.** The threshold, the comparison, the
calibration policy and every metric arrive from the API. A page that hardcoded them
would drift silently the day one of them changed, and would be showing a number that
no artefact backs.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from churn.serving.api import create_app
from churn.serving.portfolio import REQUIRED_ASSETS, default_static_path
from churn.serving.settings import ServingSettings

DEMO = "/demo"
STATIC = "/demo/static"

#: The 19 contracted features, as the form must present them.
FEATURES = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
)


@pytest.fixture(scope="session")
def ui_settings(policy_path: Path) -> ServingSettings:
    return ServingSettings(policy_path=policy_path, portfolio_ui_enabled=True)


@pytest.fixture
def demo(ui_settings: ServingSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=ui_settings)) as client:
        yield client


@pytest.fixture(scope="session")
def html() -> str:
    return (default_static_path() / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def script() -> str:
    return (default_static_path() / "js" / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def stylesheet() -> str:
    return (default_static_path() / "css" / "app.css").read_text(encoding="utf-8")


def _code_only(source: str) -> str:
    """Strip comments so a check reads the program, not the prose describing it.

    Both files document the properties these tests enforce — the script's header says
    "no localStorage", the page's footer says "does not recommend retention actions".
    A raw substring search would flag the very sentences that promise the behaviour.
    """
    without_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    without_line = re.sub(r"^\s*//.*$", " ", without_block, flags=re.M)
    return re.sub(r"<!--.*?-->", " ", without_line, flags=re.S)


def _flat(source: str) -> str:
    """Collapse whitespace so a phrase that the source wraps still matches."""
    return re.sub(r"\s+", " ", source)


def _function_body(source: str, name: str) -> str:
    """Return the body of ``function <name>(...) { ... }``, brace-matched.

    A regex cannot do this: the body contains braces, string literals with braces, and
    nested functions. Counting braces from the opening one is exact for this file,
    which contains no brace inside a string or a regex literal — asserted below by
    the fact that every extraction closes.
    """
    start = source.index("function " + name + "(")
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening : index + 1]
    raise AssertionError(f"function {name} does not close")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Opt-in.
# --------------------------------------------------------------------------- #


def test_the_demo_is_absent_when_the_ui_is_off(client: TestClient) -> None:
    assert client.get(DEMO).status_code == 404
    assert client.get(f"{STATIC}/css/app.css").status_code == 404
    assert client.get("/api/v1/portfolio").status_code == 404
    assert client.post("/api/v1/explain", json={}).status_code == 404


def test_the_phase_11_route_list_is_unchanged_by_default(client: TestClient) -> None:
    """The published contract, unaltered by Phase 13 code existing."""
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {
        "/health/live",
        "/health/ready",
        "/api/v1/model",
        "/api/v1/predict",
        "/api/v1/predict/batch",
    }


def test_enabling_the_demo_adds_only_the_demo(demo: TestClient) -> None:
    paths = set(demo.get("/openapi.json").json()["paths"])

    assert paths == {
        "/health/live",
        "/health/ready",
        "/api/v1/model",
        "/api/v1/predict",
        "/api/v1/predict/batch",
        "/api/v1/explain",
        "/api/v1/portfolio",
    }
    # The page and its assets are served but deliberately not in the API schema.
    assert DEMO not in paths


def test_the_phase_11_endpoints_answer_identically_with_the_demo_on(
    client: TestClient, demo: TestClient
) -> None:
    for route in ("/health/live", "/api/v1/model"):
        assert client.get(route).json() == demo.get(route).json()


# --------------------------------------------------------------------------- #
# The page and its assets.
# --------------------------------------------------------------------------- #


def test_the_page_and_its_assets_are_served(demo: TestClient) -> None:
    page = demo.get(DEMO)
    css = demo.get(f"{STATIC}/css/app.css")
    js = demo.get(f"{STATIC}/js/app.js")

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert css.status_code == 200
    assert js.status_code == 200
    assert len(css.content) > 1000
    assert len(js.content) > 1000


def test_every_declared_asset_exists(demo: TestClient) -> None:
    for name in REQUIRED_ASSETS:
        assert (default_static_path() / name).is_file(), name


def test_the_static_mount_refuses_to_escape_its_root(demo: TestClient) -> None:
    """No route takes a filename, and the mount will not walk upwards either."""
    for attempt in (
        f"{STATIC}/../../pyproject.toml",
        f"{STATIC}/..%2f..%2fpyproject.toml",
        f"{STATIC}/css/../../../uv.lock",
    ):
        response = demo.get(attempt)
        assert response.status_code in (403, 404, 400), attempt
        assert "[project]" not in response.text


# --------------------------------------------------------------------------- #
# What the page contains.
# --------------------------------------------------------------------------- #


def test_the_page_requests_nothing_from_the_internet(html: str, stylesheet: str) -> None:
    """Zero external assets, so the demo runs fully offline."""
    for needle in ("http://", "https://", "//cdn", "fonts.googleapis", "fonts.gstatic"):
        assert needle not in html, needle
        assert needle not in stylesheet, needle

    for match in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert match.startswith(("/demo/static/", "#")), match


def test_the_page_carries_no_analytics_and_no_storage(html: str, script: str) -> None:
    """No telemetry, and nothing survives a refresh.

    Checked over the code with comments stripped, and — for the page — over the tags
    rather than the prose: the footer tells the visitor there is no analytics, and a
    naive search would flag that sentence as evidence of analytics.
    """
    storage_apis = (
        "localStorage",
        "sessionStorage",
        "indexedDB",
        "document.cookie",
        "navigator.sendBeacon",
    )
    trackers = (
        "gtag(",
        "googletagmanager",
        "google-analytics",
        "mixpanel",
        "segment.com",
        "posthog",
        "sentry",
        "hotjar",
        "plausible",
    )
    page = _code_only(html)
    code = _code_only(script)

    for needle in (*storage_apis, *trackers):
        assert needle not in code, needle
        assert needle not in page, needle

    # Every script the page loads is this service's own.
    for source in re.findall(r"<script[^>]*src=\"([^\"]+)\"", html):
        assert source == "/demo/static/js/app.js", source
    assert html.count("<script") == 1


def test_the_form_offers_every_contracted_feature(html: str, script: str) -> None:
    """All 19, and the identifier and target are nowhere on the page."""
    combined = html + script

    for feature in FEATURES:
        assert feature in combined, feature

    assert "customerID" not in _code_only(combined)
    assert not re.search(r'name="Churn"', html)
    assert '"Churn"' not in _code_only(script)


def test_the_numeric_inputs_have_associated_labels(html: str) -> None:
    for feature in ("tenure", "MonthlyCharges", "TotalCharges"):
        assert f'for="{feature}"' in html, feature
        assert f'id="{feature}"' in html, feature


def test_the_page_is_accessible_in_the_basic_ways(html: str) -> None:
    assert '<html lang="en">' in html
    assert 'class="skip-link"' in html
    assert "<h1" in html and "<h2" in html
    assert 'aria-live="polite"' in html
    assert 'aria-live="assertive"' in html
    assert 'role="alert"' in html
    assert "<fieldset" in html and "<legend>" in html
    assert 'type="submit"' in html
    assert "<button" in html
    assert "aria-labelledby=" in html


def test_the_stylesheet_defines_focus_states_and_breakpoints(stylesheet: str) -> None:
    assert ":focus-visible" in stylesheet
    assert "@media (max-width: 1080px)" in stylesheet
    assert "@media (max-width: 720px)" in stylesheet
    assert "prefers-reduced-motion" in stylesheet


def test_the_page_uses_system_fonts_only(stylesheet: str) -> None:
    assert "@font-face" not in stylesheet
    assert "-apple-system" in stylesheet


# --------------------------------------------------------------------------- #
# The frontend holds no authority.
# --------------------------------------------------------------------------- #


def test_the_script_hardcodes_no_model_constant(script: str, html: str) -> None:
    """The threshold, the fingerprint and the metrics come from the API.

    A hardcoded constant would be a second source of truth, and the day the backend
    changed it the page would display a number no artefact backs.
    """
    combined = script + html

    assert "0.3272694566222328" not in combined
    assert "0.3273" not in combined
    assert "a57568e1" not in combined
    assert "574fde36" not in combined
    for metric in ("0.633702", "0.842034", "0.616438", "0.537849", "0.721925"):
        assert metric not in combined, metric


def test_the_script_never_recomputes_a_decision(script: str) -> None:
    """No threshold comparison, no sigmoid, no rounding before a decision."""
    code = _code_only(script)

    assert "Math.exp" not in code
    assert "sigmoid" not in code
    assert ">= threshold" not in code
    assert "> threshold" not in code
    # The one arithmetic it does is layout: a marker position on an axis.
    assert "drawScale" in code


def test_the_script_calls_only_this_services_endpoints(script: str) -> None:
    endpoints = set(re.findall(r'"(/(?:api|health)/[^"]*)"', script))

    assert endpoints == {
        "/api/v1/model",
        "/api/v1/explain",
        "/api/v1/portfolio",
        "/api/v1/monitoring",
        "/health/ready",
    }
    assert 'fetch("http' not in script


def test_the_page_invents_no_risk_bands_and_no_recommendations(html: str, script: str) -> None:
    """One justified threshold means two sides, not three arbitrary bands."""
    combined = _flat(_code_only(html) + _code_only(script)).lower()

    for needle in ("high risk", "medium risk", "low risk", "risk level", "risk band"):
        assert needle not in combined, needle
    for needle in ("offer a discount", "call this customer", "retention offer", "next best action"):
        assert needle not in combined, needle
    # Two sides of one justified threshold, named as such.
    assert "below threshold" in combined
    assert "at/above threshold" in combined


def test_the_page_does_not_claim_certainty(html: str, script: str) -> None:
    """calibration_policy is NONE, so the score is a ranking position."""
    combined = _flat(_code_only(html) + _code_only(script)).lower()

    for needle in ("certainty", "certain to", "will definitely", "guaranteed", "% accurate"):
        assert needle not in combined, needle
    assert "calibration-note" in combined


def test_the_synthetic_examples_are_labelled_as_synthetic(html: str, script: str) -> None:
    """No real row, no identifier, and no outcome claimed before the model is asked."""
    page = _flat(html)

    assert "Synthetic example A" in page
    assert "Synthetic example B" in page
    assert "invented for demonstration" in page
    assert "not rows from any dataset" in page
    combined = _flat(_code_only(html) + _code_only(script)).lower()
    assert "high-risk customer" not in combined
    assert "loyal customer" not in combined


# --------------------------------------------------------------------------- #
# Monitoring, as the page shows it.
# --------------------------------------------------------------------------- #


def test_the_page_cannot_bypass_the_small_window_policy(script: str) -> None:
    """It renders what the endpoint returns and reconstructs nothing that was withheld."""
    code = _flat(_code_only(script))

    assert "details_suppressed" in code
    assert "suppressed by the monitoring privacy policy" in code
    # It never reaches for the fields a suppressed window does not carry.
    for suppressed in ("bin_counts", "count_by_level", "n_distinct_unseen_observed"):
        assert suppressed not in code, suppressed


def test_the_page_reports_monitoring_off_honestly(demo: TestClient, script: str) -> None:
    """With monitoring disabled the endpoint is absent, and the page says so."""
    assert demo.get("/api/v1/monitoring").status_code == 404
    assert "Monitoring disabled for this deployment." in script


def test_the_demo_and_monitoring_coexist(policy_path: Path, record: dict[str, Any]) -> None:
    settings = ServingSettings(
        policy_path=policy_path, portfolio_ui_enabled=True, monitoring_enabled=True
    )
    with TestClient(create_app(settings=settings)) as client:
        assert client.get(DEMO).status_code == 200
        assert client.post("/api/v1/explain", json=record).status_code == 200
        monitoring = client.get("/api/v1/monitoring").json()

    assert monitoring["monitoring_enabled"] is True
    assert monitoring["details_suppressed"] is True


# --------------------------------------------------------------------------- #
# Settings.
# --------------------------------------------------------------------------- #


def test_the_ui_flag_is_read_from_the_environment(repo_root: Path) -> None:
    from churn.serving.settings import load_settings

    assert load_settings(env={}, root=repo_root).portfolio_ui_enabled is False
    assert (
        load_settings(env={"CHURN_SERVING_PORTFOLIO_UI": "1"}, root=repo_root).portfolio_ui_enabled
        is True
    )


# --------------------------------------------------------------------------- #
# DOM safety.
#
# A categorical may carry a "custom / unseen value", which is arbitrary text the
# visitor typed. It returns in value_display and can appear in an error message, so
# every dynamic value is data and the page must insert it as text.
#
# The strategy is not "escape carefully at each call site" but "construct no HTML
# string at all": if the file never hands a string to an HTML parser, there is no
# sink to get wrong, and no sanitiser to add. These tests enforce that as a property
# of the whole file.
# --------------------------------------------------------------------------- #

#: Sinks that parse a string as HTML or as code. Banned from the program.
#:
#: Checked over the code with comments stripped, like every other audit in this file:
#: the script's header names these very sinks in the sentence promising it does not
#: use them, and a raw search would flag that promise as the violation.
DANGEROUS_SINKS = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "eval(",
    "new Function",
    "createContextualFragment",
    "srcdoc",
)


def test_the_script_uses_no_dangerous_dom_sink(script: str) -> None:
    """Not one of them appears in the program.

    The only occurrences in the file are in the header comment that promises their
    absence, which is why the program is read with the prose stripped out.
    """
    code = _code_only(script)

    for sink in DANGEROUS_SINKS:
        assert sink not in code, sink

    # Substring bans cover the property-access and bracket-access forms alike, so
    # what is left to check is the attribute route: nothing sets an event handler
    # or a URL-bearing attribute, which is how a value becomes code without a sink.
    for attribute in re.findall(r"setAttribute\(\s*.([^\"']+).", code):
        assert not attribute.startswith("on"), attribute
        assert attribute not in {"href", "src", "srcdoc", "formaction"}, attribute


def test_the_page_uses_no_dangerous_dom_sink_and_no_inline_handler(html: str) -> None:
    """Static markup only: no inline script, and no on*= attribute anywhere."""
    page = _code_only(html)

    for sink in DANGEROUS_SINKS:
        assert sink not in page, sink
    assert not re.search(r"<\s*script(?![^>]*\ssrc=)", page)
    assert not re.search(r"\son[a-z]+\s*=", page)
    assert "javascript:" not in page


def test_dynamic_content_is_written_through_text_only(script: str) -> None:
    """The permitted primitives are present; the forbidden ones are not.

    ``el()`` is the single node factory and it assigns textContent, so a node built
    anywhere in this file carries its value as text by construction.
    """
    code = _code_only(script)

    assert "node.textContent = String(text)" in _flat(code)
    for primitive in ("createElement", "textContent", "replaceChildren", "appendChild"):
        assert primitive in code, primitive


def test_containers_are_emptied_with_replace_children(script: str) -> None:
    """One clearing primitive, and it is not an HTML assignment."""
    code = _code_only(script)

    assert "node.replaceChildren()" in _flat(code)
    assert 'innerHTML = ""' not in code
    assert "innerHTML = ''" not in code


def test_no_html_string_is_built_from_data(script: str) -> None:
    """No template that interpolates a value into markup, anywhere."""
    code = _code_only(script)

    # A backtick template containing a tag, or a concatenation that opens one.
    assert not re.search(r"`[^`]*<[a-zA-Z][^`]*\$\{", code)
    assert not re.search(r'"\s*<[a-zA-Z][^"]*"\s*\+', code)
    assert "document.createRange" not in code
    assert "DOMParser" not in code


def test_no_sanitiser_or_third_party_escaping_library_was_added(script: str, html: str) -> None:
    """A value that is never parsed as HTML has nothing to sanitise."""
    for library in ("DOMPurify", "sanitize-html", "xss(", "escapeHtml"):
        assert library not in script, library
        assert library not in html, library


def test_untrusted_response_fields_reach_the_dom_only_through_the_text_factory(
    script: str,
) -> None:
    """value_display and active_level are the fields a visitor controls.

    Wherever they appear, they appear as an argument to ``el()`` — whose third
    parameter becomes textContent — or as a comparison, never next to a sink.
    """
    code = _code_only(script)

    for line in code.splitlines():
        if "value_display" in line or "active_level" in line:
            assert "el(" in line or "//" in line, line
            for sink in DANGEROUS_SINKS:
                assert sink not in line, line


# --------------------------------------------------------------------------- #
# One submission is one scoring call.
#
# /api/v1/explain already returns churn_probability, prediction, decision, threshold,
# comparison and calibration_policy, so there is no mathematical reason to score the
# same form twice. A page that also posted to /api/v1/predict would double the
# latency, double the monitoring records, and open a gap for the two answers to
# diverge one day.
# --------------------------------------------------------------------------- #


def test_the_page_never_calls_the_prediction_endpoint(script: str) -> None:
    """/api/v1/predict is untouched and still there — the page simply is not its client."""
    code = _code_only(script)

    assert "/api/v1/predict" not in code
    assert "predict/batch" not in code


def test_one_submission_makes_exactly_one_scoring_call(script: str) -> None:
    """The handler posts once, and it posts to the analysis endpoint."""
    code = _code_only(script)
    body = _function_body(code, "submit")
    posts = re.findall(r"postJSON\(\s*([A-Za-z0-9_.]+)", body)

    assert posts == ["API.explain"]
    assert body.count("postJSON(") == 1
    assert body.count("fetch(") == 0


def test_the_only_post_helper_is_used_by_the_only_scoring_call(script: str) -> None:
    """No second POST path elsewhere in the file: one writer, one call site."""
    code = _code_only(script)

    assert code.count("postJSON(") == 2  # the definition and the single call
    assert code.count('method: "POST"') == 1


def test_the_analysis_endpoint_is_the_explanation_endpoint(script: str) -> None:
    code = _code_only(script)

    assert re.search(r"explain:\s*\"/api/v1/explain\"", code)
    assert "postJSON(API.explain, payload)" in _flat(code)


def test_the_post_submission_refresh_is_read_only(script: str) -> None:
    """The window counter is re-read after an analysis; re-reading scores nothing."""
    code = _code_only(script)
    body = _function_body(code, "refreshMonitoring")

    assert "getJSON(API.monitoring)" in _flat(body)
    assert "postJSON(" not in body


def test_every_other_call_the_page_makes_is_a_get(script: str) -> None:
    """Model card, portfolio metadata, readiness and monitoring: all reads."""
    code = _code_only(script)
    reads = set(re.findall(r"getJSON\(\s*API\.([a-z]+)", code))

    assert reads == {"ready", "model", "portfolio", "monitoring"}


def test_the_page_renders_the_prediction_from_the_explanation_response(script: str) -> None:
    """No second source: the same response feeds the score card and the factors."""
    code = _code_only(script)
    body = _function_body(code, "submit")

    assert "renderResult(result.body)" in _flat(body)
    assert "renderFactors(result.body)" in _flat(body)
