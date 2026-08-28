/* Churn Risk Console — Phase 13.
 *
 * This file presents. It does not decide.
 *
 * Nothing here recomputes a probability, applies a threshold, or hardcodes a
 * model constant. The threshold, the comparison, the calibration policy, the
 * fingerprint and every metric arrive from the API — either from the live model
 * metadata or from versioned portfolio metadata built offline. If the backend
 * changed a value tomorrow, this page would show the new one without an edit.
 *
 * The one thing it *does* compute is the pixel position of a marker on an axis,
 * which is a layout concern.
 *
 * No storage of any kind: no localStorage, no sessionStorage, no IndexedDB, no
 * cookies. Submitted values live in the form and in one in-flight request.
 *
 * SECURITY INVARIANT — every dynamic value is inserted as TEXT.
 *
 * A categorical feature may carry a "custom / unseen value", which is arbitrary
 * text the visitor typed. It comes back in the explanation as value_display and
 * active_level, and it can appear in an error message. It is data, never markup.
 * So this file constructs no HTML string at all: nodes are made with
 * createElement, content is set through textContent or value, containers are
 * emptied with replaceChildren() and filled with appendChild. innerHTML,
 * outerHTML, insertAdjacentHTML, document.write, eval and new Function do not
 * appear, and the absence is enforced by a test rather than by this comment.
 *
 * That is why there is no sanitiser here: a value that is never parsed as HTML
 * has nothing to sanitise, and a sanitiser would be a dependency guarding a sink
 * that does not exist.
 */

(function () {
  "use strict";

  var API = {
    model: "/api/v1/model",
    explain: "/api/v1/explain",
    portfolio: "/api/v1/portfolio",
    monitoring: "/api/v1/monitoring",
    ready: "/health/ready"
  };

  /* Which fieldset each categorical feature is rendered into, and in what order.
   * Grouping is presentational; the payload always carries all 19 features. */
  var LAYOUT = [
    ["group-demographics", ["gender", "SeniorCitizen", "Partner", "Dependents"]],
    ["group-phone", ["PhoneService", "MultipleLines"]],
    ["group-internet", ["InternetService", "OnlineSecurity", "OnlineBackup",
      "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]],
    ["group-contract", ["Contract", "PaperlessBilling", "PaymentMethod"]]
  ];

  var NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"];

  /* Structural coupling, used ONLY to prefill the form. The API is unchanged:
   * any of these can be overridden, and the request is still accepted. */
  var INTERNET_DEPENDENTS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"];
  var NO_INTERNET = "No internet service";
  var NO_PHONE = "No phone service";

  var CUSTOM = "__custom__";

  /* Two invented customers. Neither is a row from any dataset, and neither is
   * labelled by outcome before the model has been asked. */
  var EXAMPLES = {
    a: {
      tenure: 2, MonthlyCharges: 84.5, TotalCharges: "169.00",
      gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No",
      PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic",
      OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No",
      TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes",
      Contract: "Month-to-month", PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check"
    },
    b: {
      tenure: 58, MonthlyCharges: 24.9, TotalCharges: "1444.20",
      gender: "Male", SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes",
      PhoneService: "Yes", MultipleLines: "No", InternetService: "No",
      OnlineSecurity: NO_INTERNET, OnlineBackup: NO_INTERNET,
      DeviceProtection: NO_INTERNET, TechSupport: NO_INTERNET,
      StreamingTV: NO_INTERNET, StreamingMovies: NO_INTERNET,
      Contract: "Two year", PaperlessBilling: "No",
      PaymentMethod: "Bank transfer (automatic)"
    }
  };

  var metadata = null;
  var inFlight = false;

  /* ------------------------------------------------------------------ dom */

  function $(id) { return document.getElementById(id); }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  /* The only way this file empties a container. replaceChildren() with no argument
   * is used rather than innerHTML = "" so that the security invariant below is a
   * property of the whole file and not of each call site: no HTML string is
   * constructed anywhere, so none can be constructed from untrusted data. */
  function clear(node) { node.replaceChildren(); }

  function setPill(id, text, kind) {
    var node = $(id);
    node.textContent = text;
    node.className = "pill pill--" + kind;
  }

  /* --------------------------------------------------------------- format */

  function fmtProbability(value) { return (value * 100).toFixed(1) + "%"; }
  function fmtFixed(value, digits) { return Number(value).toFixed(digits); }
  function fmtSigned(value) {
    var text = Math.abs(value).toFixed(4);
    return (value > 0 ? "+" : value < 0 ? "−" : "±") + text;
  }

  /* ----------------------------------------------------------------- form */

  function buildCategoricalField(feature, levels) {
    var field = el("div", "field");
    var selectId = "f-" + feature;

    var label = el("label", null, feature);
    label.setAttribute("for", selectId);

    var select = el("select");
    select.id = selectId;
    select.name = feature;
    select.setAttribute("data-feature", feature);

    var blank = el("option", null, "— select —");
    blank.value = "";
    select.appendChild(blank);

    levels.forEach(function (level) {
      var option = el("option", null, level);
      option.value = level;
      select.appendChild(option);
    });

    var custom = el("option", null, "Custom / unseen value…");
    custom.value = CUSTOM;
    select.appendChild(custom);

    var text = el("input", "field__custom");
    text.type = "text";
    text.id = selectId + "-custom";
    text.name = feature + "__custom";
    text.hidden = true;
    text.placeholder = "Value not seen during training";
    text.setAttribute("aria-label", feature + " custom value");

    select.addEventListener("change", function () {
      var isCustom = select.value === CUSTOM;
      text.hidden = !isCustom;
      if (isCustom) { text.focus(); } else { text.value = ""; }
      if (feature === "InternetService" || feature === "PhoneService") { applyCoupling(feature); }
    });

    field.appendChild(label);
    field.appendChild(select);
    field.appendChild(text);
    return field;
  }

  function renderForm(levels) {
    LAYOUT.forEach(function (entry) {
      var container = $(entry[0]);
      clear(container);
      entry[1].forEach(function (feature) {
        var known = levels[feature] || [];
        container.appendChild(buildCategoricalField(feature, known));
      });
    });
  }

  /* UI assistance only. The backend contract is untouched: these values are
   * ordinary categories, and a user who overrides one still gets a scored
   * prediction — monitoring simply counts it as a structural inconsistency. */
  function applyCoupling(trigger) {
    if (trigger === "InternetService") {
      var noInternet = valueOf("InternetService") === "No";
      INTERNET_DEPENDENTS.forEach(function (feature) {
        var select = $("f-" + feature);
        if (!select) { return; }
        if (noInternet) {
          if (hasOption(select, NO_INTERNET)) { select.value = NO_INTERNET; }
        } else if (select.value === NO_INTERNET) {
          select.value = "";
        }
      });
      var hint = $("internet-hint");
      if (hint) { hint.classList.toggle("is-active", noInternet); }
    }
    if (trigger === "PhoneService") {
      var lines = $("f-MultipleLines");
      if (!lines) { return; }
      if (valueOf("PhoneService") === "No") {
        if (hasOption(lines, NO_PHONE)) { lines.value = NO_PHONE; }
      } else if (lines.value === NO_PHONE) {
        lines.value = "";
      }
    }
  }

  function hasOption(select, value) {
    return Array.prototype.some.call(select.options, function (option) {
      return option.value === value;
    });
  }

  function valueOf(feature) {
    var select = $("f-" + feature);
    if (!select) { return ""; }
    if (select.value === CUSTOM) { return ($(select.id + "-custom") || {}).value || ""; }
    return select.value;
  }

  function fillExample(record) {
    NUMERIC.forEach(function (feature) { $(feature).value = record[feature]; });
    Object.keys(record).forEach(function (feature) {
      var select = $("f-" + feature);
      if (!select) { return; }
      var value = String(record[feature]);
      var custom = $(select.id + "-custom");
      if (hasOption(select, value)) {
        select.value = value;
        if (custom) { custom.hidden = true; custom.value = ""; }
      } else if (custom) {
        select.value = CUSTOM;
        custom.hidden = false;
        custom.value = value;
      }
    });
    applyCoupling("InternetService");
    applyCoupling("PhoneService");
    clearFieldErrors();
  }

  function clearForm() {
    $("customer-form").reset();
    document.querySelectorAll(".field__custom").forEach(function (node) {
      node.hidden = true;
      node.value = "";
    });
    clearFieldErrors();
    hideAlert();
    $("result-panel").hidden = true;
    $("factors-panel").hidden = true;
  }

  /* Assembles the payload. TotalCharges is sent as typed, including blank: the
   * structural-zero rule lives in the frozen preprocessing and is not
   * reimplemented here. */
  function collect() {
    var payload = {
      tenure: numberOrNull($("tenure").value),
      MonthlyCharges: numberOrNull($("MonthlyCharges").value),
      TotalCharges: $("TotalCharges").value.trim()
    };
    LAYOUT.forEach(function (entry) {
      entry[1].forEach(function (feature) { payload[feature] = valueOf(feature); });
    });
    payload.SeniorCitizen = numberOrNull(payload.SeniorCitizen);
    return payload;
  }

  function numberOrNull(value) {
    if (value === null || value === undefined || String(value).trim() === "") { return null; }
    var parsed = Number(value);
    return isNaN(parsed) ? null : parsed;
  }

  function missingFields(payload) {
    var missing = [];
    if (payload.tenure === null) { missing.push("tenure"); }
    if (payload.MonthlyCharges === null) { missing.push("MonthlyCharges"); }
    LAYOUT.forEach(function (entry) {
      entry[1].forEach(function (feature) {
        if (!payload[feature] && payload[feature] !== 0) { missing.push(feature); }
      });
    });
    return missing;
  }

  function markFields(fields) {
    clearFieldErrors();
    fields.forEach(function (name) {
      var node = $(name) || $("f-" + name);
      if (node) { node.setAttribute("aria-invalid", "true"); }
    });
  }

  function clearFieldErrors() {
    document.querySelectorAll("[aria-invalid]").forEach(function (node) {
      node.removeAttribute("aria-invalid");
    });
  }

  /* ---------------------------------------------------------------- alerts */

  function showAlert(title, detail, fields) {
    var box = $("alert");
    clear(box);
    box.appendChild(el("strong", null, title));
    if (detail) { box.appendChild(el("span", null, detail)); }
    if (fields && fields.length) {
      var list = el("ul");
      fields.forEach(function (item) { list.appendChild(el("li", null, item)); });
      box.appendChild(list);
    }
    box.hidden = false;
  }

  function hideAlert() { $("alert").hidden = true; }

  /* Maps the API's stable error codes onto something a person can act on. The
   * message the backend sends is already sanitised; it is shown, never a
   * traceback and never a path. */
  function describeError(status, body) {
    var code = body && body.error && body.error.code;
    var message = (body && body.error && body.error.message) || "";
    var fields = [];
    if (body && body.error && Array.isArray(body.error.details)) {
      fields = body.error.details.map(function (item) {
        return (item.loc || []).filter(function (part) { return part !== "body"; }).join(".")
          + ": " + (item.msg || "");
      });
    }
    if (code === "INVALID_REQUEST_SCHEMA") {
      return ["Check the highlighted input fields.", "The request did not match the 19-feature contract.", fields];
    }
    if (code === "INVALID_FEATURE_VALUE" || code === "INVALID_FEATURE_PAYLOAD") {
      return ["This customer was rejected by the feature contract.", message, []];
    }
    if (code === "EXPLANATION_UNAVAILABLE") {
      return ["The explanation could not be verified and was withheld.", message, []];
    }
    if (code === "SERVICE_NOT_READY" || status === 503) {
      return ["The service is not ready.", message || "No verified model is loaded.", []];
    }
    return ["The request failed.", message || ("HTTP " + status), fields];
  }

  /* ----------------------------------------------------------------- fetch */

  function getJSON(url) {
    return fetch(url, { headers: { Accept: "application/json" } }).then(function (response) {
      return response.json().then(function (body) {
        return { ok: response.ok, status: response.status, body: body };
      });
    });
  }

  function postJSON(url, payload) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.json().then(function (body) {
        return { ok: response.ok, status: response.status, body: body };
      });
    });
  }

  /* ---------------------------------------------------------------- render */

  function renderResult(data) {
    var churn = data.prediction === 1;
    var panel = $("result-panel");
    panel.hidden = false;
    panel.classList.toggle("is-churn", churn);
    panel.classList.toggle("is-retained", !churn);

    var tag = $("result-decision-tag");
    tag.textContent = churn ? "At/above threshold" : "Below threshold";
    tag.className = "panel__tag " + (churn ? "panel__tag--churn" : "panel__tag--retained");

    $("score-display").textContent = fmtProbability(data.churn_probability);
    $("score-exact").textContent = String(data.churn_probability);
    $("decision-value").textContent = data.decision;
    $("threshold-value").textContent = fmtFixed(data.threshold, 4);
    $("rule-value").textContent = "score " + data.comparison + " " + fmtFixed(data.threshold, 4);
    $("calibration-value").textContent = data.calibration_policy;
    $("calibration-note").textContent = data.calibration_note;

    drawScale(data.churn_probability, data.threshold);
  }

  /* Layout arithmetic only: the decision was already made by the backend. */
  function drawScale(score, threshold) {
    var x = Math.max(0, Math.min(100, threshold * 100));
    var high = $("scale-high");
    high.setAttribute("x", String(x));
    high.setAttribute("width", String(100 - x));
    var line = $("scale-threshold");
    line.setAttribute("x1", String(x));
    line.setAttribute("x2", String(x));
    $("scale-marker").setAttribute("cx", String(Math.max(0, Math.min(100, score * 100))));
  }

  function factorNode(item) {
    var isGroup = Object.prototype.hasOwnProperty.call(item, "members");
    var amount = item.contribution_log_odds;
    var up = amount > 0;

    var node = el("li", "factor " + (up ? "factor--up" : "factor--down"));

    var top = el("div", "factor__top");
    var name = el("span", "factor__name", isGroup ? item.label : item.feature);
    if (isGroup) {
      name.appendChild(el("span", "factor__badge", item.n_members + " features"));
    } else if (item.is_unseen_level) {
      name.appendChild(el("span", "factor__badge factor__badge--unseen", "unseen level"));
    }
    top.appendChild(name);
    top.appendChild(el("span", "factor__amount", fmtSigned(amount)));
    node.appendChild(top);

    node.appendChild(el("p", "factor__value", item.value_display));

    var bar = el("div", "factor__bar");
    var fill = el("span");
    fill.style.width = item.__share + "%";
    bar.appendChild(fill);
    node.appendChild(bar);

    if (isGroup) {
      node.appendChild(el("p", "factor__group", item.members.join(", ")));
    }
    return node;
  }

  function renderFactors(data) {
    $("factors-panel").hidden = false;

    /* Coupled blocks replace their members; ungrouped features stand alone.
     * Together they partition the 19 features, so the displayed rows still sum
     * to the model logit. */
    var grouped = data.grouped_contributions || [];
    var byFeature = {};
    data.contributions.forEach(function (item) { byFeature[item.feature] = item; });
    var items = grouped.concat((data.ungrouped_features || []).map(function (name) {
      return byFeature[name];
    })).filter(Boolean);

    var largest = items.reduce(function (max, item) {
      return Math.max(max, Math.abs(item.contribution_log_odds));
    }, 0) || 1;
    items.forEach(function (item) {
      item.__share = Math.round((Math.abs(item.contribution_log_odds) / largest) * 100);
    });
    items.sort(function (left, right) {
      return Math.abs(right.contribution_log_odds) - Math.abs(left.contribution_log_odds);
    });

    var up = $("factors-up");
    var down = $("factors-down");
    clear(up);
    clear(down);

    items.forEach(function (item) {
      if (item.contribution_log_odds > 0) { up.appendChild(factorNode(item)); }
      else if (item.contribution_log_odds < 0) { down.appendChild(factorNode(item)); }
    });
    if (!up.childNodes.length) { up.appendChild(el("li", "factor-list__empty", "None.")); }
    if (!down.childNodes.length) { down.appendChild(el("li", "factor-list__empty", "None.")); }

    var sum = data.model_logit - data.intercept;
    $("maths-intercept").textContent = fmtFixed(data.intercept, 6);
    $("maths-sum").textContent = fmtSigned(sum);
    $("maths-logit").textContent = fmtFixed(data.model_logit, 6);
    $("maths-threshold-logit").textContent = fmtFixed(data.threshold_logit, 6);
    $("maths-margin").textContent = fmtSigned(data.margin_log_odds);
    $("maths-error").textContent = Math.max(
      data.reconstruction.logit_error, data.reconstruction.probability_error
    ).toExponential(2);
    $("causal-note").textContent = data.causal_note;
  }

  function renderModelCard(model) {
    $("mc-estimator").textContent = model.estimator;
    $("mc-freeze").textContent = model.freeze_commit_short;
    $("mc-fingerprint").textContent = model.model_fingerprint.slice(0, 16) + "…";
    $("mc-features").textContent = model.n_features;
    $("mc-transformed").textContent = model.n_transformed_features;
    $("mc-calibration").textContent = model.calibration_policy;
    $("mc-threshold-policy").textContent = model.threshold_policy;
    $("mc-threshold").textContent = String(model.threshold);
    $("mc-positive").textContent = model.positive_class_meaning;
    $("mc-serving").textContent = model.serving_version;
  }

  function renderEvaluation(evaluation) {
    $("perf-n").textContent = evaluation.n_samples + " held-out customers";

    var list = $("metrics-headline");
    clear(list);
    evaluation.headline.forEach(function (metric) {
      var item = el("li", "metric");
      item.appendChild(el("p", "metric__label", metric.label));
      item.appendChild(el("p", "metric__value", fmtFixed(metric.value, 3)));
      if (metric.ci_lower !== null && metric.ci_lower !== undefined) {
        item.appendChild(el("p", "metric__ci",
          "95% CI " + fmtFixed(metric.ci_lower, 3) + "–" + fmtFixed(metric.ci_upper, 3)));
      }
      list.appendChild(item);
    });

    var aux = $("metrics-auxiliary");
    clear(aux);
    evaluation.auxiliary.forEach(function (metric) {
      var row = el("div");
      row.appendChild(el("dt", null, metric.label));
      row.appendChild(el("dd", null, fmtFixed(metric.value, 3)));
      aux.appendChild(row);
    });
    var baseline = el("div");
    baseline.appendChild(el("dt", null, "No-skill AP"));
    baseline.appendChild(el("dd", null, fmtFixed(evaluation.no_skill_average_precision, 3)));
    aux.appendChild(baseline);

    $("metric-reading-note").textContent = evaluation.metric_reading_note;
    $("analyst-caveat").textContent = evaluation.caveat;
  }

  function renderMonitoring(body) {
    var container = $("monitoring-body");
    var tag = $("monitoring-tag");
    clear(container);

    if (!body || body.monitoring_enabled === false) {
      tag.textContent = "disabled";
      container.appendChild(el("p", "note", "Monitoring disabled for this deployment."));
      setPill("status-monitoring", "disabled", "off");
      return;
    }

    tag.textContent = body.status;
    setPill("status-monitoring", (body.collector_health || "unknown").toLowerCase(),
      body.collector_health === "HEALTHY" ? "ok" : "warn");

    var rows = [
      ["Monitoring status", body.status],
      ["Window records", String(body.n_records)],
      ["Minimum window size", String(body.minimum_window_size)]
    ];
    if (body.details_suppressed) {
      container.appendChild(el("p", "note",
        "This window is below the operational minimum, so the per-feature "
        + "distributions are suppressed by the monitoring privacy policy. Only the "
        + "window size and the global operational counters are reported."));
    } else if (body.section_status) {
      rows.push(["Data drift", body.section_status.feature_drift]);
      rows.push(["Prediction drift", body.section_status.prediction_drift]);
      rows.push(["Structural consistency", body.section_status.structural_consistency]);
    }

    var list = el("ul", "mono-list");
    rows.forEach(function (row) {
      var item = el("li");
      item.appendChild(el("span", null, row[0]));
      item.appendChild(el("span", null, row[1]));
      list.appendChild(item);
    });
    container.appendChild(list);
  }

  /* ---------------------------------------------------------------- submit */

  function submit(event) {
    event.preventDefault();
    if (inFlight) { return; }

    var payload = collect();
    var missing = missingFields(payload);
    if (missing.length) {
      markFields(missing);
      showAlert("Check the highlighted input fields.",
        "These features are required by the contract:", missing);
      return;
    }

    clearFieldErrors();
    hideAlert();
    setBusy(true);

    postJSON(API.explain, payload).then(function (result) {
      if (!result.ok) {
        var described = describeError(result.status, result.body);
        showAlert(described[0], described[1], described[2]);
        $("result-panel").hidden = true;
        $("factors-panel").hidden = true;
        return;
      }
      renderResult(result.body);
      renderFactors(result.body);
      refreshMonitoring();
      $("result-panel").scrollIntoView({ block: "nearest" });
    }).catch(function () {
      showAlert("The request could not be sent.", "The API did not respond.", []);
    }).finally(function () {
      setBusy(false);
    });
  }

  function setBusy(busy) {
    inFlight = busy;
    var button = $("submit-btn");
    button.disabled = busy;
    button.classList.toggle("is-busy", busy);
    button.querySelector(".btn__label").textContent = busy ? "Scoring…" : "Score customer";
  }

  /* ------------------------------------------------------------------ boot */

  function refreshMonitoring() {
    getJSON(API.monitoring).then(function (result) {
      renderMonitoring(result.ok ? result.body : null);
    }).catch(function () { renderMonitoring(null); });
  }

  function boot() {
    getJSON(API.ready).then(function (result) {
      var ready = result.ok && result.body.status === "ready";
      setPill("status-api", ready ? "ready" : "not ready", ready ? "ok" : "bad");
      setPill("status-artifact", ready ? "verified" : "unverified", ready ? "ok" : "bad");
    }).catch(function () {
      setPill("status-api", "unreachable", "bad");
      setPill("status-artifact", "unknown", "bad");
    });

    getJSON(API.model).then(function (result) {
      if (result.ok) { renderModelCard(result.body); }
    });

    getJSON(API.portfolio).then(function (result) {
      if (!result.ok) { return; }
      metadata = result.body;
      renderForm(metadata.known_levels);
      renderEvaluation(metadata.evaluation);
    });

    refreshMonitoring();

    $("customer-form").addEventListener("submit", submit);
    $("load-a").addEventListener("click", function () { fillExample(EXAMPLES.a); });
    $("load-b").addEventListener("click", function () { fillExample(EXAMPLES.b); });
    $("clear-form").addEventListener("click", clearForm);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
