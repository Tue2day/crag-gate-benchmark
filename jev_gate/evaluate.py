import csv
import os
from pathlib import Path
import random
import numpy as np

from .io import digest, metadata, read_json, read_jsonl, write_json
from .jev import cache_key, request_body, ENDPOINT as JEV_ENDPOINT
from .gpt import request_body as gpt_request_body, ENDPOINT as GPT_ENDPOINT
from .metrics import (accepted_mask, bad_average_precision, choose_threshold, gate_metrics,
                      paired_coverage_interval, reliability, risk_coverage_curve)

METHODS = ("all_pass", "ce_threshold", "logistic", "jev", "gpt")
API_METHODS = ("jev", "gpt")


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile_summary(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "p50_ms": float(np.percentile(values, 50)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None}


def evaluate(cfg, mock=False):
    root = cfg.root(mock)
    groups = read_jsonl(root / "ranked.jsonl")
    baseline_rows = read_jsonl(root / "baselines.jsonl")
    jev_rows = read_jsonl(root / "jev.jsonl")
    gpt_rows = read_jsonl(root / "gpt.jsonl")
    for rows in (groups, baseline_rows, jev_rows, gpt_rows):
        if any(g.get("mock") != mock for g in rows):
            raise ValueError("Mixed mock/real input rejected")
        ids = [g["query_id"] for g in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate query IDs in prediction artifacts")
    fingerprint = digest(groups)
    baseline_meta = read_json(root / "baselines_metadata.json")
    unavailable = {}
    if baseline_meta.get("logistic_status") == "unavailable":
        unavailable["logistic"] = baseline_meta["logistic_unavailable_reason"]
    methods = tuple(m for m in METHODS if m not in unavailable)
    for stage in ("baselines", "jev", "gpt"):
        meta = read_json(root / f"{stage}_metadata.json")
        if meta["ranked_fingerprint"] != fingerprint:
            raise ValueError(f"Stale {stage} outputs: rerun that stage after reranking")
    baseline = {r["query_id"]: r for r in baseline_rows}
    api = {"jev": {r["query_id"]: r for r in jev_rows},
           "gpt": {r["query_id"]: r for r in gpt_rows}}
    identities = {}
    for method in API_METHODS:
        resolved = sorted({r["model"] for r in api[method].values() if r["status"] == "success"})
        if len(resolved) > 1:
            raise ValueError(f"Multiple resolved {method} builds in predictions; use one model build per protocol")
        identities[method] = {"requested": cfg.jev_model if method == "jev" else cfg.gpt_model,
                              "resolved": resolved}
    for group in groups:
        if group["query_id"] not in baseline:
            raise ValueError("Baseline outputs incomplete")
        for method, builder, endpoint in (("jev", request_body, JEV_ENDPOINT),
                                           ("gpt", gpt_request_body, GPT_ENDPOINT)):
            record = api[method].get(group["query_id"])
            if record and record["cache_key"] != cache_key(builder(group, cfg), mock, endpoint):
                raise ValueError(f"{method} predictions do not match current model/prompt/input; rerun {method}")
    common = [g for g in groups if all(api[m].get(g["query_id"], {}).get("status") == "success" for m in API_METHODS)]

    def scores(rows, method):
        return np.asarray([api[method][g["query_id"]]["probability"] if method in API_METHODS
                           else baseline[g["query_id"]][method] for g in rows], dtype=float)

    validation = [g for g in common if g["split"] == "validation"]
    all_validation = [g for g in groups if g["split"] == "validation"]
    val_y = np.asarray([g["y"] for g in validation], dtype=int)
    op_input = {"validation": [{"id": g["query_id"], "y": g["y"],
                                **{m: float(scores([g], m)[0]) for m in methods}} for g in validation],
                "target_risk": cfg.target_risk, "min_accepted": cfg.min_accepted,
                "api_model_identities": identities, "unavailable_methods": unavailable,
                "protocol": "openrouter-five-way-v2"}
    op_hash = digest(op_input)
    op_path = root / "operating_points.json"
    if op_path.exists():
        frozen = read_json(op_path)
        if frozen["validation_hash"] != op_hash:
            raise ValueError("Frozen validation operating points changed. Use a new run_dir for a new protocol; do not retune on test results")
        operating = frozen["methods"]
    else:
        operating = {m: choose_threshold(val_y, scores(validation, m), cfg.target_risk,
                                         cfg.min_accepted) for m in methods if m != "all_pass"}
        operating["all_pass"] = {"available": True, "threshold": 0.0, "accepted": len(val_y),
                                  "risk": float((1-val_y).mean()) if len(val_y) else None}
        for m in unavailable:
            operating[m] = {"available": False, "threshold": None, "accepted": None, "risk": None}
        write_json(op_path, {"mock": mock, "validation_hash": op_hash,
                            "common_validation_n": len(validation), "methods": operating})

    test = [g for g in common if g["split"] == "test"]
    all_test = [g for g in groups if g["split"] == "test"]
    challenge = [g for g in common if g["split"] == "challenge"]
    y = np.asarray([g["y"] for g in test], dtype=int)
    bad_n = sum(g["y"] == 0 for g in all_test)
    warnings = []
    warnings.extend(f"{m} unavailable: {reason}" for m, reason in unavailable.items())
    if bad_n < 50:
        warnings.append(f"Natural test has only {bad_n} no-Exact queries (<50); limited discrimination and precision")
    if all_test and bad_n/len(all_test) <= cfg.target_risk:
        warnings.append("All-pass already meets the target risk on the natural test; coverage uplift is not a useful success claim")
    if len(set(val_y.tolist())) < 2:
        warnings.append("Common validation set has fewer than two classes")
    if len(validation) < len(all_validation):
        warnings.append("API failures reduced the common validation set used for threshold selection")
    if len(test) < len(all_test):
        warnings.append("Model comparisons use common successful queries; API failures are reported separately")
    if len(test) < 100:
        warnings.append("Fewer than 100 common natural test queries; intervals are exploratory")
    masks = {m: accepted_mask(scores(test, m), operating[m]) for m in methods}
    summary = {}
    errors = []
    for m in methods:
        values = scores(test, m)
        metrics = gate_metrics(y, masks[m])
        metrics.update({"status": "evaluated", "operating_point_available": operating[m]["available"],
                        "threshold": operating[m]["threshold"],
                        "bad_average_precision": bad_average_precision(y, values),
                        "brier": float(np.mean((values-y)**2)) if len(y) and m in {"jev", "logistic", "gpt"} else None,
                        "probability_source": {"jev": "jev_native_noul", "gpt": "gpt_elicited_json",
                                               "logistic": "supervised_logistic"}.get(m),
                        "reliability": reliability(y, values) if m in {"jev", "logistic", "gpt"} else None})
        challenge_mask = accepted_mask(scores(challenge, m), operating[m])
        metrics["challenge_n"] = len(challenge)
        metrics["challenge_intercept_rate"] = float((~challenge_mask).mean()) if len(challenge) and operating[m]["available"] else None
        if m in API_METHODS:
            uncached = [api[m][g["query_id"]] for g in all_test
                        if g["query_id"] in api[m] and not api[m][g["query_id"]]["cache_hit"]]
            metrics["latency"] = percentile_summary([r["total_elapsed_ms"] for r in uncached])
            metrics["single_attempt_latency"] = percentile_summary(
                [a["elapsed_ms"] for r in uncached for a in r["attempts"]])
        else:
            metrics["latency"] = percentile_summary([baseline[g["query_id"]]["latency_ms"][m] for g in all_test])
        metrics["coverage_differences"] = {}
        for reference in ("ce_threshold", "logistic", "jev", "gpt"):
            if reference != m and operating[m]["available"] and operating[reference]["available"]:
                delta = float((masks[m].astype(float)-masks[reference].astype(float)).mean()) if len(y) else None
                metrics["coverage_differences"][reference] = {
                    "delta": delta, "paired_bootstrap_95": paired_coverage_interval(
                        masks[m], masks[reference], cfg.bootstrap_samples, cfg.seed)}
        summary[m] = metrics
        for group, value, accepted in zip(test, values, masks[m]):
            if int(accepted) != group["y"]:
                errors.append({"method": m, "query_id": group["query_id"], "query": group["query"],
                    "y": group["y"], "score": float(value), "accepted": bool(accepted),
                    "kind": "bad_pass" if accepted else "unnecessary_fallback",
                    "titles": " | ".join(c["title"] for c in group["candidates"]),
                    "labels": " ".join(c["label"] for c in group["candidates"])})
    failed = {m: [g for g in all_test if api[m].get(g["query_id"], {}).get("status") != "success"]
              for m in API_METHODS}
    for m in API_METHODS:
        for g in failed[m]:
            errors.append({"method": m, "query_id": g["query_id"], "query": g["query"],
                           "y": g["y"], "kind": "api_failure_or_missing", "accepted": False})
    total_test = len(all_test)
    operational = {}
    for m in methods:
        decisions = []
        for g in all_test:
            if m in API_METHODS and api[m].get(g["query_id"], {}).get("status") != "success":
                decisions.append(False)
            else:
                decisions.append(bool(accepted_mask(scores([g], m), operating[m])[0]))
        operational[m] = {**gate_metrics([g["y"] for g in all_test], decisions),
                          "note": "Missing/failed API requests dispatch to fallback; not counted as correct model predictions"}
    # Describe cost of unique successful predictions separately from the calls made this invocation.
    def cost_stats(records, method):
        input_price = cfg.input_price_per_million if method == "jev" else cfg.gpt_input_price_per_million
        output_price = 0.0 if method == "jev" else cfg.gpt_output_price_per_million
        known, reported, estimates = [], [], []
        for r in records:
            usage = r["usage"]
            a, b = usage.get("input_tokens"), usage.get("output_tokens")
            if a is not None:
                known.append(a)
                if output_price == 0 or b is not None:
                    estimates.append((a*input_price + (b or 0)*output_price)/1e6)
            cost = usage.get("reported_cost_usd")
            if cost is not None:
                reported.append(cost)
        return {"requests": len(records), "known_usage_requests": len(known),
                "input_tokens": sum(known) if known else None,
                "known_token_estimated_cost_usd": sum(estimates) if estimates else None,
                "reported_cost_requests": len(reported),
                "known_reported_cost_usd": sum(reported) if reported else None,
                "estimated_usd_per_1000_successful_queries": float(np.mean(estimates))*1000
                    if estimates and len(estimates) == len(records) else None,
                "reported_usd_per_1000_successful_queries": float(np.mean(reported))*1000
                    if reported and len(reported) == len(records) else None,
                "potentially_unaccounted_failed_attempts": any(r.get("cost_may_exclude_failed_attempts") for r in records)}
    costs = {}
    for m in API_METHODS:
        records = [api[m][g["query_id"]] for g in all_test if g["query_id"] in api[m]]
        costs[m] = {"unique_natural_test_predictions": cost_stats([r for r in records if r["status"] == "success"], m),
                    "latest_invocation_natural_test": cost_stats([r for r in records if not r["cache_hit"]], m)}
    for m, stats in summary.items():
        stats["estimated_usd_per_1000_successful_queries"] = (
            costs[m]["unique_natural_test_predictions"]["estimated_usd_per_1000_successful_queries"]
            if m in API_METHODS else 0.0)
    for m, reason in unavailable.items():
        summary[m] = {"status": "unavailable", "reason": reason, "n": 0, "accepted": None,
            "bad_accepted": None, "risk": None, "risk_wilson_95": None, "coverage": None,
            "operating_point_available": False, "threshold": None, "bad_average_precision": None,
            "brier": None, "reliability": None, "probability_source": None,
            "challenge_n": 0, "challenge_intercept_rate": None, "latency": percentile_summary([]),
            "coverage_differences": {}, "estimated_usd_per_1000_successful_queries": None}
        operational[m] = {"status": "unavailable", "reason": reason, "n": 0}
    summary = {m: summary[m] for m in METHODS}
    report = {"mock": mock, "label": "SIMULATED — NOT BENCHMARK RESULTS" if mock else "REAL EXPERIMENT",
        "warnings": warnings, "methods": summary, "operational_all_test": operational,
        "completeness": {"natural_test_n": total_test, "natural_test_no_exact_n": bad_n,
            "common_success_test_n": len(test), "common_success_validation_n": len(validation),
            "validation_n": len(all_validation),
            **{f"{m}_failed_or_missing_test_n": len(failed[m]) for m in API_METHODS},
            **{f"{m}_failure_or_missing_test_rate": len(failed[m])/total_test if total_test else None for m in API_METHODS},
            "common_test_fraction": len(test)/total_test if total_test else None},
        "cost": {"by_model": costs,
                 "price_usd_per_million_tokens": {"jev": {"input": cfg.input_price_per_million, "output": 0},
                    "gpt": {"input": cfg.gpt_input_price_per_million, "output": cfg.gpt_output_price_per_million}},
                 "note": "Usage-based estimate, not an invoice; failed attempts may be billed without usage"},
        "model_identities": identities,
        "unavailable_methods": unavailable,
        "development_class_counts": baseline_meta.get("development_class_counts"),
        "calibration_note": "Jev native Noul vs GPT elicited JSON probability; neither is assumed calibrated. Differences do not isolate training method effects.",
        "bootstrap_note": "Paired query bootstrap, frozen thresholds; excludes uncertainty from fitting and threshold selection",
        "latency_note": "Gate only; latest invocation excludes cache hits. Both APIs include OpenRouter/network overhead, rate limiter and retries, but exclude pending queue wait; baseline lookup has zero modeled overhead",
        "ranked_fingerprint": fingerprint, "validation_hash": op_hash}
    out = root / "report"
    write_json(out / "metrics.json", report)
    table = [{"mode": report["label"], "method": m, **{k: v for k, v in s.items() if not isinstance(v, (dict, list))},
              "p50_ms": s["latency"]["p50_ms"], "p95_ms": s["latency"]["p95_ms"]}
             for m, s in summary.items()]
    write_csv(out / "summary.csv", table, ["mode", "method", "status", "reason", "n", "accepted", "bad_accepted", "risk",
        "coverage", "operating_point_available", "threshold", "bad_average_precision", "brier",
        "challenge_n", "challenge_intercept_rate", "probability_source", "estimated_usd_per_1000_successful_queries", "p50_ms", "p95_ms"])
    write_csv(out / "errors.csv", errors, ["method", "query_id", "query", "y", "score", "accepted", "kind", "titles", "labels"])
    audit = random.Random(cfg.seed).sample(all_test, min(100, len(all_test)))
    audit_rows = [{"query_id": g["query_id"], "query": g["query"], "product_id": c["id"],
                   "title": c["title"], "brand": c.get("brand"), "color": c.get("color"),
                   "bullet_points": c.get("bullet_points"), "original_label": c["label"],
                   "reviewer_label": "", "reviewer_notes": ""} for g in audit for c in g["candidates"]]
    write_csv(out / "manual_audit.csv", audit_rows, ["query_id", "query", "product_id", "title", "brand", "color",
                                                 "bullet_points", "original_label", "reviewer_label", "reviewer_notes"])
    plot_results(out, report["label"], y, {m: scores(test, m) for m in methods})
    lines = [f"# {report['label']}", "", "All probabilities and rates in metrics.json/summary.csv are fractions, not percentages.", "",
             report["bootstrap_note"], "", report["latency_note"], "", report["calibration_note"], "",
             "Risk target is an empirical validation constraint, not a test-set or production guarantee.", "",
             "Unavailable operating point dispatches all requests to fallback; its coverage is not a useful success result.", "",
             "Constructed accessory challenge results must never be pooled into natural-test metrics.", "",
             "API failure is an operational fallback, not a correct semantic prediction.", "",
             "## Warnings", "", *[f"- {w}" for w in warnings]]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", "utf-8")
    metadata(cfg, mock, "evaluate", {"ranked_fingerprint": fingerprint, "validation_hash": op_hash})
    print(f"{report['label']}: report saved to {out}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return report


def plot_results(out, label, y, predictions):
    cache = out.parent / ".plot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in predictions.items():
        coverage, risk = risk_coverage_curve(y, values)
        ax.plot(coverage, risk, label=name, marker="o" if name == "all_pass" else None)
    ax.set(xlabel="Accepted / all common successful test queries", ylabel="Bad / accepted",
           title=f"{label}\nRisk–coverage (exploratory test curve)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for name in ("logistic", "jev", "gpt"):
        if name not in predictions:
            continue
        points = reliability(y, predictions[name])
        ax.plot([p["predicted"] for p in points], [p["observed"] for p in points], "o-", label=name)
    ax.set(xlabel="Predicted P(at least one Exact)", ylabel="Observed fraction with Exact",
           xlim=(0, 1), ylim=(0, 1), title=f"{label}\nReliability (10 equal-width bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=160)
    plt.close(fig)
