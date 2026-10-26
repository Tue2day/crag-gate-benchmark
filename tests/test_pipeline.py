import asyncio
import json
from pathlib import Path
import pytest

from jev_gate.baselines import baselines
from jev_gate.data import prepare
from jev_gate.evaluate import evaluate
from jev_gate.io import read_json, read_jsonl, write_jsonl
from jev_gate.jev import run_async
from jev_gate.rerank import rerank


def build_mock(cfg):
    prepare(cfg, True)
    rerank(cfg, True)
    baselines(cfg, True)
    asyncio.run(run_async(cfg, True))
    asyncio.run(run_async(cfg, True, method="gpt"))


def test_full_offline_pipeline_and_resume(cfg, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(Path(cfg.run_dir)/"mpl-cache"))
    build_mock(cfg)
    report = evaluate(cfg, True)
    assert report["mock"]
    assert report["completeness"]["natural_test_n"] == 80
    assert set(report["methods"]) == {"all_pass", "ce_threshold", "logistic", "jev", "gpt"}
    out = cfg.root(True)/"report"
    for filename in ["summary.csv", "metrics.json", "errors.csv", "manual_audit.csv",
                     "REPORT.md", "risk_coverage.png", "reliability.png"]:
        assert (out/filename).stat().st_size > 0
    assert "SIMULATED" in (out/"REPORT.md").read_text(encoding="utf-8")
    assert report["methods"]["jev"]["latency"]["n"] == 0
    assert report["cost"]["by_model"]["jev"]["unique_natural_test_predictions"]["known_token_estimated_cost_usd"] is None
    operating = read_json(cfg.root(True)/"operating_points.json")
    asyncio.run(run_async(cfg, True))
    asyncio.run(run_async(cfg, True, method="gpt"))
    again = evaluate(cfg, True)
    assert read_json(cfg.root(True)/"operating_points.json") == operating
    assert again["methods"]["jev"]["coverage"] == report["methods"]["jev"]["coverage"]
    assert all(r["cache_hit"] for r in read_jsonl(cfg.root(True)/"jev.jsonl"))
    assert not cfg.root(False).exists()


def test_failures_excluded_from_comparison_not_labeled_correct(cfg):
    build_mock(cfg)
    records = read_jsonl(cfg.root(True)/"jev.jsonl")
    failed = next(r for r in records if r["split"] == "test")
    failed.update(status="failed", probability=None, error="http_503")
    write_jsonl(cfg.root(True)/"jev.jsonl", records)
    report = evaluate(cfg, True)
    assert report["completeness"]["common_success_test_n"] == 79
    assert report["completeness"]["jev_failed_or_missing_test_n"] == 1
    assert all(m["n"] == 79 for m in report["methods"].values())
    assert report["operational_all_test"]["jev"]["n"] == 80


def test_no_working_point_is_reported(cfg):
    cfg.min_accepted = 1000
    build_mock(cfg)
    report = evaluate(cfg, True)
    for name in ("ce_threshold", "logistic", "jev", "gpt"):
        assert not report["methods"][name]["operating_point_available"]
        assert report["methods"][name]["risk"] is None
        assert report["methods"][name]["challenge_intercept_rate"] is None


def test_single_class_training_marked_unavailable_without_fake_predictions(cfg):
    prepare(cfg, True)
    rerank(cfg, True)
    groups = read_jsonl(cfg.root(True)/"ranked.jsonl")
    for g in groups:
        if g["split"] == "development":
            g["y"] = 1
    write_jsonl(cfg.root(True)/"ranked.jsonl", groups)
    baselines(cfg, True)
    assert read_json(cfg.root(True)/"baselines_metadata.json")["logistic_status"] == "unavailable"
    assert all(r["logistic"] is None for r in read_jsonl(cfg.root(True)/"baselines.jsonl"))
    asyncio.run(run_async(cfg, True))
    asyncio.run(run_async(cfg, True, method="gpt"))
    report = evaluate(cfg, True)
    assert report["methods"]["logistic"]["status"] == "unavailable"
    assert report["methods"]["logistic"]["coverage"] is None
    assert report["methods"]["logistic"]["brier"] is None
    assert report["methods"]["jev"]["n"] == 80


def test_frozen_protocol_and_mixed_mode_rejected(cfg):
    build_mock(cfg)
    evaluate(cfg, True)
    cfg.target_risk = 0.2
    with pytest.raises(ValueError, match="Frozen"):
        evaluate(cfg, True)
    cfg.target_risk = 0.05
    records = read_jsonl(cfg.root(True)/"jev.jsonl")
    records[0]["mock"] = False
    write_jsonl(cfg.root(True)/"jev.jsonl", records)
    with pytest.raises(ValueError, match="Mixed"):
        evaluate(cfg, True)
