import asyncio
import copy
import json

import httpx
import pytest

from jev_gate import gpt
from jev_gate.data import prepare
from jev_gate.evaluate import evaluate
from jev_gate.io import read_json, read_jsonl, write_jsonl
from jev_gate.jev import cache_key, parse_response as jev_parse, run_async
from jev_gate.rerank import rerank
from test_pipeline import build_mock


def response(p=0.85, model="openai/gpt-4.1-mini-2025-04-14"):
    return {"model": model, "provider": "OpenAI", "id": "test-request",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                "content": json.dumps({"probability_exact": p})}}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 12, "cost": 0.00062}}


def test_jev_openrouter_dated_build_response(cfg):
    data = {"model": "typesafe/jev-1.13-20260917",
            "answers": {"has_exact_match": {"type": "noul", "noul": 0.8}},
            "usage": {"input_tokens": 1000, "output_tokens": 1, "cost": 0.000042}}
    p, usage = jev_parse(data, cfg.jev_model)
    assert p == 0.8 and usage["reported_cost_usd"] == 0.000042
    with pytest.raises(ValueError):
        jev_parse({**data, "model": "typesafe/jev-1.14-20261014"}, cfg.jev_model)


def test_gpt_response_requires_valid_schema_and_finish(cfg):
    p, usage = gpt.parse_response(response(), cfg.gpt_model)
    assert p == 0.85 and usage["input_tokens"] == 1500 and usage["output_tokens"] == 12
    bad_responses = [response(p=True), response(p=-0.1), response(p=float("nan")),
                     response(model="openai/gpt-4.1"), response(model="other/gpt-4.1-mini")]
    for finish in ("length", "content_filter"):
        bad = response()
        bad["choices"][0]["finish_reason"] = finish
        bad_responses.append(bad)
    bad = response()
    bad["choices"][0]["message"]["content"] = '{"probability_exact":0.7,"reason":"extra"}'
    bad_responses.append(bad)
    bad = response()
    bad["choices"][0]["message"]["refusal"] = "refused"
    bad_responses.append(bad)
    for bad in bad_responses:
        with pytest.raises(ValueError):
            gpt.parse_response(bad, cfg.gpt_model)


def test_gpt_uses_identical_state_and_isolated_cache(cfg):
    from jev_gate.jev import request_body
    prepare(cfg, True)
    rerank(cfg, True)
    group = read_jsonl(cfg.root(True)/"ranked.jsonl")[0]
    body = gpt.request_body(group, cfg)
    assert json.loads(body["messages"][1]["content"]) == request_body(group, cfg)["state"]
    assert '"label"' not in json.dumps(body) and '"y"' not in json.dumps(body)
    assert body["temperature"] == 0 and body["response_format"]["json_schema"]["strict"]
    assert body["provider"]["require_parameters"] is True
    assert cache_key(body, endpoint=gpt.ENDPOINT) != cache_key(body)
    changed = copy.deepcopy(body)
    changed["max_tokens"] += 1
    assert cache_key(changed, endpoint=gpt.ENDPOINT) != cache_key(body, endpoint=gpt.ENDPOINT)


def test_gpt_transport_and_resume(cfg, monkeypatch):
    prepare(cfg, True)
    rerank(cfg, True)
    groups = read_jsonl(cfg.root(True)/"ranked.jsonl")[:3]
    for group in groups:
        group["mock"] = False
    write_jsonl(cfg.root()/"ranked.jsonl", groups)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    calls = []
    def handler(request):
        calls.append(request)
        assert str(request.url) == gpt.ENDPOINT
        assert request.headers["authorization"] == "Bearer test-openrouter-key"
        assert json.loads(request.content)["model"] == cfg.gpt_model
        return httpx.Response(200, json=response())
    for _ in range(2):
        asyncio.run(run_async(cfg, transport=httpx.MockTransport(handler), method="gpt"))
    assert len(calls) == 3
    records = read_jsonl(cfg.root()/"gpt.jsonl")
    assert all(r["cache_hit"] and r["probability_source"] == "gpt_elicited_json" for r in records)
    assert read_json(cfg.root()/"gpt_metadata.json")["resolved_models"] == ["openai/gpt-4.1-mini-2025-04-14"]


def test_gpt_failure_excludes_same_query_from_all_five_methods(cfg):
    build_mock(cfg)
    records = read_jsonl(cfg.root(True)/"gpt.jsonl")
    failed = next(r for r in records if r["split"] == "test")
    failed.update(status="failed", probability=None)
    write_jsonl(cfg.root(True)/"gpt.jsonl", records)
    report = evaluate(cfg, True)
    assert report["completeness"]["gpt_failed_or_missing_test_n"] == 1
    assert report["completeness"]["jev_failed_or_missing_test_n"] == 0
    assert all(m["n"] == 79 for m in report["methods"].values())
    assert report["methods"]["gpt"]["probability_source"] == "gpt_elicited_json"
    assert "gpt" in report["model_identities"]


def test_gpt_cost_includes_output_and_reported_charge(cfg):
    build_mock(cfg)
    records = read_jsonl(cfg.root(True)/"gpt.jsonl")
    for r in records:
        r["usage"] = {"input_tokens": 1000, "output_tokens": 100, "reported_cost_usd": 0.00055}
    write_jsonl(cfg.root(True)/"gpt.jsonl", records)
    report = evaluate(cfg, True)
    costs = report["cost"]["by_model"]["gpt"]["unique_natural_test_predictions"]
    assert costs["estimated_usd_per_1000_successful_queries"] == pytest.approx(0.56)
    assert costs["reported_usd_per_1000_successful_queries"] == pytest.approx(0.55)


def test_old_typesafe_key_is_not_used(cfg, monkeypatch):
    write_jsonl(cfg.root()/"ranked.jsonl", [])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "wrong-provider-key")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY missing"):
        asyncio.run(run_async(cfg))


def test_mixed_resolved_builds_rejected_by_evaluator(cfg):
    build_mock(cfg)
    records = read_jsonl(cfg.root(True)/"gpt.jsonl")
    records[0]["model"] += "-different-build"
    write_jsonl(cfg.root(True)/"gpt.jsonl", records)
    with pytest.raises(ValueError, match="Multiple resolved"):
        evaluate(cfg, True)
