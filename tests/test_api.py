import asyncio
import copy
import socket
import httpx
import pytest

from jev_gate.jev import (AuthenticationError, RateLimiter, cache_key, call_api,
                          parse_response, request_body, run_async)
from jev_gate.data import prepare
from jev_gate.io import read_jsonl, write_jsonl
from jev_gate.rerank import rerank


def test_network_guard_blocks_real_http_and_arbitrary_localhost():
    with socket.socket() as sock:
        with pytest.raises(AssertionError, match="connections"):
            sock.connect(("127.0.0.1", 12345))
    with httpx.Client() as client:
        with pytest.raises(AssertionError, match="MockTransport"):
            client.get("https://openrouter.ai")


def payload(model="typesafe/jev-1.13", p=0.85):
    return {"model": model, "answers": {"has_exact_match": {"type": "noul", "noul": p}},
            "usage": {"input_tokens": 1000, "output_tokens": 5}}


def test_response_validation():
    assert parse_response(payload(), "typesafe/jev-1.13") == (0.85, {"input_tokens": 1000, "output_tokens": 5, "reported_cost_usd": None})
    for bad in [payload(p=2), payload(p=True), payload(p=float("nan")), payload(model="jev-latest"), {}]:
        with pytest.raises(ValueError):
            parse_response(bad, "typesafe/jev-1.13")


def test_no_ground_truth_leak_and_cache_invalidation(cfg):
    prepare(cfg, True)
    rerank(cfg, True)
    group = read_jsonl(cfg.root(True)/"ranked.jsonl")[0]
    body = request_body(group, cfg)
    import json
    text = json.dumps(body)
    assert '"label"' not in text and '"y"' not in text and '"constructed"' not in text
    changed = copy.deepcopy(group)
    changed["y"] = 1-changed["y"]
    changed["candidates"][0]["label"] = "I"
    assert request_body(changed, cfg) == body
    original_key = cache_key(body)
    changed_body = copy.deepcopy(body)
    changed_body["state"]["candidates"][0]["title"] += " changed"
    assert cache_key(changed_body) != original_key
    changed_body = copy.deepcopy(body)
    changed_body["model"] = "typesafe/jev-1.14"
    assert cache_key(changed_body) != original_key
    changed_body = copy.deepcopy(body)
    changed_body["questions"]["has_exact_match"]["instructions"] += " new"
    assert cache_key(changed_body) != original_key
    assert cache_key(body, True) != original_key


def test_retry_and_authentication(cfg):
    calls = []
    delays = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json=payload())
    async def sleep(seconds):
        delays.append(seconds)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await call_api(client, {"model": cfg.jev_model}, cfg,
                                    RateLimiter(100000), sleep=sleep)
        assert result["status"] == "success"
        assert len(result["attempts"]) == 2
        assert result["cost_may_exclude_failed_attempts"]
        assert delays == [0]
        for code in (401, 402, 403):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(code))) as client:
                with pytest.raises(AuthenticationError):
                    await call_api(client, {}, cfg, RateLimiter(100000), sleep=sleep)
    asyncio.run(scenario())


def test_exhausted_retry_and_invalid_response(cfg):
    async def sleep(seconds):
        pass
    async def scenario():
        for status, body, count in [(503, {}, 3), (400, {}, 1), (200, {"bad": True}, 1)]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda r: httpx.Response(status, json=body))) as client:
                result = await call_api(client, {}, cfg, RateLimiter(100000), sleep=sleep)
            assert result["status"] == "failed"
            assert len(result["attempts"]) == count
            assert result["probability"] is None
        def timeout(request):
            raise httpx.ReadTimeout("simulated timeout", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
            result = await call_api(client, {}, cfg, RateLimiter(100000), sleep=sleep)
        assert result["error"] == "transport_error"
        assert len(result["attempts"]) == cfg.max_attempts
    asyncio.run(scenario())


def test_real_client_contract_and_resume_without_network(cfg, monkeypatch):
    prepare(cfg, True)
    rerank(cfg, True)
    groups = read_jsonl(cfg.root(True)/"ranked.jsonl")[:3]
    for group in groups:
        group["mock"] = False
    write_jsonl(cfg.root()/"ranked.jsonl", groups)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-never-log")
    calls = []
    def handler(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-secret-never-log"
        assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
        return httpx.Response(200, json=payload())
    asyncio.run(run_async(cfg, False, transport=httpx.MockTransport(handler)))
    assert len(calls) == 3
    asyncio.run(run_async(cfg, False, transport=httpx.MockTransport(handler)))
    assert len(calls) == 3
    records = read_jsonl(cfg.root()/"jev.jsonl")
    assert all(r["cache_hit"] for r in records)
    assert "test-secret-never-log" not in (cfg.root()/"jev.jsonl").read_text(encoding="utf-8")


def test_missing_key_never_uses_mock(cfg, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    write_jsonl(cfg.root()/"ranked.jsonl", [])
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY missing"):
        asyncio.run(run_async(cfg))
