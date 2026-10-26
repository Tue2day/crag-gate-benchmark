"""GPT-4.1 Mini baseline: elicited JSON probability, explicitly not token logprobs."""
import json
import math

from .io import digest
from .jev import PROMPT, CRITERIA, parse_usage, valid_model, request_body as jev_body

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PROBABILITY_SOURCE = "gpt_elicited_json"


def request_body(group, cfg):
    state = jev_body(group, cfg)["state"]
    prompt, criteria = PROMPT, CRITERIA
    if cfg.task == "top1_exact":
        from .tasks import TOP1_PROMPT, TOP1_CRITERIA
        prompt, criteria = TOP1_PROMPT, TOP1_CRITERIA
    instruction = (prompt + "\nYes: " + criteria["true"] + "\nNo: " + criteria["false"] +
                   "\nReturn only JSON with probability_exact, your estimated probability (0 to 1) "
                   "that the answer is yes. Do not give an explanation.")
    return {"model": cfg.gpt_model, "temperature": 0, "max_tokens": cfg.gpt_max_tokens,
            "provider": {"require_parameters": True, "allow_fallbacks": False},
            "messages": [{"role": "system", "content": instruction},
                         {"role": "user", "content": json.dumps(state, ensure_ascii=False, sort_keys=True)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "exact_match_gate", "strict": True, "schema": {
                    "type": "object", "properties": {"probability_exact": {"type": "number"}},
                    "required": ["probability_exact"], "additionalProperties": False}}}}


def parse_response(data, expected_model):
    if not isinstance(data, dict) or not valid_model(data.get("model"), expected_model):
        raise ValueError("GPT response model does not match the requested family/build")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one GPT choice")
    choice = choices[0]
    message = choice.get("message", {})
    if choice.get("finish_reason") != "stop" or message.get("refusal"):
        raise ValueError("GPT response refused or truncated")
    try:
        content = json.loads(message.get("content", ""))
    except (ValueError, TypeError) as exc:
        raise ValueError("GPT response is not valid JSON") from exc
    if not isinstance(content, dict) or set(content) != {"probability_exact"}:
        raise ValueError("GPT JSON schema mismatch")
    p = content["probability_exact"]
    if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("GPT probability outside [0,1]")
    return float(p), parse_usage(data)


def mock_response(body):
    state = json.loads(body["messages"][1]["content"])
    candidates = [state["target_candidate"]] if state.get("task") == "top1_exact" else state["candidates"]
    exact = any("requested model" in c["title"] for c in candidates)
    u = int(digest(["gpt-mock-v1", state])[:8], 16) / 0xFFFFFFFF
    p = (0.7 + 0.29*u) if exact else (0.01 + 0.29*u)
    if u < 0.10:
        p = 1-p
    return {"status": "success", "probability": p, "model": "SIMULATED-"+body["model"],
            "probability_source": PROBABILITY_SOURCE,
            "usage": {"input_tokens": None, "output_tokens": None, "reported_cost_usd": None},
            "attempts": [], "total_elapsed_ms": None, "cost_may_exclude_failed_attempts": False}
