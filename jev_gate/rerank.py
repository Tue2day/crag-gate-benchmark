from pathlib import Path
import time

from .data import file_digest
from .io import digest, metadata, read_json, read_jsonl, write_json, write_jsonl


def public_candidate(c, cfg):
    """Explicit allowlist: no ground truth, derived labels, or fake attributes."""
    return {"id": c["id"], "title": c["title"][:cfg.title_chars],
            "brand": c.get("brand"), "color": c.get("color"),
            "bullet_points": (c.get("bullet_points") or "")[:cfg.bullet_chars]}


def candidate_text(c, cfg):
    p = public_candidate(c, cfg)
    return "\n".join(f"{k}: {p[k]}" for k in ("title", "brand", "color", "bullet_points")
                     if p[k])


class TransformerReranker:
    def __init__(self, cfg):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; install a CUDA-enabled PyTorch build and check the driver")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available on this machine")
        source = cfg.reranker_local_path or cfg.reranker_model
        if cfg.reranker_local_path and not Path(source).is_dir():
            raise ValueError(f"Local model snapshot directory does not exist: {source}")
        kwargs = {"local_files_only": cfg.local_files_only, "trust_remote_code": False}
        if not cfg.reranker_local_path:
            kwargs["revision"] = cfg.reranker_revision
        config = AutoConfig.from_pretrained(source, **kwargs)
        revision = getattr(config, "_commit_hash", None)
        if cfg.reranker_local_path:
            path = Path(source)
            files = {str(p.relative_to(path)): file_digest(p) for p in sorted(path.rglob("*"))
                     if p.is_file() and ".cache" not in p.relative_to(path).parts}
            revision = "local-sha256:" + digest(files)
        elif not revision:
            raise ValueError("Cannot resolve model to a reproducible Hugging Face commit")
        else:
            kwargs["revision"] = revision
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, config=config, torch_dtype=dtype, **kwargs).to(device).eval()
        self.device = device
        self.batch_size = cfg.batch_size
        self.max_length = cfg.max_length
        self.identity = {"model": cfg.reranker_model, "revision": revision,
                         "device": device, "dtype": str(dtype), "max_length": cfg.max_length,
                         "configured_batch_size": cfg.batch_size}

    def score(self, pairs):
        torch = self.torch
        scores = []
        cursor = 0
        while cursor < len(pairs):
            batch = pairs[cursor:cursor + self.batch_size]
            try:
                encoded = self.tokenizer(batch, padding=True, truncation=True,
                                         max_length=self.max_length, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.inference_mode():
                    logits = self.model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] != 1:
                    raise ValueError("Expected a one-logit Cross-Encoder, not a multi-class model")
                scores.extend(float(s) for s in logits[:, 0].float().cpu().tolist())
                cursor += len(batch)
            except torch.cuda.OutOfMemoryError:
                if self.device != "cuda" or self.batch_size == 1:
                    raise ValueError("Reranker OOM at batch size 1; reduce max_length or use CPU")
                self.batch_size = max(1, self.batch_size // 2)
                # Release the failed batch before emptying the allocator cache.
                encoded = None
                logits = None
                torch.cuda.empty_cache()
                print(f"CUDA OOM: retrying with batch_size={self.batch_size}")
        return scores


class MockReranker:
    identity = {"model": "SIMULATED-reranker", "revision": "mock-v1", "device": "cpu"}
    batch_size = 0

    def score(self, pairs):
        scores = []
        for query, text in pairs:
            u = int(digest([query, text])[:8], 16) / 0xFFFFFFFF
            signal = 2.0 if "requested model" in text else (1.5 if "accessory" in text else 0)
            scores.append(signal + 4 * u - 2)
        return scores


def rerank(cfg, mock=False):
    groups = read_jsonl(cfg.root(mock) / "prepared.jsonl")
    if any(g.get("mock") != mock for g in groups):
        raise ValueError("Mixed mock/real input rejected")
    ranker = MockReranker() if mock else TransformerReranker(cfg)
    result, skips = [], []
    cache_hits = 0
    for i, group in enumerate(groups):
        candidates = group["candidates"]
        pairs = [(group["query"], candidate_text(c, cfg)) for c in candidates]
        key = digest({"pairs": pairs, "identity": ranker.identity})
        cache = cfg.root(mock) / "rerank_cache" / f"{key}.json"
        start = time.perf_counter()
        hit = cache.exists()
        if hit:
            scores = read_json(cache)["scores"]
            cache_hits += 1
        else:
            scores = ranker.score(pairs)
            write_json(cache, {"scores": scores, "identity": ranker.identity})
        if len(scores) != len(candidates):
            raise ValueError("Reranker score count mismatch")
        ranked = sorted(({**c, "ce_score": float(s)} for c, s in zip(candidates, scores)),
                        key=lambda c: (-c["ce_score"], c["id"]))[:5]
        if group["split"] == "challenge" and not any(c["label"] == "C" for c in ranked):
            skips.append(group["query_id"])
            continue
        result.append({**group, "candidates": ranked,
                       "y": int(any(c["label"] == "E" for c in ranked)),
                       "rerank_cache_hit": hit, "rerank_elapsed_ms": (time.perf_counter() - start) * 1000})
        if (i + 1) % 50 == 0:
            print(f"Reranked {i + 1}/{len(groups)}")
    write_jsonl(cfg.root(mock) / "ranked.jsonl", result)
    metadata(cfg, mock, "rerank", {"identity": ranker.identity,
        "effective_batch_size": ranker.batch_size, "cache_hits": cache_hits,
        "prepared_fingerprint": digest(groups), "ranked_fingerprint": digest(result),
        "challenge_skipped_no_complement_in_top5": skips,
        "development_no_exact_fraction": sum(g["y"] == 0 for g in result if g["split"] == "development") /
                                         max(1, sum(g["split"] == "development" for g in result))})
    print(f"Saved {len(result)} ranked query groups; skipped {len(skips)} challenge groups")
