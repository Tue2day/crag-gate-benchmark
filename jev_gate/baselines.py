import time
import numpy as np

from .io import digest, metadata, read_jsonl, write_jsonl

FEATURE_NAMES = [f"top_{i}_raw" for i in range(1, 6)] + [
    f"gap_{i}_{i+1}" for i in range(1, 5)] + ["mean", "std"]


def features(group):
    scores = np.asarray([c["ce_score"] for c in group["candidates"]], dtype=float)
    if len(scores) != 5 or not np.isfinite(scores).all() or (np.diff(scores) > 0).any():
        raise ValueError("Expected 5 finite descending Cross-Encoder scores")
    return np.r_[scores, scores[:-1] - scores[1:], scores.mean(), scores.std()]


def baselines(cfg, mock=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    groups = read_jsonl(cfg.root(mock) / "ranked.jsonl")
    if any(g.get("mock") != mock for g in groups):
        raise ValueError("Mixed mock/real input rejected")
    dev = [g for g in groups if g["split"] == "development"]
    y = [g["y"] for g in dev]
    if not dev:
        raise ValueError("Development set is empty")
    model = None
    reason = None
    if len(set(y)) != 2:
        reason = f"Development labels have one class: {sum(y)}/{len(y)} contain Exact; logistic cannot be fitted"
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000,
                                                                   random_state=cfg.seed))
        weights = np.asarray([g.get("sampling_weight", 1.0) for g in dev], dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid inverse-probability sampling weights")
        x_dev = np.asarray([features(g) for g in dev])
        model[0].fit(x_dev, sample_weight=weights)
        model[1].fit(model[0].transform(x_dev), y, sample_weight=weights)
    result = []
    for group in groups:
        start = time.perf_counter()
        x = features(group)
        feature_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        p = float(model.predict_proba(x.reshape(1, -1))[0, 1]) if model is not None else None
        logistic_ms = (time.perf_counter() - start) * 1000 + feature_ms if model is not None else None
        result.append({"query_id": group["query_id"], "split": group["split"],
                       "mock": mock, "task": cfg.task, "all_pass": 1.0, "ce_threshold": float(x[0]), "logistic": p,
                       "latency_ms": {"all_pass": 0.0, "ce_threshold": 0.0, "logistic": logistic_ms}})
    write_jsonl(cfg.root(mock) / "baselines.jsonl", result)
    metadata(cfg, mock, "baselines", {"ranked_fingerprint": digest(groups),
        "features": FEATURE_NAMES, "scaler_mean": model[0].mean_.tolist() if model is not None else None,
        "scaler_scale": model[0].scale_.tolist() if model is not None else None,
        "coefficients": model[1].coef_.tolist() if model is not None else None,
        "intercept": model[1].intercept_.tolist() if model is not None else None, "training_n": len(dev),
        "development_class_counts": {"no_exact": y.count(0), "has_exact": y.count(1)},
        "logistic_status": "available" if model is not None else "unavailable",
        "task": cfg.task,
        "sampling_weights": [g.get("sampling_weight", 1.0) for g in dev],
        "logistic_unavailable_reason": reason})
    if reason:
        print(f"WARNING: {reason}. No surrogate probabilities were generated.")
    print(f"Saved {len(result)} baseline rows; logistic {'trained' if model is not None else 'unavailable'}")
