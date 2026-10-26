"""Metrics have explicit denominators. No accepted queries means undefined risk."""
from math import sqrt
import numpy as np


def choose_threshold(y, score, target_risk=0.05, min_accepted=30):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError("Invalid threshold inputs")
    if not len(y):
        return {"available": False, "threshold": None, "accepted": 0, "risk": None}
    # Sort once; evaluate only complete tie groups so threshold >= never splits ties.
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    errors = np.cumsum(1 - y[order])
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y) - 1]
    valid = [int(i) for i in ends if i + 1 >= min_accepted and errors[i] / (i + 1) <= target_risk]
    if not valid:
        return {"available": False, "threshold": None, "accepted": 0, "risk": None}
    i = valid[-1]
    return {"available": True, "threshold": float(sorted_scores[i]),
            "accepted": i + 1, "risk": float(errors[i] / (i + 1))}


def accepted_mask(score, operating_point):
    return (np.asarray(score) >= operating_point["threshold"] if operating_point["available"]
            else np.zeros(len(score), dtype=bool))


def wilson(errors, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p = errors / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0.0, center-half), min(1.0, center+half)]


def gate_metrics(y, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    n = len(y)
    accepted = int(mask.sum())
    errors = int(((y == 0) & mask).sum())
    return {"n": n, "accepted": accepted, "bad_accepted": errors,
            "coverage": accepted/n if n else None,
            "risk": errors/accepted if accepted else None,
            "risk_wilson_95": wilson(errors, accepted)}


def paired_coverage_interval(a, b, samples=2000, seed=42):
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not len(difference):
        return None
    rng = np.random.default_rng(seed)
    means = [float(difference[rng.integers(0, len(difference), len(difference))].mean())
             for _ in range(samples)]
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_ap_interval(y,a,b,samples=2000,seed=42):
    y=np.asarray(y);a=np.asarray(a);b=np.asarray(b)
    if len(np.unique(y))<2: return None
    rng=np.random.default_rng(seed);differences=[]
    for _ in range(samples):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))<2: continue
        differences.append(bad_average_precision(y[ix],a[ix])-bad_average_precision(y[ix],b[ix]))
    return {'delta':bad_average_precision(y,a)-bad_average_precision(y,b),
            'ci95':[float(x) for x in np.quantile(differences,[0.025,0.975])],
            'valid_bootstrap_samples':len(differences)} if differences else None


def bad_average_precision(y, good_score):
    from sklearn.metrics import average_precision_score
    bad = 1 - np.asarray(y)
    if len(np.unique(bad)) < 2:
        return None
    return float(average_precision_score(bad, -np.asarray(good_score)))


def reliability(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    result = []
    for i in range(bins):
        mask = (p >= i/bins) & ((p < (i+1)/bins) if i < bins-1 else (p <= 1))
        if mask.any():
            result.append({"bin": i, "n": int(mask.sum()),
                           "predicted": float(p[mask].mean()), "observed": float(y[mask].mean())})
    return result


def risk_coverage_curve(y, score):
    y, score = np.asarray(y), np.asarray(score)
    order = np.argsort(-score, kind="stable")
    if not len(order):
        return [], []
    ends = np.r_[np.flatnonzero(score[order][:-1] != score[order][1:]), len(y)-1]
    return ((ends + 1)/len(y)).tolist(), (np.cumsum(1-y[order])[ends]/(ends+1)).tolist()
