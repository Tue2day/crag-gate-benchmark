import numpy as np
import pytest

from jev_gate.baselines import FEATURE_NAMES, features
from jev_gate.metrics import (accepted_mask, bad_average_precision, choose_threshold,
                              gate_metrics, paired_coverage_interval, reliability,
                              risk_coverage_curve, wilson)


def test_threshold_risk_and_minimum():
    y = [1, 1, 0, 1]
    op = choose_threshold(y, [0.9, 0.8, 0.7, 0.6], target_risk=0.2, min_accepted=2)
    assert op == {"available": True, "threshold": 0.8, "accepted": 2, "risk": 0.0}
    assert accepted_mask([0.9, 0.8, 0.7], op).tolist() == [True, True, False]
    assert not choose_threshold(y, [4, 3, 2, 1], min_accepted=30)["available"]


def test_ties_cannot_be_split():
    op = choose_threshold([1, 0, 1], [0.9, 0.9, 0.8], target_risk=0, min_accepted=1)
    assert not op["available"]
    assert risk_coverage_curve([1, 0, 1], [0.9, 0.9, 0.8])[0] == [2/3, 1]


def test_empty_acceptance_is_not_zero_risk():
    result = gate_metrics([1, 0], [False, False])
    assert result["risk"] is None
    assert result["risk_wilson_95"] is None
    assert result["coverage"] == 0
    assert wilson(0, 0) is None


def test_known_metrics_and_ap_orientation():
    result = gate_metrics([1, 0, 1, 0], [True, True, True, False])
    assert result["accepted"] == 3
    assert result["risk"] == 1/3
    assert result["coverage"] == 0.75
    assert result["risk_wilson_95"][0] < 1/3 < result["risk_wilson_95"][1]
    assert bad_average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1
    assert bad_average_precision([1, 1], [0.1, 0.2]) is None


def test_paired_bootstrap_and_single_class():
    assert paired_coverage_interval([1, 0, 1], [1, 0, 1], samples=50) == [0, 0]
    a = paired_coverage_interval([1, 1, 0], [0, 1, 0], samples=100)
    assert a == paired_coverage_interval([1, 1, 0], [0, 1, 0], samples=100)
    assert 0 <= a[0] <= a[1] <= 1
    assert choose_threshold([1]*30, list(range(30)))["accepted"] == 30
    assert not choose_threshold([0]*30, list(range(30)))["available"]


def test_probability_boundaries_in_reliability():
    result = reliability([0, 1], [0.0, 1.0])
    assert sum(r["n"] for r in result) == 2
    assert result[1]["observed"] == 1


def test_feature_order():
    group = {"candidates": [{"ce_score": x} for x in [5, 4, 2, 1, -2]]}
    x = features(group)
    assert len(x) == len(FEATURE_NAMES) == 11
    np.testing.assert_allclose(x[:9], [5, 4, 2, 1, -2, 1, 2, 1, 3])
    assert x[9] == 2
    with pytest.raises(ValueError):
        features({"candidates": [{"ce_score": x} for x in range(5)]})
