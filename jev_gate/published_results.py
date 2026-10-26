"""Offline verification of the public, derived benchmark exports.

This entry point never constructs an HTTP client, downloads data, or runs a
model. It checks file hashes, decisions, calibration, paired-query statistics,
recorded latency quantiles, and usage receipts.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .io import read_json, read_jsonl

METRICS = ('query_contamination_risk', 'kept_precision', 'exact_retention',
           'query_coverage', 'safe_route_success', 'rescue_success_rate', 'candidate_f1')


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return bool(np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _selected(rows, method, threshold):
    return [r['product_id'] for r in rows if (r['core_probability'] >= threshold if method == 'jev' else r['core'])
            and not r['accessory'] and r['constraint'] == 'NONE']


def _counts(rows, selected):
    selected = set(selected)
    good = {r['product_id'] for r in rows if r['label'] == 'E'}
    bad = {r['product_id'] for r in rows} - good
    tp, fp, fn, tn = len(selected & good), len(selected & bad), len(good - selected), len(bad - selected)
    passed, has = bool(selected), bool(good)
    rescue = rows[0]['label'] != 'E' and has
    action = 'FALLBACK' if not selected else ('PASS_ALL' if len(selected) == 5 else 'PASS_FILTERED')
    oracle = 'FALLBACK' if not good else ('PASS_ALL' if len(good) == 5 else 'PASS_FILTERED')
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'kept': tp + fp,
            'passed': int(passed), 'polluted': int(fp > 0), 'has_exact': int(has),
            'unnecessary_fallback': int(has and not passed), 'route_correct': int(action == oracle),
            'safe_route_success': int(passed and fp == 0) if has else int(not passed),
            'rescue_eligible': int(rescue),
            'rescue_success': int(rescue and passed and fp == 0 and rows[0]['product_id'] not in selected),
            **{f'{label}_total': sum(r['label'] == label for r in rows) for label in ('S', 'C', 'I')},
            **{f'{label}_blocked': sum(r['label'] == label and r['product_id'] not in selected for r in rows)
               for label in ('S', 'C', 'I')}}


def _metrics(total, n):
    def ratio(x, y):
        return x / y if y else None
    tp, fp, fn = (total[k] for k in ('tp', 'fp', 'fn'))
    return {'kept_precision': ratio(tp, tp + fp), 'exact_retention': ratio(tp, tp + fn),
            'candidate_f1': ratio(2 * tp, 2 * tp + fp + fn),
            'query_coverage': ratio(total['passed'], n),
            'query_contamination_risk': ratio(total['polluted'], total['passed']),
            'safe_route_success': ratio(total['safe_route_success'], n),
            'rescue_success_rate': ratio(total['rescue_success'], total['rescue_eligible'])}


def _aggregate(counts):
    total = Counter()
    for row in counts:
        total.update(row)
    return dict(total), _metrics(total, len(counts))


def _bootstrap(a, b, samples, seed):
    keys = sorted(a[0])
    av = np.asarray([[r[k] for k in keys] for r in a])
    bv = np.asarray([[r[k] for k in keys] for r in b])
    rng = np.random.default_rng(seed)
    differences = {k: [] for k in METRICS}
    for _ in range(samples):
        ii = rng.integers(0, len(a), len(a))
        aa = _metrics(dict(zip(keys, av[ii].sum(axis=0))), len(a))
        bb = _metrics(dict(zip(keys, bv[ii].sum(axis=0))), len(b))
        for k in METRICS:
            if aa[k] is not None and bb[k] is not None:
                differences[k].append(aa[k] - bb[k])
    intervals = {k: np.quantile(v, [.025, .975]).tolist() if v else None for k, v in differences.items()}
    lower = float(np.quantile(differences['exact_retention'], .05)) if differences['exact_retention'] else None
    return intervals, lower


def _jsonl_bytes(data):
    # U+0085/U+2028/U+2029 can occur inside valid JSON strings. JSONL framing
    # uses physical LF bytes, not str.splitlines()'s Unicode line boundaries.
    return (json.loads(line) for line in data.split(b'\n') if line.strip())


def _raw_archive(path, experiment, indexed):
    from . import topk
    with ZipFile(path) as archive:
        names = archive.namelist()
        _require(all(not n.startswith(('/', '\\')) and '..' not in Path(n).parts for n in names), 'Unsafe release archive paths')
        manifest = json.loads(archive.read('manifest.json'))
        for name, expected in manifest['sha256'].items():
            _require(hashlib.sha256(archive.read(name)).hexdigest() == expected, 'Release archive checksum mismatch')
        judgments = {g['query_id']: g for g in _jsonl_bytes(archive.read(f'{experiment}/judgments.jsonl'))}
        n = 0
        for method in ('jev', 'gpt'):
            rows = _jsonl_bytes(archive.read(f'{experiment}/{method}_responses.jsonl'))
            for row in rows:
                g = judgments[row['query_id']]
                parsed, usage = topk.decoder(g, method)(row['response'], row['model'])
                _require(parsed == row['prediction'] and usage == row['usage'], 'Derived response differs from native API answer')
                published = indexed[(g['split'], g['query_id'], method)]
                for actual, expected in zip(parsed['candidate_decisions'], published):
                    _require(actual['id'] == expected['product_id'] and actual['is_core_intent_matched'] == expected['core']
                             and actual['is_accessory_trap'] == expected['accessory']
                             and actual['primary_constraint'] == expected['constraint'], 'Raw/public candidate decision mismatch')
                    if method == 'jev':
                        _require(actual['core_probability'] == expected['core_probability']
                                 and actual['accessory_probability'] == expected['accessory_probability'], 'Native Noul probability mismatch')
                n += 1
    return n


def verify(experiment='v5', artifacts=None, raw_artifacts=None):
    default = Path(__file__).resolve().parents[1] / 'benchmarks' / experiment
    root = Path(artifacts) if artifacts else default
    checksums = read_json(root / 'checksums.json')['sha256']
    for name, expected in checksums.items():
        relative = Path(name)
        _require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path')
        p = root / relative
        _require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == expected, f'Artifact checksum mismatch: {name}')
    report = read_json(root / 'metrics.json')
    protocol = read_json(root / 'protocol.json')
    rows = read_jsonl(root / 'candidates.jsonl')
    indexed = defaultdict(list)
    for r in rows:
        _require(r['method'] in ('jev', 'gpt') and r['label'] in ('E', 'S', 'C', 'I'), 'Invalid method/label')
        _require(type(r['core']) is bool and type(r['accessory']) is bool, 'Expected native boolean decisions')
        indexed[(r['split'], r['query_id'], r['method'])].append(r)
    for key, group in indexed.items():
        group.sort(key=lambda r: r['rank'])
        _require(len(group) == 5 and [r['rank'] for r in group] == [1, 2, 3, 4, 5]
                 and len({r['product_id'] for r in group}) == 5, 'Expected five unique ranked candidates')
        if key[-1] == 'jev':
            _require(all(isinstance(r['core_probability'], (int, float)) and 0 <= r['core_probability'] <= 1
                         and (r['core_probability'] >= .5) == r['core'] for r in group), 'Invalid native core probabilities')
        other = indexed.get((*key[:2], 'gpt' if key[-1] == 'jev' else 'jev'))
        _require(other is not None, 'Unpaired query')
    queries = [(r['split'], r['query_id']) for r in read_json(root / 'query_order.json')]
    _require(len(queries) == len(set(queries)) and set(queries) == {(r['split'], r['query_id']) for r in rows}, 'Query replay order mismatch')
    for split, qid in queries:
        j, g = indexed[(split, qid, 'jev')], indexed[(split, qid, 'gpt')]
        _require([(r['product_id'], r['label']) for r in j] == [(r['product_id'], r['label']) for r in g], 'Models must share ranked gold candidates')
    predictions = read_jsonl(root / 'predictions.jsonl')
    _require(len(predictions) == len(indexed) and {(p['split'], p['query_id'], p['method']) for p in predictions} == set(indexed), 'Missing or duplicate published predictions')
    for p in predictions:
        selected = _selected(indexed[(p['split'], p['query_id'], p['method'])], p['method'], .5)
        _require(selected == p['selected_ids'], 'Default raw decision differs from published candidates')
    threshold = .5
    if experiment == 'v5':
        frozen = read_json(root / 'threshold.json')
        calibration = [qid for split, qid in queries if split == 'calibration']
        _require(len(calibration) == 800, 'Expected 800 calibration queries')
        _, gpt = _aggregate([_counts(indexed[('calibration', qid, 'gpt')], _selected(indexed[('calibration', qid, 'gpt')], 'gpt', .5)) for qid in calibration])
        grid = []
        for t in protocol['config']['core_threshold_grid']:
            _, value = _aggregate([_counts(indexed[('calibration', qid, 'jev')], _selected(indexed[('calibration', qid, 'jev')], 'jev', t)) for qid in calibration])
            feasible = value['exact_retention'] >= gpt['exact_retention'] - protocol['config']['calibration_retention_slack']
            grid.append((t, value, feasible))
        available = [(t, v) for t, v, feasible in grid if feasible and v['kept_precision'] is not None]
        _require(available, 'No feasible calibration threshold')
        threshold = max(available, key=lambda x: (x[1]['kept_precision'], x[0]))[0]
        _require(threshold == frozen['threshold'] == report['threshold'] == .35, 'Frozen calibration threshold mismatch')
        for (t, value, feasible), saved in zip(grid, frozen['grid_results']):
            _require(t == saved['threshold'] and feasible == saved['feasible']
                     and _close(value['exact_retention'], saved['exact_retention'])
                     and _close(value['kept_precision'], saved['kept_precision']), 'Calibration grid mismatch')
    test_ids = [qid for split, qid in queries if split == 'test']
    _require(len(test_ids) == (2000 if experiment == 'v5' else 500), 'Wrong natural test count')
    counts = {}
    for method in ('jev', 'gpt'):
        counts[method] = [_counts(indexed[('test', qid, method)], _selected(indexed[('test', qid, method)], method, threshold)) for qid in test_ids]
        total, observed = _aggregate(counts[method])
        _require(total == report['metrics'][method]['counts'], 'Test confusion/query counts mismatch')
        for k, v in observed.items():
            _require(_close(v, report['metrics'][method][k]), f'Test metric mismatch: {method}/{k}')
    intervals, lower = _bootstrap(counts['jev'], counts['gpt'], protocol['config']['bootstrap_samples'], protocol['config']['seed'])
    for k in METRICS:
        _require(_close(intervals[k], report['bootstrap']['delta_ci95'][k]), f'Paired bootstrap mismatch: {k}')
    if experiment == 'v5':
        _require(_close(lower, report['bootstrap']['retention_delta_one_sided95_lower']), 'One-sided retention bound mismatch')
        _require((lower > -.02) == report['conclusions']['retention_noninferiority'] == False, 'Non-inferiority result changed')
        _require((intervals['kept_precision'][0] > 0) == report['conclusions']['precision_superiority'] == True, 'Precision conclusion changed')
        _require(report['conclusions']['joint_quality_evidence'] is False, 'Joint quality conclusion changed')
    timing = read_jsonl(root / 'timings.jsonl')
    costs = read_json(root / 'costs.json')
    for method in ('jev', 'gpt'):
        records = [r for r in timing if r['method'] == method]
        _require(len(records) == len({(r['split'], r['query_id']) for r in records}), 'Duplicate timing invocations')
        tests = [r for r in records if r['split'] == 'test']
        reference = report['latency'] if experiment == 'v5' else report['latency_uncached_history']
        for metric, values in [('http', [a['elapsed_ms'] for r in tests for a in r['attempts']]), ('gate', [r['gate_ms'] for r in tests])]:
            _require(_close(np.quantile(values, [.5, .95]), [reference[method][metric]['p50_ms'], reference[method][metric]['p95_ms']]), 'Recorded latency quantiles mismatch')
        _require(_close(sum(r['known_cost_usd'] for r in records), costs[method]['known_cost_usd']), 'Usage receipt sum mismatch')
        _require(_close(costs[method]['known_cost_usd'], report['costs'][method]['known_cost_usd']), 'Reported cost mismatch')
    raw_count = _raw_archive(raw_artifacts, experiment, indexed) if raw_artifacts else None
    return {'experiment': experiment, 'verified': True, 'network_requests': 0,
            'natural_queries': len(test_ids), 'candidate_decisions': len(rows), 'frozen_core_threshold': threshold,
            'bootstrap_samples': protocol['config']['bootstrap_samples'], 'all_seven_delta_intervals_match': True,
            'raw_api_records_verified': raw_count, 'note': 'Verification replays recorded evidence; it is not a new model benchmark.'}
