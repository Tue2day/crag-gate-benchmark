"""Pre-specified one-dimensional Jev calibration, followed by untouched testing.

API evidence, prompts and raw parsers are identical to v4. Only the local core
threshold is fitted; accessory < .5 and constraint == NONE stay fixed.
"""
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np

from . import topk
from .budget import Budget
from .config import Config
from .data import normalize_query, synthetic_groups
from .evaluate import percentile_summary, write_csv
from .fusion import catalog_quote, source_hashes
from .io import digest, metadata, read_json, read_jsonl, write_json, write_jsonl
from .jev import ENDPOINT, RateLimiter, cache_key, call_api
from .gpt import ENDPOINT as GPT_ENDPOINT
from .rerank import MockReranker, TransformerReranker, candidate_text
from .top1 import load_groups

PROTOCOL = 'topk_calibrated_v5'
METHODS = ('jev', 'gpt')


def partition(groups, cfg, excluded=(), exclusion_hashes=()):
    seen = set(excluded)
    pools = {'train': [], 'test': []}
    for g in sorted(groups, key=lambda g: (g['source_split'] != 'train', g['query_id'])):
        q = normalize_query(g['query'])
        if q and q not in seen and digest(q) not in exclusion_hashes and len(g['candidates']) >= 5:
            seen.add(q)
            pools[g['source_split']].append(g)
    rng = random.Random(cfg.seed)
    for pool in pools.values():
        rng.shuffle(pool)
    if len(pools['train']) < cfg.calibration_size or len(pools['test']) < cfg.test_size:
        raise ValueError('Insufficient untouched data; never reduce the fixed sample sizes')
    rows = [{**g, 'split': split, 'task': topk.TASK, 'experiment_protocol': PROTOCOL, 'constructed': False}
            for split, pool, size in [('calibration', pools['train'], cfg.calibration_size),
                                      ('test', pools['test'], cfg.test_size)] for g in pool[:size]]
    if len({g['query_id'] for g in rows}) != len(rows):
        raise ValueError('Query IDs must also be disjoint')
    return rows


def prepare(cfg, mock=False):
    root = cfg.root(mock)
    if (root / 'protocol_frozen.json').exists():
        raise ValueError('Protocol already frozen; cannot prepare again')
    prior, excluded, sources = [], set(), {}
    exclusion_hashes = set()
    if mock:
        proto = synthetic_groups(cfg.seed)
        groups = []
        for i in range(cfg.calibration_size + cfg.test_size + 20):
            g = proto[i % len(proto)]
            groups.append({**g, 'query_id': f'v5m{i}', 'query': f'v5 mock {i} {g["query"]}',
                           'source_split': 'train' if i < cfg.calibration_size + 10 else 'test',
                           'candidates': [{**c, 'id': f'{i}-{c["id"]}'} for c in g['candidates']]})
        products = None
    else:
        if not cfg.exclusion_manifest_path and len(cfg.prior_prepared_paths or []) != 4:
            raise ValueError('Exclude all four prior prepared pools, including unselected reservations')
        groups, products, sources = load_groups(cfg)
        if cfg.exclusion_manifest_path:
            from .publication import exclusions
            exclusion_hashes, prior = exclusions(cfg, sources, 4)
        for name in cfg.prior_prepared_paths or []:
            path = Path(name)
            old = read_jsonl(path)
            meta = read_json(path.parent / 'prepare_metadata.json')
            rank = read_json(path.parent / 'rerank_metadata.json')
            if meta.get('mock') or digest(old) != meta['prepared_fingerprint'] or sources != meta['sources']:
                raise ValueError('Prior data fingerprint/version mismatch')
            if rank['identity']['revision'] != cfg.reranker_revision or root == path.parent:
                raise ValueError('Prior reranker differs or output directory overlaps')
            excluded.update(normalize_query(g['query']) for g in old)
            prior.append({'path': str(path), 'n': len(old), 'fingerprint': digest(old)})
    rows = partition(groups, cfg, excluded, exclusion_hashes)
    if not mock:
        from .publication import check_splits
        check_splits(rows, cfg)
    if products is not None:
        wanted = {c['id'] for g in rows for c in g['candidates']}
        lookup = products[products.product_id.isin(wanted)].set_index('product_id').fillna('').to_dict('index')
        for g in rows:
            for c in g['candidates']:
                p = lookup[c['id']]
                c.update(title=str(p['product_title']), brand=str(p['product_brand']) or None,
                         color=str(p['product_color']) or None, bullet_points=str(p['product_bullet_point']))
    for g in rows:
        g['mock'] = mock
    write_jsonl(root / 'prepared.jsonl', rows)
    metadata(cfg, mock, 'prepare', {'sources': sources, 'prior_artifacts': prior, 'excluded_queries': len(excluded) + len(exclusion_hashes),
                                  'prepared_fingerprint': digest(rows), 'counts': dict(Counter(g['split'] for g in rows))})
    print(f'V5 prepared {dict(Counter(g["split"] for g in rows))}; excluded={len(excluded) + len(exclusion_hashes)}', flush=True)


def rerank(cfg, mock=False):
    root = cfg.root(mock)
    if (root / 'protocol_frozen.json').exists():
        raise ValueError('Protocol already frozen; cannot rerank')
    rows = read_jsonl(root / 'prepared.jsonl')
    ranker = MockReranker() if mock else TransformerReranker(cfg)
    result = []
    for i, g in enumerate(rows):
        pairs = [(g['query'], candidate_text(c, cfg)) for c in g['candidates']]
        key = digest({'pairs': pairs, 'identity': ranker.identity})
        path = root / 'rerank_cache' / f'{key}.json'
        values = read_json(path)['scores'] if path.exists() else ranker.score(pairs)
        if len(values) != len(g['candidates']) or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid reranker scores')
        if not path.exists():
            write_json(path, {'scores': values, 'identity': ranker.identity})
        ranked = sorted(({**c, 'ce_score': float(v)} for c, v in zip(g['candidates'], values)),
                        key=lambda c: (-c['ce_score'], c['id']))[:5]
        result.append({**g, 'candidates': ranked})
        if (i + 1) % 100 == 0:
            print(f'V5 reranked {i+1}/{len(rows)}', flush=True)
    write_jsonl(root / 'ranked.jsonl', result)
    # No test label/quality summary before calibration threshold is frozen.
    metadata(cfg, mock, 'rerank', {'identity': ranker.identity, 'effective_batch_size': ranker.batch_size,
                                 'prepared_fingerprint': digest(rows), 'ranked_fingerprint': digest(result),
                                 'counts': dict(Counter(g['split'] for g in result))})


def checked(cfg, mock):
    root = cfg.root(mock)
    f = read_json(root / 'protocol_frozen.json')
    groups = read_jsonl(root / 'ranked.jsonl')
    if (f['source_hashes'] != source_hashes() or f['config'] != cfg.dict()
            or f['ranked_fingerprint'] != digest(groups)
            or f['prepared_fingerprint'] != digest(read_jsonl(root / 'prepared.jsonl'))):
        raise ValueError('Frozen source/config/data changed')
    if len({normalize_query(g['query']) for g in groups}) != len(groups) or len({g['query_id'] for g in groups}) != len(groups):
        raise ValueError('Duplicate queries')
    if Counter(g['split'] for g in groups) != Counter(calibration=cfg.calibration_size, test=cfg.test_size):
        raise ValueError('Fixed 800/2000 protocol sample counts changed')
    if any(g.get('mock') != mock or g.get('experiment_protocol') != PROTOCOL
           or g.get('task') != topk.TASK or len(g['candidates']) != 5 for g in groups):
        raise ValueError('Mixed protocol/mode/candidate count')
    return groups


def preflight(cfg, mock=False):
    import httpx
    root = cfg.root(mock)
    if (root / 'protocol_frozen.json').exists():
        checked(cfg, mock)
        return
    groups = read_jsonl(root / 'ranked.jsonl')
    if Counter(g['split'] for g in groups) != Counter(calibration=cfg.calibration_size, test=cfg.test_size):
        raise ValueError('Fixed sample size not met')
    prices, account, estimates = {}, None, {}
    for g in groups:
        if topk.jev_body(g, cfg)['state'] != json.loads(topk.gpt_body(g, cfg)['messages'][1]['content']):
            raise ValueError('Models must receive identical evidence')
    if not mock:
        key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        if not key:
            raise ValueError('Missing OPENROUTER_API_KEY')
        with httpx.Client(timeout=60) as client:
            r = client.get('https://openrouter.ai/api/v1/models');r.raise_for_status()
            catalog = {m['id']: m for m in r.json()['data']}
            for m, model in [('jev', cfg.jev_model), ('gpt', cfg.gpt_model)]:
                q = catalog_quote(client, catalog, m, model)
                expected_in = cfg.input_price_per_million if m == 'jev' else cfg.gpt_input_price_per_million
                expected_out = 0 if m == 'jev' else cfg.gpt_output_price_per_million
                if not math.isclose(float(q['pricing']['prompt']) * 1e6, expected_in, abs_tol=1e-9) or not math.isclose(float(q['pricing']['completion']) * 1e6, expected_out, abs_tol=1e-9):
                    raise ValueError('Frozen model price changed; stop without substituting model')
                prices[m] = q
            r = client.get('https://openrouter.ai/api/v1/credits', headers={'Authorization': 'Bearer ' + key})
            r.raise_for_status();d = r.json()['data']
            account = {k: d[k] for k in ('total_credits', 'total_usage')}
        if account['total_credits'] - account['total_usage'] < cfg.api_budget_usd:
            raise ValueError('Balance below full experiment cap; recharge, never shrink sample size')
        if cfg.cost_profile_path:
            from .publication import estimate_from_profile
            estimates = estimate_from_profile(cfg, groups)
        else:
            oldroot = Path(cfg.prior_prepared_paths[-1]).parent
            oldcfg = Config(**read_json(oldroot / 'protocol_frozen.json')['config'])
            oldgroups = {g['query_id']: g for g in read_jsonl(oldroot / 'ranked.jsonl')}
            for m, builder in [('jev', topk.jev_body), ('gpt', topk.gpt_body)]:
                records = [r for r in read_jsonl(oldroot / f'{m}.jsonl') if r['status'] == 'success' and not r['mock']]
                if not records or any(r['usage'].get('reported_cost_usd') is None for r in records):
                    raise ValueError('Previous billed cost unavailable; cannot authorize batch')
                oldbytes = sum(len(json.dumps(builder(oldgroups[r['query_id']], oldcfg), ensure_ascii=False).encode()) for r in records)
                newbytes = sum(len(json.dumps(builder(g, cfg), ensure_ascii=False).encode()) for g in groups)
                estimates[m] = sum(r['known_attempt_cost_usd'] for r in records) * newbytes / oldbytes
        forecast = sum(estimates.values()) * 1.25
        if forecast > cfg.preflight_max_estimate_usd:
            raise ValueError(f'Full-size forecast with 25% margin ${forecast:.4f} exceeds budget; recharge/review cap, never reduce samples')
    else:
        forecast = 0.
    meta = metadata(cfg, mock, 'preflight', {'account_before': account, 'prices': prices,
                                          'estimated_total_with_25pct_margin': forecast, 'per_model_estimates': estimates})
    write_json(root / 'protocol_frozen.json', {
        'task': topk.TASK, 'experiment_protocol': PROTOCOL, 'mock': mock, 'config': cfg.dict(),
        'source_hashes': source_hashes(), 'prepared_fingerprint': digest(read_jsonl(root / 'prepared.jsonl')),
        'ranked_fingerprint': digest(groups), 'created_utc': meta['created_utc'], 'account_before': account, 'prices': prices,
        'selection_rule': 'Calibration only: retention >= GPT retention - 0.01; maximize kept precision; tie: larger threshold; no feasible point means stop test.',
        'grid': cfg.core_threshold_grid, 'fixed_accessory_threshold': .5, 'fixed_constraint': 'NONE',
        'primary_test': 'Exact-retention delta one-sided 95% lower > -0.02 AND precision delta two-sided 95% lower > 0.',
        'notes': ['No test/API outcomes before threshold freeze; no fusion; no prompt tuning.',
                  'Natural calibration from official train; natural test from official test; source shift may affect matching.',
                  'Bootstrap is conditional on selected threshold; does not integrate calibration uncertainty.']})
    source = root / 'source';source.mkdir(parents=True, exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p, source / p.name)
    checked(cfg, mock)
    print(f'V5 protocol frozen; full-size forecast including 25% margin ${forecast:.4f}', flush=True)


def apply_core_threshold(prediction, threshold):
    if not 0 < threshold < 1:
        raise ValueError('Core threshold outside (0,1)')
    rows = []
    for d in prediction['candidate_decisions']:
        p = d.get('core_probability')
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Missing valid native core probability')
        rows.append({**d, 'is_core_intent_matched': p >= threshold})
    return topk.adapt(rows, [d['id'] for d in rows], prediction['reasoning'], prediction['reasoning_source'])


def prediction_index(cfg, mock, groups, splits):
    root = cfg.root(mock)
    index = {}
    for m, builder, endpoint in [('jev', topk.jev_body, ENDPOINT), ('gpt', topk.gpt_body, GPT_ENDPOINT)]:
        rows = read_jsonl(root / f'{m}.jsonl')
        idx = {r['query_id']: r for r in rows}
        if len(idx) != len(rows):
            raise ValueError('Duplicate predictions')
        for g in (g for g in groups if g['split'] in splits):
            r = idx.get(g['query_id'])
            if not r or r['experiment_protocol'] != PROTOCOL or r['mock'] != mock or r['split'] != g['split']:
                raise ValueError('Incomplete/mixed predictions')
            if r['cache_key'] != cache_key(builder(g, cfg), mock, endpoint):
                raise ValueError('Prediction request mismatch')
            if r['status'] == 'success' and not mock:
                parsed, usage = topk.decoder(g, m)(r['response'], cfg.jev_model if m == 'jev' else cfg.gpt_model)
                if parsed != r['prediction'] or usage != r['usage']:
                    raise ValueError('Stored prediction/probability differs from raw API response')
        if len({r['model'] for r in rows if r['status'] == 'success'}) > 1:
            raise ValueError('Multiple resolved model builds')
        index[m] = idx
    return index


def select_threshold(groups, index, cfg):
    if not groups or any(g['split'] != 'calibration' for g in groups):
        raise ValueError('Threshold fitting must use calibration only')
    successful = [g for g in groups if all(index[m][g['query_id']]['status'] == 'success' for m in METHODS)]
    if len(successful) / len(groups) < .99:
        raise ValueError('Calibration common-success coverage <99%; no fitting')
    target = topk.aggregate([topk.query_counts(g, index['gpt'][g['query_id']]['prediction']) for g in successful])['exact_retention']
    if target is None:
        raise ValueError('Calibration lacks any E; no retention target')
    rows = []
    for t in cfg.core_threshold_grid:
        v = topk.aggregate([topk.query_counts(g, apply_core_threshold(index['jev'][g['query_id']]['prediction'], t)) for g in successful])
        rows.append({'threshold': t, 'kept_precision': v['kept_precision'], 'exact_retention': v['exact_retention'],
                     'query_coverage': v['query_coverage'], 'query_contamination_risk': v['query_contamination_risk'],
                     'feasible': v['exact_retention'] is not None and v['exact_retention'] >= target - cfg.calibration_retention_slack})
    valid = [r for r in rows if r['feasible'] and r['kept_precision'] is not None]
    chosen = max(valid, key=lambda r: (r['kept_precision'], r['threshold'])) if valid else None
    return {'threshold': chosen['threshold'] if chosen else None, 'usable': chosen is not None,
            'gpt_calibration_retention': target, 'minimum_calibration_retention': target - cfg.calibration_retention_slack,
            'calibration_n': len(groups), 'common_n': len(successful), 'grid_results': rows,
            'note': 'No feasible grid point: do not change prompts, grid, seed, samples or fixed accessory/constraint rules.'}


def freeze_thresholds(cfg, mock=False):
    root = cfg.root(mock);groups = checked(cfg, mock)
    if any((root / f'{m}.jsonl').exists() and any(r['split'] == 'test' for r in read_jsonl(root / f'{m}.jsonl')) for m in METHODS):
        # Re-reading the already frozen file is permitted, never re-selection after test access.
        if (root / 'threshold_frozen.json').exists():
            return frozen_threshold(cfg, mock)
        raise ValueError('Test predictions already accessed; threshold selection forbidden')
    calibration = [g for g in groups if g['split'] == 'calibration']
    index = prediction_index(cfg, mock, groups, {'calibration'})
    result = select_threshold(calibration, index, cfg)
    data = {**result, 'protocol_fingerprint': digest(read_json(root / 'protocol_frozen.json')),
            'calibration_receipt_fingerprint': digest({m: [index[m][g['query_id']] for g in calibration] for m in METHODS})}
    path = root / 'threshold_frozen.json'
    if path.exists():
        old = read_json(path)
        if {k: v for k, v in old.items() if k != 'created_utc'} != data:
            raise ValueError('Frozen threshold changed')
        return old
    data['created_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(path, data)
    write_csv(root / 'calibration_grid.csv', result['grid_results'],
              ['threshold', 'kept_precision', 'exact_retention', 'query_coverage', 'query_contamination_risk', 'feasible'])
    print(f'V5 threshold frozen: {result["threshold"]}; usable={result["usable"]}', flush=True)
    return data


def frozen_threshold(cfg, mock):
    root = cfg.root(mock);groups = checked(cfg, mock)
    if not (root / 'threshold_frozen.json').exists():
        raise ValueError('Freeze calibration threshold before any test API calls')
    f = read_json(root / 'threshold_frozen.json')
    if not f['usable']:
        raise ValueError('No calibration threshold matches retention; test phase stopped, not retuned')
    if f['threshold'] not in cfg.core_threshold_grid or f['protocol_fingerprint'] != digest(read_json(root / 'protocol_frozen.json')):
        raise ValueError('Threshold/protocol fingerprint mismatch')
    index = prediction_index(cfg, mock, groups, {'calibration'})
    calibration = [g for g in groups if g['split'] == 'calibration']
    receipts = {m: [index[m][g['query_id']] for g in calibration] for m in METHODS}
    if f['calibration_receipt_fingerprint'] != digest(receipts):
        raise ValueError('Frozen calibration receipts changed')
    # Reject hand-edited threshold files even if a chosen value happens to lie on the grid.
    result = select_threshold(calibration, index, cfg)
    if any(f[k] != v for k, v in result.items()):
        raise ValueError('Frozen threshold is inconsistent with pre-registered calibration rule')
    return f


def mock_prediction(g, method):
    rows = []
    for c in g['candidates']:
        positive = 'requested model' in c['title']
        p = .8 if positive else .15
        rows.append({'id': c['id'], 'is_core_intent_matched': positive,
                     'is_accessory_trap': False, 'primary_constraint': 'NONE',
                     **({'core_probability': p, 'accessory_probability': .1} if method == 'jev' else {})})
    return topk.adapt(rows, [r['id'] for r in rows], 'SIMULATED', 'simulated')


async def run_async(cfg, mock=False, splits=None, transport=None):
    import httpx
    groups = checked(cfg, mock);root = cfg.root(mock)
    if splits not in (['calibration'], ['test']):
        raise ValueError('Use api-all --splits calibration, freeze-thresholds, then api-all --splits test')
    if splits == ['test']:
        frozen_threshold(cfg, mock)
    elif (root / 'threshold_frozen.json').exists():
        raise ValueError('Calibration frozen; cannot make further calibration calls')
    selected = [g for g in groups if g['split'] in splits]
    lock = root / 'api-all.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError('API lock exists; verify previous process stopped before resuming') from None
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    tasks = []
    try:
        key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        if not mock and not key:
            raise ValueError('Missing OPENROUTER_API_KEY')
        budget = None if mock else Budget(root / 'budget.json', cfg)
        async def model_run(m):
            builder = topk.jev_body if m == 'jev' else topk.gpt_body
            endpoint = ENDPOINT if m == 'jev' else GPT_ENDPOINT
            path = root / f'{m}.jsonl'
            old = [r for r in read_jsonl(path) if r['split'] not in splits] if path.exists() else []
            queue = asyncio.Queue();results = [];failures = 0
            for g in selected:
                queue.put_nowait(g)
            limiter = RateLimiter(cfg.requests_per_second)
            async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + key} if not mock else {},
                                         timeout=cfg.request_timeout_s, transport=transport) as client:
                async def worker():
                    nonlocal failures
                    while not queue.empty():
                        g = queue.get_nowait();body = builder(g, cfg);hashed = cache_key(body, mock, endpoint)
                        cache = root / f'{m}_cache' / f'{hashed}.json'
                        previous = read_json(cache) if cache.exists() else None
                        hit = previous is not None and previous['status'] == 'success'
                        if hit:
                            r = previous
                        elif mock:
                            r = {'status': 'success', 'prediction': mock_prediction(g, m), 'usage': {},
                                 'model': 'SIMULATED-' + (cfg.jev_model if m == 'jev' else cfg.gpt_model),
                                 'attempts': [], 'total_elapsed_ms': None, 'known_attempt_cost_usd': 0,
                                 'cost_may_exclude_failed_attempts': False}
                        else:
                            r = await call_api(client, body, cfg, limiter, method=m, budget=budget, parse_fn=topk.decoder(g, m))
                        r = {**r, 'query_id': g['query_id'], 'split': g['split'], 'task': topk.TASK,
                             'experiment_protocol': PROTOCOL, 'mock': mock, 'cache_key': hashed, 'cache_hit': hit,
                             'prompt_hash': digest(body.get('questions', body.get('messages')))}
                        if not hit:
                            write_json(cache, r)
                        results.append(r)
                        with (root / f'{m}_journal.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
                        queue.task_done();failures = failures + 1 if r['status'] != 'success' else 0
                        if failures >= 3:
                            raise ValueError('Three consecutive API/schema failures; inspect before more spending')
                        if len(results) % 50 == 0:
                            print(f'V5 {m} {splits[0]} {len(results)}/{len(selected)}', flush=True)
                workers = [asyncio.create_task(worker()) for _ in range(cfg.concurrency)]
                try:
                    await asyncio.gather(*workers)
                finally:
                    for w in workers:
                        if not w.done():
                            w.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
                    write_jsonl(path, sorted(old + results, key=lambda r: r['query_id']))
                    metadata(cfg, mock, m, {'completed': len(old) + len(results), 'splits': splits,
                                           'expected': len(old) + len(selected), 'ranked_fingerprint': digest(groups)})
        tasks = [asyncio.create_task(model_run(m)) for m in METHODS]
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        lock.unlink(missing_ok=True)


def run(cfg, mock=False, splits=None):
    asyncio.run(run_async(cfg, mock, splits))


def bootstrap(a, b, samples, seed):
    # Calculate ratio metrics from query-level count vectors, never resample item rows.
    if not a or len(a) != len(b):
        raise ValueError('Unpaired/empty bootstrap')
    keys = sorted(a[0]);aa = np.array([[c[k] for k in keys] for c in a]);bb = np.array([[c[k] for k in keys] for c in b])
    if aa.shape != bb.shape or not len(a):
        raise ValueError('Unpaired/empty bootstrap')
    pos = {k: i for i, k in enumerate(keys)}
    def values(x):
        tp, fp, fn = (x[pos[k]] for k in ('tp', 'fp', 'fn'))
        def ratio(num, den):
            return num / den if den else np.nan
        return {'kept_precision': ratio(tp, tp + fp), 'exact_retention': ratio(tp, tp + fn),
                'candidate_f1': ratio(2 * tp, 2 * tp + fp + fn), 'query_coverage': x[pos['passed']] / len(a),
                'query_contamination_risk': ratio(x[pos['polluted']], x[pos['passed']]),
                'safe_route_success': x[pos['safe_route_success']] / len(a),
                'rescue_success_rate': ratio(x[pos['rescue_success']], x[pos['rescue_eligible']])}
    rng = np.random.default_rng(seed)
    individual = {m: {k: [] for k in topk.BOOT_METRICS} for m in METHODS};delta = {k: [] for k in topk.BOOT_METRICS}
    for _ in range(samples):
        ii = rng.integers(0, len(a), len(a));av = values(aa[ii].sum(axis=0));bv = values(bb[ii].sum(axis=0))
        for k in topk.BOOT_METRICS:
            individual['jev'][k].append(av[k]);individual['gpt'][k].append(bv[k]);delta[k].append(av[k] - bv[k])
    def quantile(v, q):
        v = np.asarray(v);v = v[np.isfinite(v)]
        return np.quantile(v, q).tolist() if len(v) else None
    return {'delta_ci95': {k: quantile(v, [.025, .975]) for k, v in delta.items()},
            'individual_ci95': {m: {k: quantile(v, [.025, .975]) for k, v in d.items()} for m, d in individual.items()},
            'retention_delta_one_sided95_lower': quantile(delta['exact_retention'], .05),
            'samples': samples, 'unit': 'paired query clusters', 'conditional_on_frozen_threshold': True}


def decisions(boot, cfg, complete=True):
    lower = boot['retention_delta_one_sided95_lower'];pci = boot['delta_ci95']['kept_precision']
    ni = lower is not None and lower > -cfg.retention_noninferiority_margin
    superior = pci is not None and pci[0] > 0
    return {'retention_noninferiority': bool(ni), 'precision_superiority': bool(superior),
            'joint_quality_evidence': bool(complete and ni and superior),
            'retention_margin': cfg.retention_noninferiority_margin,
            'note': 'Primary conjunctive claim requires both tests; CI crossing zero does not establish equivalence.'}


def evaluate(cfg, mock=False):
    root = cfg.root(mock);frozen = frozen_threshold(cfg, mock);groups = checked(cfg, mock)
    idx = prediction_index(cfg, mock, groups, {'calibration', 'test'})
    natural = [g for g in groups if g['split'] == 'test']
    common = [g for g in natural if all(idx[m][g['query_id']]['status'] == 'success' for m in METHODS)]
    if not common:
        raise ValueError('No common test successes')
    counts, metrics = {}, {}
    for m in METHODS:
        counts[m] = [topk.query_counts(g, apply_core_threshold(idx[m][g['query_id']]['prediction'], frozen['threshold'])
                                      if m == 'jev' else idx[m][g['query_id']]['prediction']) for g in common]
        metrics[m] = topk.aggregate(counts[m])
    boot = bootstrap(counts['jev'], counts['gpt'], cfg.bootstrap_samples, cfg.seed)
    complete = len(common) / len(natural) >= .99 and sum(any(c['label'] != 'E' for c in g['candidates']) for g in common) >= 50
    costs, latency, models = {}, {}, {}
    for m in METHODS:
        journal = read_jsonl(root / f'{m}_journal.jsonl')
        network = [r for r in journal if not r['cache_hit']]
        first_success = {}
        for r in network:
            if r['status'] == 'success':
                first_success.setdefault(r['query_id'], r)
        testrows = [first_success[g['query_id']] for g in common if g['query_id'] in first_success]
        costs[m] = {'logical_queries': len(idx[m]), 'known_cost_usd': sum(r['known_attempt_cost_usd'] for r in network),
                    'recorded_http_attempts': sum(len(r['attempts']) for r in network),
                    'input_tokens': sum(r['usage'].get('input_tokens') or 0 for r in network),
                    'output_tokens': sum(r['usage'].get('output_tokens') or 0 for r in network),
                    'unknown_attempt_cost': any(r['cost_may_exclude_failed_attempts'] for r in network)}
        latency[m] = {'natural_query_n': len(testrows), 'http': percentile_summary([a['elapsed_ms'] for r in testrows for a in r['attempts']]),
                      'gate': percentile_summary([r['total_elapsed_ms'] for r in testrows if r['total_elapsed_ms'] is not None])}
        models[m] = {'requested': cfg.jev_model if m == 'jev' else cfg.gpt_model,
                     'resolved': sorted({r['model'] for r in idx[m].values() if r['status'] == 'success'}),
                     'providers': sorted({r['provider'] for r in idx[m].values() if r.get('provider')})}
    report = {'experiment_protocol': PROTOCOL, 'mock': mock, 'threshold': frozen['threshold'], 'metrics': metrics,
              'bootstrap': boot, 'conclusions': decisions(boot, cfg, complete), 'costs': costs, 'latency': latency, 'models': models,
              'completeness': {'natural_n': len(natural), 'common_n': len(common), 'common_fraction': len(common) / len(natural),
                               'failed': {m: sum(idx[m][g['query_id']]['status'] != 'success' for g in natural) for m in METHODS}},
              'budget': read_json(root / 'budget.json') if not mock else None,
              'threshold_fingerprint': digest(frozen), 'protocol_fingerprint': digest(read_json(root / 'protocol_frozen.json')),
              'notes': ['ESCI public proxy, not original production prompt; no independently gold-labeled diagnostic fields.',
                        'Only core threshold calibrated; accessory <0.5 and constraint NONE remain fixed. GPT is unchanged.',
                        'Confidence intervals conditional on frozen threshold; test never used for selection.',
                        'Latency: real non-cache network calls; Gate includes pacing/retries, excludes queue wait, shared reranking and manual resume downtime.',
                        'Natural calibration/train and test/test source distributions can differ. No 5% risk certification.']}
    dest = root / 'report';write_json(dest / 'metrics.json', report)
    write_csv(dest / 'summary.csv', [{'method': m, **metrics[m]} for m in METHODS],
              ['method', 'n', 'candidate_n', *topk.BOOT_METRICS, 'route_action_accuracy', 'unnecessary_fallback_rate'])
    queryrows, itemrows, errors = [], [], []
    for g in common:
        for m in METHODS:
            raw = idx[m][g['query_id']]['prediction']
            p = apply_core_threshold(raw, frozen['threshold']) if m == 'jev' else raw
            c = topk.query_counts(g, p)
            queryrows.append({'query_id': g['query_id'], 'method': m, 'selected_ids': json.dumps(p['selected_ids']),
                              'routing_action': p['routing_action'], **c})
            for item, d in zip(g['candidates'], p['candidate_decisions']):
                row = {'query_id': g['query_id'], 'method': m, 'query': g['query'], 'product_id': item['id'], 'title': item['title'],
                       'label': item['label'], 'keep': d['keep'], 'core_probability': d.get('core_probability'),
                       'accessory_probability': d.get('accessory_probability'), 'primary_constraint': d['primary_constraint']}
                itemrows.append(row)
                if d['keep'] != (item['label'] == 'E'):
                    errors.append(row)
    write_csv(dest / 'query_predictions.csv', queryrows,
              ['query_id', 'method', 'selected_ids', 'routing_action', 'tp', 'fp', 'fn', 'tn', 'passed', 'polluted', 'safe_route_success', 'rescue_eligible', 'rescue_success'])
    cols = ['query_id', 'method', 'query', 'product_id', 'title', 'label', 'keep', 'core_probability', 'accessory_probability', 'primary_constraint']
    write_csv(dest / 'candidate_predictions.csv', itemrows, cols);write_csv(dest / 'errors.csv', errors, cols)
    sample = random.Random(cfg.seed).sample(common, min(100, len(common)))
    write_csv(dest / 'manual_audit.csv', [{'query_id': g['query_id'], 'query': g['query'], 'product_id': c['id'],
              'title': c['title'], 'official_label': c['label'], 'human_checked': ''} for g in sample for c in g['candidates']],
              ['query_id', 'query', 'product_id', 'title', 'official_label', 'human_checked'])
    topk.plots(dest, report, mock)
    metadata(cfg, mock, 'evaluate', {'threshold_fingerprint': digest(frozen), 'ranked_fingerprint': digest(groups)})
    print(f'V5 report: {dest}; conclusions={report["conclusions"]}', flush=True)
    return report
