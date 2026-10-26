"""Portable manifests for historical split reconstruction and conservative costing."""
from pathlib import Path
import json

from .data import file_digest, normalize_query
from .io import digest, read_json


def load_manifest(path, expected_sha256):
    p = Path(path)
    if not p.is_file() or file_digest(p) != expected_sha256:
        raise ValueError('Published manifest missing or SHA-256 mismatch')
    value = read_json(p)
    if not isinstance(value, dict):
        raise ValueError('Manifest must be a JSON object')
    return value


def exclusions(cfg, sources, prior_count):
    value = load_manifest(cfg.exclusion_manifest_path, cfg.exclusion_manifest_sha256)
    if (value['dataset']['files'] != sources
            or value['dataset']['revision'] != cfg.dataset_revision
            or value['reranker_revision'] != cfg.reranker_revision
            or len(value['prior_artifacts']) != prior_count):
        raise ValueError('Exclusion manifest dataset/reranker/prior-round identity mismatch')
    hashes = value['query_hashes']
    if hashes != sorted(set(hashes)) or any(len(h) != 64 for h in hashes):
        raise ValueError('Exclusion query hashes must be sorted, unique SHA-256 strings')
    return set(hashes), value['prior_artifacts']


def check_splits(rows, cfg):
    if not cfg.split_manifest_path:
        return
    manifest = load_manifest(cfg.split_manifest_path, cfg.split_manifest_sha256)
    if manifest['seed'] != cfg.seed or manifest['dataset']['revision'] != cfg.dataset_revision:
        raise ValueError('Fixed split manifest seed/data revision mismatch')
    actual = [{'query_id': g['query_id'], 'query_hash': digest(normalize_query(g['query'])),
               'source_split': g['source_split'], 'split': g['split'],
               'candidate_pool_fingerprint': digest([{'id': c['id'], 'label': c['label']} for c in g['candidates']])}
              for g in rows]
    if actual != manifest['rows']:
        raise ValueError('Prepared membership or judged candidate pools differ from published splits; no resampling')


def estimate_from_profile(cfg, groups):
    from . import topk
    value = load_manifest(cfg.cost_profile_path, cfg.cost_profile_sha256)
    if value['models'] != {'jev': cfg.jev_model, 'gpt': cfg.gpt_model}:
        raise ValueError('Cost profile uses different models')
    contract = digest({'rules': topk.RULES, 'core': topk.CORE, 'accessory': topk.ACCESSORY,
                       'constraints': topk.CONSTRAINTS, 'attribute': topk.ATTR})
    if value['prompt_contract_hash'] != contract:
        raise ValueError('Cost profile prompt contract changed')
    estimates = {}
    for method, builder in [('jev', topk.jev_body), ('gpt', topk.gpt_body)]:
        old = value['per_model'][method]
        if old['request_utf8_bytes'] <= 0 or old['known_cost_usd'] <= 0:
            raise ValueError('Historical usage profile lacks usable billing data')
        nbytes = sum(len(json.dumps(builder(g, cfg), ensure_ascii=False).encode()) for g in groups)
        estimates[method] = old['known_cost_usd'] * nbytes / old['request_utf8_bytes']
    return estimates
