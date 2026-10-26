from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import httpx
import pytest
from jev_gate import calibrated, cli
from jev_gate.config import Config
from jev_gate.data import FILES, download
from jev_gate.io import digest, read_json, write_json
from jev_gate.publication import check_splits, exclusions, load_manifest
from jev_gate.published_results import verify
from test_topk import group

ROOT = Path(__file__).resolve().parents[1]


def refresh(root):
    write_json(root/'checksums.json', {'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.iterdir()) if p.name != 'checksums.json'}})


@pytest.mark.parametrize('experiment', ['v4', 'v5'])
def test_published_results_verify_without_network(experiment, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    result = verify(experiment)
    assert result['verified'] and result['network_requests'] == 0
    assert result['natural_queries'] == (2000 if experiment == 'v5' else 500)
    assert result['all_seven_delta_intervals_match']


def test_verify_cli_does_not_load_credentials_or_configuration(monkeypatch, capsys):
    def forbidden(*a, **kw):
        raise AssertionError('Offline verification must not load credentials/configuration')
    monkeypatch.setattr(cli, 'load_env', forbidden)
    monkeypatch.setattr(Config, 'load', forbidden)
    cli.main(['verify-results', '--experiment', 'v5'])
    assert json.loads(capsys.readouterr().out)['verified']


def test_artifact_corruption_is_rejected(tmp_path):
    root = tmp_path/'artifacts';shutil.copytree(ROOT/'benchmarks/v5', root)
    with (root/'candidates.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify('v5', root)


def test_rehashed_threshold_edit_still_cannot_change_selection(tmp_path):
    root = tmp_path/'artifacts';shutil.copytree(ROOT/'benchmarks/v5', root)
    value = read_json(root/'threshold.json');value['threshold'] = .05
    write_json(root/'threshold.json', value);refresh(root)
    with pytest.raises(ValueError, match='Frozen calibration threshold mismatch'):
        verify('v5', root)


def test_rehashed_metric_edit_fails_independent_counts(tmp_path):
    root = tmp_path/'artifacts';shutil.copytree(ROOT/'benchmarks/v5', root)
    value = read_json(root/'metrics.json');value['metrics']['jev']['kept_precision'] = .999
    write_json(root/'metrics.json', value);refresh(root)
    with pytest.raises(ValueError, match='Test metric mismatch'):
        verify('v5', root)


def test_public_exclusion_manifests_cover_all_prior_reservations():
    cfg = Config.load(str(ROOT/'config.topk-calibrated-v5.toml'))
    source = read_json(ROOT/'manifests/dataset.json')['files']
    values, prior = exclusions(cfg, source, 4)
    assert len(values) == 25680 and len(prior) == 4
    with pytest.raises(ValueError, match='identity mismatch'):
        exclusions(cfg, {}, 4)
    with pytest.raises(ValueError, match='identity mismatch'):
        exclusions(cfg, source, 3)
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        load_manifest(cfg.exclusion_manifest_path, '0'*64)


def test_hashed_exclusions_preserve_original_partition(cfg):
    rows = []
    for i in range(30):
        rows.append({**group(), 'query_id': str(i), 'query': f'query {i}', 'source_split': 'train' if i<15 else 'test'})
    cfg = replace(cfg, calibration_size=5, test_size=5)
    old = {'query 0', 'query 20'}
    assert calibrated.partition(rows, cfg, old) == calibrated.partition(rows, cfg, exclusion_hashes={digest(q) for q in old})


def test_fixed_split_membership_rejects_candidate_pool_changes(cfg, tmp_path):
    row = {**group(), 'source_split': 'test'}
    manifest = {'seed': cfg.seed, 'dataset': {'revision': 'a'*40}, 'rows': [
        {'query_id': row['query_id'], 'query_hash': digest(row['query']), 'split': row['split'],
         'source_split': row['source_split'],
         'candidate_pool_fingerprint': digest([{'id': c['id'], 'label': c['label']} for c in row['candidates']])}]}
    path = tmp_path/'splits.json';write_json(path, manifest)
    cfg = replace(cfg, dataset_revision='a'*40, split_manifest_path=str(path), split_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    check_splits([row], cfg)
    row['candidates'][0]['label'] = 'I'
    with pytest.raises(ValueError, match='candidate pools differ'):
        check_splits([row], cfg)


def test_final_config_can_dispatch_data_download(monkeypatch):
    calls=[]
    import jev_gate.data
    monkeypatch.setattr(jev_gate.data, 'download', lambda cfg: calls.append(cfg.dataset_revision))
    cli.main(['download-data', '--config', str(ROOT/'config.topk-calibrated-v5.toml')])
    assert calls == ['7916cdf6ab75a462e77f20ab40428a10923998d5']


def test_pinned_download_rejects_existing_wrong_commit(cfg, monkeypatch):
    monkeypatch.setattr(httpx, 'Client', lambda **kw: nullcontext(object()))
    cfg = replace(cfg, dataset_revision='a'*40)
    write_json(Path(cfg.data_dir)/'download_manifest.json', {'revision': 'b'*40, 'files': {}})
    with pytest.raises(ValueError, match='pinned dataset commit'):
        download(cfg)


def test_pinned_download_rejects_changed_local_data(cfg, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    monkeypatch.setattr(httpx, 'Client', lambda **kw: nullcontext(object()))
    Path(cfg.data_dir).mkdir(parents=True)
    for name in FILES:
        pq.write_table(pa.table({'id': [1]}), Path(cfg.data_dir)/name)
    cfg = replace(cfg, dataset_revision='a'*40, dataset_files_sha256={name: '0'*64 for name in FILES})
    with pytest.raises(ValueError, match='pinned published SHA-256'):
        download(cfg)


def test_release_jsonl_preserves_unicode_line_separators():
    from jev_gate.published_results import _jsonl_bytes
    rows = [{'title': 'one\u0085two\u2028three\u2029four'}, {'title': 'another product'}]
    data = ('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n').encode('utf-8')
    assert list(_jsonl_bytes(data)) == rows
