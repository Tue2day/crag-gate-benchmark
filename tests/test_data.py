import json
from pathlib import Path
import pandas as pd
import pytest

from jev_gate.data import (FILES, normalize_query, prepare, split_groups,
                           synthetic_groups, validate_parquet)
from jev_gate.io import read_jsonl
from jev_gate.rerank import rerank


def test_normalization_and_cross_split_dedup(cfg):
    assert normalize_query("  ＰＨＯＮＥ\t Case ") == "phone case"
    groups = synthetic_groups()
    duplicate = dict(groups[0], query_id="duplicate", source_split="test",
                     query="  "+groups[0]["query"].upper()+"  ")
    rows = split_groups(groups+[duplicate], cfg, mock=True)
    texts = [normalize_query(g["query"]) for g in rows]
    assert len(texts) == len(set(texts))
    assert "duplicate" not in [g["query_id"] for g in rows]
    assert rows == split_groups(groups+[duplicate], cfg, mock=True)


def test_top5_label_and_challenge_separation(cfg):
    prepare(cfg, True)
    rerank(cfg, True)
    rows = read_jsonl(cfg.root(True)/"ranked.jsonl")
    assert all(g["y"] == int(any(c["label"] == "E" for c in g["candidates"])) for g in rows)
    challenge = [g for g in rows if g["split"] == "challenge"]
    assert challenge
    assert all(g["y"] == 0 and any(c["label"] == "C" for c in g["candidates"]) for g in challenge)
    assert not {g["query_id"] for g in challenge} & {g["query_id"] for g in rows if g["split"] == "test"}


def test_parquet_validation(tmp_path):
    path = tmp_path/"test.parquet"
    path.write_text("version https://git-lfs.github.com/spec/v1\n", "utf-8")
    with pytest.raises(ValueError, match="LFS pointer"):
        validate_parquet(path)
    path.write_bytes(b"PAR1broken file")
    with pytest.raises(ValueError, match="truncated"):
        validate_parquet(path)
    pd.DataFrame({"x": [1]}).to_parquet(path)
    validate_parquet(path)


def test_real_parquet_prepare_without_download(cfg):
    root = Path(cfg.data_dir)
    root.mkdir()
    examples, products = [], []
    for i in range(14):
        for j in range(8):
            pid = f"p{i}-{j}"
            products.append({"product_id": pid, "product_locale": "us", "product_title": f"Product {pid}",
                             "product_brand": None, "product_color": None, "product_bullet_point": None})
            examples.append({"query_id": i, "query": f"Shopping query {i}", "product_id": pid,
                             "product_locale": "us", "esci_label": "C" if j == 0 else "I",
                             "split": "train" if i < 6 else "test"})
    pd.DataFrame(examples).to_parquet(root/FILES[0])
    pd.DataFrame(products).to_parquet(root/FILES[1])
    prepare(cfg)
    rows = read_jsonl(cfg.root()/"prepared.jsonl")
    assert sum(g["split"] == "development" for g in rows) == 2
    assert sum(g["split"] == "test" for g in rows) == 3
    assert all(c["brand"] is None for g in rows for c in g["candidates"])
    assert all(not g["mock"] for g in rows)


def test_fixed_version_config(cfg, tmp_path):
    cfg.jev_model = "jev-latest"
    with pytest.raises(ValueError, match="fixed"):
        cfg.validate()
    path = tmp_path/"bad.toml"
    path.write_text('sead = 42\n')
    from jev_gate.config import Config
    with pytest.raises(ValueError, match="Unknown"):
        Config.load(str(path))


def test_jsonl_preserves_unicode_line_separators_inside_product_text(tmp_path):
    from jev_gate.io import write_jsonl
    rows=[{'title':'a\u0085b\u2028c\u2029d','bullet':'first\nsecond'}, {'title':'next'}]
    path=tmp_path/'unicode.jsonl'
    write_jsonl(path,rows)
    assert read_jsonl(path)==rows
