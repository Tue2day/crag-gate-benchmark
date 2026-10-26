"""Only rerank judged ESCI pools. Never infer Irrelevant from missing judgments."""
from pathlib import Path
import hashlib
import random
import unicodedata

from .io import digest, metadata, read_json, write_json, write_jsonl

FILES = ("shopping_queries_dataset_examples.parquet", "shopping_queries_dataset_products.parquet")


def normalize_query(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_parquet(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Data file missing: {path}; use download-data or supply local files")
    with path.open("rb") as f:
        start = f.read(128)
        if start.startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(f"Git LFS pointer instead of actual data: {path}")
        if path.stat().st_size < 12:
            raise ValueError(f"Incomplete Parquet file: {path}")
        f.seek(-4, 2)
        if not start.startswith(b"PAR1") or f.read(4) != b"PAR1":
            raise ValueError(f"Invalid/truncated Parquet file: {path}")
    import pyarrow.parquet as pq
    try:
        pq.read_metadata(path)
    except Exception as exc:
        raise ValueError(f"Unreadable Parquet metadata: {path}") from exc


def download(cfg):
    import httpx
    root = Path(cfg.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download_manifest.json"
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        previous = read_json(manifest_path) if manifest_path.exists() else None
        if previous:
            revision = previous["revision"]
            if cfg.dataset_revision and revision != cfg.dataset_revision:
                raise ValueError('Existing download manifest differs from pinned dataset commit')
        elif cfg.dataset_revision:
            revision = cfg.dataset_revision
        else:
            r = client.get("https://api.github.com/repos/amazon-science/esci-data/commits/main")
            r.raise_for_status()
            revision = r.json()["sha"]
        files = {}
        for name in FILES:
            path = root / name
            if not path.exists():
                url = f"https://media.githubusercontent.com/media/amazon-science/esci-data/{revision}/shopping_queries_dataset/{name}"
                tmp = path.with_suffix(".download")
                try:
                    with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
                        response.raise_for_status()
                        expected = response.headers.get("content-length")
                        with tmp.open("wb") as f:
                            for chunk in response.iter_bytes():
                                f.write(chunk)
                        if expected and tmp.stat().st_size != int(expected):
                            raise ValueError(f"Incomplete download: {name}")
                    validate_parquet(tmp)
                    tmp.replace(path)
                finally:
                    tmp.unlink(missing_ok=True)
            validate_parquet(path)
            checksum = file_digest(path)
            if cfg.dataset_files_sha256 and cfg.dataset_files_sha256.get(name) != checksum:
                raise ValueError(f"Data differs from pinned published SHA-256: {name}")
            if previous and name in previous["files"] and previous["files"][name]["sha256"] != checksum:
                raise ValueError(f"Existing file differs from download manifest: {path}")
            files[name] = {"sha256": checksum, "bytes": path.stat().st_size}
            print(f"Validated {name}: {path.stat().st_size:,} bytes")
        write_json(manifest_path, {"repository": "amazon-science/esci-data",
                                   "revision": revision, "files": files})


def split_groups(groups, cfg, mock=False):
    """Partition by normalized text; a train query always excludes matching test text."""
    ordered = sorted(groups, key=lambda g: (g["source_split"] != "train", str(g["query_id"])))
    pools = {"train": [], "test": []}
    seen = set()
    for group in ordered:
        norm = normalize_query(group["query"])
        if norm and norm not in seen and len(group["candidates"]) >= 5:
            seen.add(norm)
            pools[group["source_split"]].append(group)
    rng = random.Random(cfg.seed)
    for pool in pools.values():
        rng.shuffle(pool)
    sizes = (60, 60, 80, 20) if mock else (
        cfg.development_size, cfg.validation_size, cfg.test_size, cfg.challenge_size)
    dev_n, val_n, test_n, challenge_n = sizes
    if len(pools["train"]) < dev_n + val_n or len(pools["test"]) < test_n:
        raise ValueError("Not enough unique queries with >=5 judged products for configured splits")
    result = []
    slices = [("development", pools["train"][:dev_n]),
              ("validation", pools["train"][dev_n:dev_n + val_n]),
              ("test", pools["test"][:test_n])]
    # Reserve before filtering: challenge queries can never also be natural test queries.
    reserved = pools["test"][test_n:test_n + cfg.challenge_pool_size]
    challenges = []
    for group in reserved:
        ci = [c for c in group["candidates"] if c["label"] in {"C", "I"}]
        if len(ci) >= 5 and any(c["label"] == "C" for c in ci):
            challenges.append({**group, "candidates": ci})
            if len(challenges) == challenge_n:
                break
    if challenge_n == 0:
        challenges = []
    slices.append(("challenge", challenges))
    for split, selected in slices:
        for group in selected:
            result.append({**group, "split": split, "mock": mock,
                           "constructed": split == "challenge"})
    return result


def synthetic_groups(seed=42):
    """Synthetic ground truth for plumbing tests, not benchmark evidence."""
    rng = random.Random(seed)
    groups = []
    for i in range(300):
        product_type = ("headphones", "phone", "camera", "laptop")[i % 4]
        # Both easy negatives and lexical traps; deterministic but non-perfect mock scores.
        has_exact = i % 5 != 0
        labels = (["E", "E", "S", "S"] if has_exact else ["C", "C", "S", "I"])
        labels += ["C", "C", "C", "I", "I", "I", "S", "I"]
        rng.shuffle(labels)
        candidates = []
        for j, label in enumerate(labels):
            suffix = {"E": "requested model", "S": "alternative model",
                      "C": "case accessory", "I": "unrelated stand"}[label]
            candidates.append({"id": f"p{i}-{j}", "title": f"{product_type} {suffix}",
                               "brand": "demo", "color": None, "bullet_points": "",
                               "label": label})
        groups.append({"query_id": f"q{i}", "query": f"{product_type} requested model {i}",
                       "source_split": "train" if i < 140 else "test",
                       "candidates": candidates})
    return groups


def prepare(cfg, mock=False):
    sources = {}
    if mock:
        groups = synthetic_groups(cfg.seed)
    else:
        import pandas as pd
        for name in FILES:
            path = Path(cfg.data_dir) / name
            validate_parquet(path)
            sources[name] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
        ex = pd.read_parquet(Path(cfg.data_dir) / FILES[0], columns=[
            "query_id", "query", "product_id", "product_locale", "esci_label", "split"])
        products = pd.read_parquet(Path(cfg.data_dir) / FILES[1], columns=[
            "product_id", "product_locale", "product_title", "product_brand",
            "product_color", "product_bullet_point"])
        ex = ex[(ex.product_locale == "us") & ex.split.isin(["train", "test"])].copy()
        products = products[products.product_locale == "us"].copy()
        products = products.dropna(subset=["product_id", "product_title"])
        products = products[products.product_title.str.strip().ne("")]
        if products.duplicated(["product_id", "product_locale"]).any():
            raise ValueError("Duplicate product IDs in product metadata")
        ex = ex.dropna(subset=["query_id", "query", "product_id", "esci_label"])
        ex["esci_label"] = ex.esci_label.str.upper()
        if not ex.esci_label.isin(["E", "S", "C", "I"]).all():
            raise ValueError("Unknown ESCI label")
        conflicts = ex.groupby(["query_id", "product_id"]).esci_label.nunique()
        if (conflicts > 1).any():
            raise ValueError("Conflicting judgments for a query/product pair")
        ex = ex.drop_duplicates(["query_id", "product_id"])
        ex = ex[ex.product_id.isin(products.product_id)]
        # Select query groups first, then attach only needed product text to limit RAM.
        groups = []
        for (split, qid), frame in ex.groupby(["split", "query_id"], sort=True):
            if frame["query"].nunique() != 1:
                raise ValueError(f"Query ID has multiple texts: {qid}")
            if len(frame) >= 5:
                groups.append({"query_id": str(qid), "query": str(frame["query"].iloc[0]),
                               "source_split": split,
                               "candidates": [{"id": str(pid), "label": label} for pid, label
                                              in zip(frame.product_id, frame.esci_label)]})
        # A cheap preselection bounds the text lookup, retaining challenge candidate pools.
        preliminary = split_groups(groups, cfg)
        wanted = {c["id"] for g in preliminary for c in g["candidates"]}
        products["product_id"] = products.product_id.astype(str)
        lookup = products[products.product_id.isin(wanted)].set_index("product_id").fillna("").to_dict("index")
        for group in preliminary:
            for c in group["candidates"]:
                p = lookup[c["id"]]
                c.update(title=str(p["product_title"]), brand=str(p["product_brand"]) or None,
                         color=str(p["product_color"]) or None,
                         bullet_points=str(p["product_bullet_point"]))
        rows = preliminary
    if mock:
        rows = split_groups(groups, cfg, mock=True)
    write_jsonl(cfg.root(mock) / "prepared.jsonl", rows)
    counts = {s: sum(g["split"] == s for g in rows)
              for s in ("development", "validation", "test", "challenge")}
    metadata(cfg, mock, "prepare", {"sources": sources, "counts": counts,
                                   "prepared_fingerprint": digest(rows)})
    print(f"{'SIMULATED ' if mock else ''}Prepared: {counts}")
