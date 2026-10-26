import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Missing {path}; run the previous pipeline stage first")
    # JSON strings may legally contain U+0085/U+2028/U+2029. str.splitlines()
    # splits those inside a record; JSONL framing uses physical CR/LF only.
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_env(path=Path(".env")):
    """Minimal dotenv reader; no evaluation, interpolation, or logging of values."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("Invalid .env line; expected KEY=VALUE")
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def metadata(cfg, mock, stage, extra=None):
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "httpx", "matplotlib",
                 "torch", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    value = {"stage": stage, "mock": mock, "config": cfg.dict(),
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "python": sys.version, "platform": platform.platform(), "packages": packages,
             **(extra or {})}
    write_json(cfg.root(mock) / f"{stage}_metadata.json", value)
    return value
