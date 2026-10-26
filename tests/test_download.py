from pathlib import Path

import httpx
import pandas as pd
import pytest

from jev_gate.data import FILES, download
from jev_gate.io import read_json


def test_download_validates_and_resumes_existing_files(cfg, tmp_path, monkeypatch):
    source = tmp_path/"source.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(source)
    content = source.read_bytes()
    requests = []
    def handler(request):
        requests.append(str(request.url))
        if "api.github.com" in str(request.url):
            return httpx.Response(200, json={"sha": "a"*40})
        return httpx.Response(200, content=content)
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handler), **kw))
    download(cfg)
    assert len(requests) == 3
    manifest = read_json(Path(cfg.data_dir)/"download_manifest.json")
    assert manifest["revision"] == "a"*40
    assert all((Path(cfg.data_dir)/name).is_file() for name in FILES)
    download(cfg)
    assert len(requests) == 3  # Does not even re-resolve main on resume.


def test_incomplete_download_is_not_promoted(cfg, monkeypatch):
    def handler(request):
        if "api.github.com" in str(request.url):
            return httpx.Response(200, json={"sha": "a"*40})
        return httpx.Response(200, content=b"partial", headers={"content-length": "99"})
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(ValueError, match="Incomplete"):
        download(cfg)
    assert not (Path(cfg.data_dir)/FILES[0]).exists()
    assert not list(Path(cfg.data_dir).glob("*.download"))
