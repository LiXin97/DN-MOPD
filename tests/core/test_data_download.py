# SPDX-License-Identifier: Apache-2.0
"""data/download.py against a local stand-in for the Hub (plain-HTTP backend, no network), plus an opt-in live test
against the real dataset (DN_MOPD_NETWORK_TESTS=1)."""
import functools
import hashlib
import http.server
import importlib.util
import json
import os
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("dn_mopd_data_download", REPO / "data" / "download.py")
download = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download)

FILES = {"sub/a.jsonl": b'{"prompt": "a"}\n' * 1000, "b.jsonl": b'{"prompt": "b"}\n'}
REPO_ID, REV = "someone/fake-data", "0123456789abcdef0123456789abcdef01234567"


def _entry(blob):
    return {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}


class _Hub:
    """Serves <root>/datasets/<repo>/resolve/<rev>/<path> and counts the requests."""

    def __init__(self, root: Path):
        self.root, self.requests = root, []
        hub = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                hub.requests.append(self.path)
                super().do_GET()

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                      functools.partial(Handler, directory=str(root)))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def put(self, rel, blob):
        path = self.root / "datasets" / REPO_ID / "resolve" / REV / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    h = _Hub(tmp_path / "srv")
    for rel, blob in FILES.items():
        h.put(rel, blob)
    h.thread.start()
    monkeypatch.setenv("HF_ENDPOINT", h.url)
    yield h
    h.server.shutdown()
    h.server.server_close()


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"hub": {"repo_id": REPO_ID, "repo_type": "dataset", "revision": REV},
                                "files": {rel: _entry(blob) for rel, blob in FILES.items()}}))
    return path


def _run(manifest, out, *extra):
    return download.main(["--manifest", str(manifest), "--out-dir", str(out), "--backend", "http", *extra])


def test_fetches_verifies_and_cleans_up(hub, manifest, tmp_path):
    out = tmp_path / "data"
    assert _run(manifest, out) == 0
    for rel, blob in FILES.items():
        assert (out / rel).read_bytes() == blob
    assert not (out / download.STAGING).exists()
    # the revision's own MANIFEST.json was asked for; its absence (404 here) is not an error
    assert any(p.endswith(f"/{REV}/MANIFEST.json") for p in hub.requests)


def test_present_files_are_not_downloaded_again(hub, manifest, tmp_path):
    out = tmp_path / "data"
    assert _run(manifest, out) == 0
    before = len(hub.requests)
    assert _run(manifest, out) == 0
    assert len(hub.requests) == before


def test_hash_mismatch_fails_and_places_nothing(hub, manifest, tmp_path):
    hub.put("b.jsonl", b'{"prompt": "tampered"}\n')
    out = tmp_path / "data"
    assert _run(manifest, out) == 1
    assert not (out / "b.jsonl").exists()
    assert (out / "sub/a.jsonl").read_bytes() == FILES["sub/a.jsonl"]


def test_a_corrupt_local_file_is_replaced(hub, manifest, tmp_path):
    out = tmp_path / "data"
    (out / "sub").mkdir(parents=True)
    (out / "sub/a.jsonl").write_bytes(b"truncated")
    assert _run(manifest, out) == 0
    assert (out / "sub/a.jsonl").read_bytes() == FILES["sub/a.jsonl"]


def test_only_selects_files(hub, manifest, tmp_path):
    out = tmp_path / "data"
    assert _run(manifest, out, "--only", "data/sub/a.jsonl") == 0
    assert (out / "sub/a.jsonl").exists() and not (out / "b.jsonl").exists()
    with pytest.raises(SystemExit) as e:
        _run(manifest, out, "--only", "nope.jsonl")
    assert e.value.code == 2


def test_check_mode_never_downloads(hub, manifest, tmp_path):
    out = tmp_path / "data"
    assert _run(manifest, out, "--check") == 1
    assert hub.requests == []
    assert _run(manifest, out) == 0
    assert _run(manifest, out, "--check") == 0


def test_revision_whose_manifest_disagrees_is_refused(hub, manifest, tmp_path):
    hub.put("MANIFEST.json", json.dumps({"files": {"b.jsonl": {"sha256": "0" * 64}}}).encode())
    out = tmp_path / "data"
    assert _run(manifest, out) == 1
    assert not (out / "b.jsonl").exists() and not (out / "sub/a.jsonl").exists()


def test_release_manifest_pins_every_file():
    m = json.loads((REPO / "data" / "MANIFEST.json").read_text())
    assert m["hub"]["repo_id"] and len(m["hub"]["revision"]) == 40
    assert all(len(v["sha256"]) == 64 and v["bytes"] > 0 for v in m["files"].values())


@pytest.mark.skipif(os.environ.get("DN_MOPD_NETWORK_TESTS") != "1",
                    reason="live Hugging Face download; set DN_MOPD_NETWORK_TESTS=1")
@pytest.mark.parametrize("backend", ["http", "hub"])
def test_live_download_of_the_small_files(backend, tmp_path, monkeypatch):
    if backend == "hub":
        pytest.importorskip("huggingface_hub")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    rc = download.main(["--out-dir", str(tmp_path), "--backend", backend,
                        "--only", "teacher/math_train.jsonl", "--only", "teacher/if_train.jsonl"])
    assert rc == 0
    assert (tmp_path / "teacher/math_train.jsonl").stat().st_size == 3838298
