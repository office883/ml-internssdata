from __future__ import annotations

import json
from pathlib import Path

import heocr_unified.finalize as finalize


def test_finalize_ready_records_all_trust_tiers(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "BUILD_FINGERPRINT").write_text("f" * 64, encoding="ascii")
    (tmp_path / "data.bin").write_bytes(b"x")
    monkeypatch.setattr(finalize, "verify_output_dataset", lambda *a, **k: {
        "all_rows": 30, "gold_rows": 20, "extended_rows": 7, "quarantine_rows": 3,
    })
    monkeypatch.setattr(finalize, "generate_previews", lambda root: {"sheets": []})
    monkeypatch.setattr(finalize, "run_corruption_suite", lambda *a, **k: {
        "status": "PASS", "test_count": 7, "tests": []
    })
    ready = finalize.finalize_local_release(
        tmp_path, registry_path=tmp_path / "r.sqlite",
        config={"sources": {"fixture": {"repo_id": "r", "revision": "a" * 40}}},
        mini=True
    )
    assert ready["total_rows"] == 30
    assert ready["gold_rows"] == 20
    assert ready["extended_rows"] == 7
    assert ready["quarantine_rows"] == 3
    assert json.loads((tmp_path / "LOCAL_READY.json").read_text())["status"] == "PASS"
