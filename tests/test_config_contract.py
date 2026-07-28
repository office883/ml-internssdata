from __future__ import annotations

from pathlib import Path

from heocr_unified.config import DEFAULT_CONFIG, build_fingerprint, load_config


def test_source_revisions_are_pinned() -> None:
    sources = DEFAULT_CONFIG["sources"]
    assert all(len(item["revision"]) == 40 for item in sources.values())


def test_operational_work_dir_does_not_change_content_fingerprint(tmp_path: Path) -> None:
    a = load_config(None, overrides={"work_dir": str(tmp_path / "a")})
    b = load_config(None, overrides={"work_dir": str(tmp_path / "b")})
    assert build_fingerprint(a) == build_fingerprint(b)
