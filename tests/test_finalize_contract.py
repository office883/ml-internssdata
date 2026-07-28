from __future__ import annotations

import json
from pathlib import Path

from heocr_unified.finalize import write_checksums
from heocr_unified.release import build_release_manifest, verify_release_manifest


def test_checksums_and_release_manifest_exclude_readiness_markers(tmp_path: Path) -> None:
    (tmp_path/"a.txt").write_text("a",encoding="utf-8")
    (tmp_path/"LOCAL_READY.json").write_text("{}",encoding="utf-8")
    rows=write_checksums(tmp_path)
    assert [row["path"] for row in rows] == ["a.txt"]
    manifest=build_release_manifest(tmp_path)
    assert "LOCAL_READY.json" not in {row["path"] for row in manifest["files"]}
    verify_release_manifest(tmp_path,manifest)
