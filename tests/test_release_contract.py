from __future__ import annotations

import json
from pathlib import Path

import pytest

from heocr_unified.release import build_release_manifest, verify_release_manifest


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    manifest = build_release_manifest(tmp_path)
    path = tmp_path / "RELEASE_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    verify_release_manifest(tmp_path, manifest)
    (tmp_path / "a.txt").write_text("b", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_release_manifest(tmp_path, manifest)


def test_manifest_detects_unexpected_extra_files_and_total_bytes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    manifest = build_release_manifest(tmp_path)
    (tmp_path / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected release files"):
        verify_release_manifest(tmp_path, manifest)
    (tmp_path / "extra.txt").unlink()
    manifest["total_bytes"] += 1
    with pytest.raises(ValueError, match="total bytes"):
        verify_release_manifest(tmp_path, manifest)
