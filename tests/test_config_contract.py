from __future__ import annotations

from pathlib import Path

from heocr_unified.config import (
    DEFAULT_CONFIG, build_fingerprint, builder_code_fingerprint, load_config,
)


def test_source_revisions_are_pinned() -> None:
    sources = DEFAULT_CONFIG["sources"]
    assert all(len(item["revision"]) == 40 for item in sources.values())


def test_operational_work_dir_does_not_change_content_fingerprint(tmp_path: Path) -> None:
    a = load_config(None, overrides={"work_dir": str(tmp_path / "a")})
    b = load_config(None, overrides={"work_dir": str(tmp_path / "b")})
    assert build_fingerprint(a) == build_fingerprint(b)


def test_builder_code_fingerprint_changes_when_source_bytes_change(tmp_path: Path) -> None:
    package = tmp_path / "heocr_unified"
    package.mkdir()
    (package / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "requirements-lock.txt").write_text("p==1\n", encoding="utf-8")
    first = builder_code_fingerprint(tmp_path)
    (package / "a.py").write_text("x = 2\n", encoding="utf-8")
    second = builder_code_fingerprint(tmp_path)
    assert first != second


def test_loaded_config_records_builder_code_hash(tmp_path: Path) -> None:
    config = load_config(None, overrides={"work_dir": str(tmp_path)})
    assert len(config["builder_code_sha256"]) == 64
    assert config["builder_code_sha256"] == builder_code_fingerprint()


def test_builder_code_fingerprint_includes_locked_dependencies(tmp_path: Path) -> None:
    package = tmp_path / "heocr_unified"
    package.mkdir()
    (package / "a.py").write_text("x = 1\n", encoding="utf-8")
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("p==1\n", encoding="utf-8")
    first = builder_code_fingerprint(tmp_path)
    lock.write_text("p==2\n", encoding="utf-8")
    assert builder_code_fingerprint(tmp_path) != first


def test_user_cannot_spoof_builder_code_hash(tmp_path: Path) -> None:
    config = load_config(
        None,
        overrides={"work_dir": str(tmp_path), "builder_code_sha256": "0" * 64},
    )
    assert config["builder_code_sha256"] == builder_code_fingerprint()
    assert config["builder_code_sha256"] != "0" * 64


def test_builder_code_hash_changes_content_fingerprint(tmp_path: Path) -> None:
    config = load_config(None, overrides={"work_dir": str(tmp_path)})
    other = dict(config)
    other["builder_code_sha256"] = "f" * 64
    assert build_fingerprint(config) != build_fingerprint(other)
