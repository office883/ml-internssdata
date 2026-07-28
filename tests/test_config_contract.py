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


def test_architecture_extra_variant_rate_must_be_probability(tmp_path: Path) -> None:
    import pytest
    from heocr_unified.config import load_config
    for value in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="architecture_extra_variant_rate"):
            load_config(None, overrides={
                "work_dir": str(tmp_path), "upload": False,
                "architecture_extra_variant_rate": value,
            })


def test_private_repository_is_mandatory(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(ValueError, match="private"):
        load_config(None, overrides={"work_dir": str(tmp_path), "private": False})


def test_output_repo_must_be_owner_and_name(tmp_path: Path) -> None:
    import pytest
    for value in ("missing-slash", "/repo", "owner/", "owner/repo/extra", "owner/re po"):
        with pytest.raises(ValueError, match="output_repo"):
            load_config(None, overrides={"work_dir": str(tmp_path), "output_repo": value})


def test_positive_build_counts_and_nonnegative_free_space(tmp_path: Path) -> None:
    import pytest
    positive = (
        "rows_per_shard", "page_rows_per_shard", "architecture_chunk_size",
        "structured_chunk_size", "page_chunk_size", "page_pool_limit",
        "architecture_structured_lines", "architecture_pages",
        "architecture_max_graphemes", "pointed_variants_per_text",
        "pointed_chunk_size", "pointed_mini_per_split", "pointed_max_graphemes",
    )
    for key in positive:
        with pytest.raises(ValueError, match=key):
            load_config(None, overrides={"work_dir": str(tmp_path), key: 0})
    with pytest.raises(ValueError, match="minimum_free_gib"):
        load_config(None, overrides={"work_dir": str(tmp_path), "minimum_free_gib": -1})


def test_font_paths_must_be_unique_nonempty_repository_paths(tmp_path: Path) -> None:
    import pytest
    base = list(DEFAULT_CONFIG["font_repo"]["paths"])
    for paths in ([], base + [base[0]], ["/absolute"], ["../escape"], [""]):
        with pytest.raises(ValueError, match="font_repo.paths"):
            load_config(None, overrides={"work_dir": str(tmp_path), "font_repo": {"paths": paths}})
