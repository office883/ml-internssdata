from __future__ import annotations

from pathlib import Path

from heocr_unified.config import load_config
from heocr_unified.metadata import write_usage_documents


def test_usage_documents_define_safe_training_and_source_policy(tmp_path: Path) -> None:
    config = load_config(None, overrides={"work_dir": str(tmp_path), "upload": False})
    write_usage_documents(tmp_path, config=config)
    source = (tmp_path / "SOURCE_POLICY.md").read_text(encoding="utf-8")
    recipe = (tmp_path / "TRAINING_RECIPE.md").read_text(encoding="utf-8")

    assert "license: other" in source
    assert "hebrew-architecture-corpus" in source
    assert "Samaritan" in source
    assert "extended" in source
    assert "quarantine" in source
    for source_config in config["sources"].values():
        assert source_config["revision"] in source

    assert "unified_recognition_lines" in recipe
    assert "extended_recognition_lines" in recipe
    assert "document_pages" in recipe
    assert "human validation" in recipe.lower()
    assert "Grapheme-CER" in recipe
    assert "quarantine" in recipe
    assert "recommended_sampling_weight" in recipe
