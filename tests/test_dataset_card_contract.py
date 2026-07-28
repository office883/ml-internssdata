from __future__ import annotations

from pathlib import Path

from heocr_unified.metadata import write_dataset_card


def _touch(root: Path, config: str, split: str) -> None:
    path = root / "data" / config / split / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"parquet")


def test_default_dataset_card_excludes_extended_and_quarantine(tmp_path: Path) -> None:
    _touch(tmp_path, "modern_print_lines", "train")
    _touch(tmp_path, "modern_print_lines_extended", "train")
    _touch(tmp_path, "modern_print_lines_quarantine", "train")
    _touch(tmp_path, "architecture_synthetic_pages", "train")
    _touch(tmp_path, "modern_print_words", "train")
    _touch(tmp_path, "handwriting_real_characters", "test")
    write_dataset_card(tmp_path, {"ok": True})
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    default = text.split("- config_name: extended_recognition_lines", 1)[0]
    assert "data/modern_print_lines/train/*.parquet" in default
    assert "_extended" not in default
    assert "_quarantine" not in default
    assert "- config_name: quarantine_audit" in text
    assert "data/modern_print_lines_quarantine/train/*.parquet" in text
    assert "default: true" in text


def test_card_never_declares_nonexistent_config_paths(tmp_path: Path) -> None:
    _touch(tmp_path, "modern_print_lines", "train")
    write_dataset_card(tmp_path, {})
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "modern_print_words" not in text
    assert "architecture_synthetic_pages" not in text
