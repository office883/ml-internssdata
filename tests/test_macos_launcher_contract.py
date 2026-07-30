from pathlib import Path


def test_macos_launcher_is_fail_closed_for_raqm() -> None:
    script = (Path(__file__).resolve().parents[1] / "RUN_ME.command").read_text(encoding="utf-8")
    required = [
        "libraqm",
        "harfbuzz",
        "fribidi",
        "--no-binary Pillow",
        "features.check_feature(\"raqm\")",
        "direction=\"rtl\"",
        "language=\"he\"",
        "WEBP",
        "hf\" auth whoami",
        "grep -q 'skipped'",
        "heocr_unified run",
    ]
    missing = [token for token in required if token not in script]
    assert not missing, missing


def test_runtime_release_version_is_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    assert 'version = "15.0.0"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"builder_version": "15.0.0"' in (root / "config.json").read_text(encoding="utf-8")


def test_macos_smoke_test_uses_the_same_filtered_font_discovery_as_the_builder() -> None:
    script = (Path(__file__).resolve().parents[1] / "RUN_ME.command").read_text(encoding="utf-8")
    assert "from heocr_unified.fonts import discover_fonts" in script
    assert "TTFont(candidate" not in script
