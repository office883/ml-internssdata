from __future__ import annotations

from fontTools.ttLib import TTFont, newTable

from heocr_unified.fonts import _font_family_name, _font_style_name


def _font_with_names(records: list[tuple[int, str]]) -> TTFont:
    font = TTFont()
    table = newTable("name")
    table.names = []
    for name_id, value in records:
        table.setName(value, name_id, 3, 1, 0x409)
    font["name"] = table
    return font


def test_typographic_family_name_id_16_wins_over_legacy_instance_name() -> None:
    font = _font_with_names([(1, "Assistant ExtraLight"), (16, "Assistant"), (2, "Regular"), (17, "Roman")])
    assert _font_family_name(font, "fallback") == "Assistant"
    assert _font_style_name(font, "Regular") == "Roman"


def _font(family: str):
    from pathlib import Path
    from heocr_unified.fonts import FontInfo
    return FontInfo(
        path=Path(f"/{family}.ttf"), family=family, style="Regular",
        sha256=(family.encode().hex() + "0" * 64)[:64], cmap=frozenset({0x05D0}),
        has_gpos=True, is_rashi=False,
    )


def test_font_family_split_holds_out_multiple_families_when_pool_is_large() -> None:
    from heocr_unified.fonts import split_font_families
    fonts = [_font(f"Family-{index:02d}") for index in range(15)]
    pools = split_font_families(fonts)
    families = {split: {font.family for font in rows} for split, rows in pools.items()}
    assert len(families["validation_synthetic"]) == 3
    assert len(families["test_synthetic"]) == 3
    assert len(families["train"]) == 9
    assert not (families["train"] & families["validation_synthetic"])
    assert not (families["train"] & families["test_synthetic"])
    assert not (families["validation_synthetic"] & families["test_synthetic"])


def test_font_family_split_is_stable_for_input_order() -> None:
    from heocr_unified.fonts import split_font_families
    fonts = [_font(f"Family-{index:02d}") for index in range(15)]
    forward = split_font_families(fonts)
    reverse = split_font_families(list(reversed(fonts)))
    for split in forward:
        assert {font.family for font in forward[split]} == {font.family for font in reverse[split]}


def _coverage_font(family: str, *, full: bool):
    from pathlib import Path
    from heocr_unified.fonts import FontInfo, POINTED_COVERAGE_CODEPOINTS
    cmap = set(POINTED_COVERAGE_CODEPOINTS if full else {0x05D0})
    return FontInfo(
        path=Path(f"/{family}.ttf"), family=family, style="Regular",
        sha256=(family.encode().hex() + "0" * 64)[:64], cmap=frozenset(cmap),
        has_gpos=True, is_rashi=False,
    )


def test_font_split_reserves_full_pointed_coverage_for_every_split() -> None:
    from heocr_unified.fonts import POINTED_COVERAGE_CODEPOINTS, split_font_families
    fonts = [
        _coverage_font("Full-A", full=True),
        _coverage_font("Full-B", full=True),
        _coverage_font("Full-C", full=True),
        *[_coverage_font(f"Partial-{index:02d}", full=False) for index in range(12)],
    ]
    pools = split_font_families(fonts)
    for split, rows in pools.items():
        assert any(
            POINTED_COVERAGE_CODEPOINTS.issubset(font.cmap) and font.has_gpos
            for font in rows if not font.is_rashi
        ), split


def test_system_font_discovery_rejects_last_resort_fallback(monkeypatch, tmp_path) -> None:
    """A cmap-wide fallback font must never be treated as Hebrew training ink."""
    from pathlib import Path
    import heocr_unified.fonts as fonts_module
    from heocr_unified.fonts import FontInfo

    candidate = tmp_path / "LastResort.otf"
    candidate.write_bytes(b"fixture")
    fake = FontInfo(
        path=candidate,
        family="LastResort",
        style="Regular",
        sha256="a" * 64,
        cmap=frozenset(range(0x110000)),
        has_gpos=True,
        is_rashi=False,
    )
    monkeypatch.setattr(fonts_module, "_load_one", lambda _path: fake)
    fonts_module._discover_fonts_cached.cache_clear()
    try:
        assert fonts_module.discover_fonts([tmp_path], include_system=False) == []
    finally:
        fonts_module._discover_fonts_cached.cache_clear()
