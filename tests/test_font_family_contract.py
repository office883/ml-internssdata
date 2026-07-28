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
