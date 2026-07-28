from __future__ import annotations

from pathlib import Path

import pytest
from PIL import features

from heocr_unified.render import LINE_PROFILES, PAGE_LAYOUTS, PAGE_PROFILES, TextRenderer


@pytest.mark.skipif(not features.check_feature("raqm"), reason="Pillow lacks RAQM")
def test_all_line_profiles_preserve_visible_foreground(tmp_path: Path) -> None:
    renderer = TextRenderer.from_system_fonts(extra_dirs=[])
    text = "חשבונית A-104 — סה״כ ₪1,234.50"
    for index, profile in enumerate(LINE_PROFILES):
        rendered = renderer.render_line(text, profile=profile, seed=1000 + index, split="train")
        assert rendered.visibility_fraction >= 0.42, profile
        assert rendered.text == text
        assert rendered.image.width > 10 and rendered.image.height > 10


@pytest.mark.skipif(not features.check_feature("raqm"), reason="Pillow lacks RAQM")
def test_page_annotations_match_reading_order() -> None:
    renderer = TextRenderer.from_system_fonts(extra_dirs=[])
    lines = ["שורה ראשונה 1", "שורה שנייה A-2", "שורה שלישית"] * 4
    for layout in PAGE_LAYOUTS:
        page = renderer.render_page(lines, profile=PAGE_PROFILES[0], layout=layout, seed=42, split="train")
        ordered = sorted(page.annotations, key=lambda row: row["reading_order"])
        assert [row["text"] for row in ordered] == lines[: len(ordered)]
        for row in ordered:
            x0, y0, x1, y1 = row["bbox"]
            assert 0 <= x0 < x1 <= page.image.width
            assert 0 <= y0 < y1 <= page.image.height


def test_font_discovery_can_be_limited_to_pinned_directories(tmp_path: Path) -> None:
    from heocr_unified.fonts import discover_fonts
    assert discover_fonts([tmp_path], include_system=False) == []
