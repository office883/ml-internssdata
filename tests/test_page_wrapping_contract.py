from __future__ import annotations

from pathlib import Path

from PIL import features
import pytest

from heocr_unified.render import PAGE_LAYOUTS, TextRenderer


@pytest.mark.skipif(not features.check_feature("raqm"), reason="Pillow lacks RAQM")
def test_long_page_lines_are_wrapped_without_loss_and_all_geometry_is_in_bounds() -> None:
    renderer = TextRenderer.from_system_fonts()
    long = "תכנית אדריכלית מפורטת מאוד הכוללת מידות 123.45 מטרים ופרטי ביצוע נוספים " * 5
    compact_source = "".join(long.split())
    for layout in PAGE_LAYOUTS:
        rendered = renderer.render_page([long] * 4, profile="clean_digital", layout=layout, seed=17, split="train")
        width, height = rendered.image.size
        assert rendered.annotations
        compact_rendered = "".join(str(item["text"]).replace(" ", "") for item in rendered.annotations)
        assert compact_rendered == compact_source * 4
        assert [item["reading_order"] for item in rendered.annotations] == list(range(len(rendered.annotations)))
        for item in rendered.annotations:
            x0, y0, x1, y1 = map(float, item["bbox"])
            assert 0 <= x0 < x1 <= width
            assert 0 <= y0 < y1 <= height
            for field in ("polygon", "baseline"):
                for x, y in item[field]:
                    assert 0 <= float(x) <= width
                    assert 0 <= float(y) <= height


@pytest.mark.skipif(not features.check_feature("raqm"), reason="Pillow lacks RAQM")
def test_table_layout_adapts_font_size_instead_of_dropping_long_text() -> None:
    renderer = TextRenderer.from_system_fonts()
    long = "תכנית אדריכלית מפורטת מאוד הכוללת מידות 123.45 מטרים ופרטי ביצוע נוספים " * 6
    page = renderer.render_page(
        [long] * 4,
        profile="clean_digital",
        layout="table",
        seed=17,
        split="train",
    )
    compact_source = "".join(long.split()) * 4
    compact_rendered = "".join(str(item["text"]).replace(" ", "") for item in page.annotations)
    assert compact_rendered == compact_source
    assert 11 <= int(page.metadata["page_font_px"]) <= 27


@pytest.mark.skipif(not features.check_feature("raqm"), reason="Pillow lacks RAQM")
def test_maximum_architecture_page_budget_fits_every_layout_without_text_loss() -> None:
    renderer = TextRenderer.from_system_fonts()
    source = (
        "תכנית אדריכלית מפורטת הכוללת מידות 123.45 מטרים ופרטי ביצוע נוספים " * 3
    )[:112]
    expected = "".join(source.split()) * 24
    for layout in PAGE_LAYOUTS:
        page = renderer.render_page(
            [source] * 24,
            profile="clean_digital",
            layout=layout,
            seed=20260730,
            split="train",
        )
        rendered = "".join(str(item["text"]).replace(" ", "") for item in page.annotations)
        assert rendered == expected, layout
        assert 11 <= int(page.metadata["page_font_px"]) <= 27
