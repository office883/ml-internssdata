from __future__ import annotations

import unicodedata

import pytest

from heocr_unified.unicode_utils import (
    LabelRejected,
    grapheme_clusters,
    normalize_label_strict,
    split_grapheme_safe,
)


def test_preserves_logical_mixed_bidi_order() -> None:
    text = "חשבונית INV-2026-17 — סה״כ ₪1,234.50"
    result = normalize_label_strict(text)
    assert result.text == text
    assert result.base_direction == "rtl"


def test_decomposes_hebrew_presentation_forms_but_keeps_exact_punctuation() -> None:
    result = normalize_label_strict("שָׁלוֹם־עולם ״בדיקה״")
    assert result.text == unicodedata.normalize("NFC", "שָׁלוֹם־עולם ״בדיקה״")
    assert not any(0xFB1D <= ord(ch) <= 0xFB4F for ch in result.text)


@pytest.mark.parametrize(
    "bad",
    [
        "אב\ufffdג",
        "אב\u200fג",
        "אב\u202eג",
        "אב\u2067ג",
        "אב\u200bג",
        "אב\x00ג",
    ],
)
def test_rejects_unsafe_or_invisible_characters(bad: str) -> None:
    with pytest.raises(LabelRejected):
        normalize_label_strict(bad)


def test_rejects_orphan_combining_mark() -> None:
    with pytest.raises(LabelRejected, match="orphan"):
        normalize_label_strict("\u05b7שלום")


def test_grapheme_split_never_separates_niqqud() -> None:
    text = "שָׁלוֹם עולם ארוך מאוד עם ניקוד חָזָק ומילים נוספות"
    parts = split_grapheme_safe(text, max_graphemes=12, min_graphemes=3)
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")
    for part in parts:
        for cluster in grapheme_clusters(part):
            assert not unicodedata.combining(cluster[0])
        assert len(grapheme_clusters(part)) <= 12
