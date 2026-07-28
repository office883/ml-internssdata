from __future__ import annotations

import json
from PIL import features
import pytest

from heocr_unified.generated_build import generate_page_specs, render_page_row, render_structured_row
from heocr_unified.render import TextRenderer
from heocr_unified.structured import generate_structured_examples


@pytest.mark.skipif(not features.check_feature("raqm"), reason="Pillow lacks RAQM")
def test_structured_rows_render_with_identity_and_metadata() -> None:
    renderer=TextRenderer.from_system_fonts()
    examples=list(generate_structured_examples(20,seed=7))
    rows=[render_structured_row(example,renderer=renderer,architecture_revision="a"*40) for example in examples]
    assert len({row["sample_id"] for row in rows}) == 20
    assert all(row["task"] == "line_recognition" for row in rows)
    assert all(row["source_repo"] == "ssdataanalysis/hebrew-architecture-corpus" for row in rows)
    assert all(json.loads(row["augmentation_json"])["profile"] for row in rows)


def test_page_specs_use_lines_from_their_own_split_and_are_deterministic() -> None:
    pools={
        "train":[f"שורת אימון {i}" for i in range(50)],
        "validation_synthetic":[f"שורת אימות {i}" for i in range(30)],
        "test_synthetic":[f"שורת מבחן {i}" for i in range(30)],
    }
    a=list(generate_page_specs(pools,40,seed=5))
    b=list(generate_page_specs(pools,40,seed=5))
    assert a == b
    for spec in a:
        assert all(line in pools[spec.split] for line in spec.lines)
    assert {spec.split for spec in a} == set(pools)


@pytest.mark.skipif(not features.check_feature("raqm"), reason="Pillow lacks RAQM")
def test_page_row_contains_transformed_reading_order_annotations() -> None:
    renderer=TextRenderer.from_system_fonts()
    pools={
        "train":[f"שורת אימון {i}" for i in range(50)],
        "validation_synthetic":[f"שורת אימות {i}" for i in range(30)],
        "test_synthetic":[f"שורת מבחן {i}" for i in range(30)],
    }
    spec=next(iter(generate_page_specs(pools,1,seed=9)))
    row=render_page_row(spec,renderer=renderer,architecture_revision="a"*40)
    annotations=json.loads(row["annotations_json"])
    assert row["task"] == "page_transcription"
    assert annotations
    assert [item["reading_order"] for item in annotations] == list(range(len(annotations)))
    assert row["text"].splitlines() == [item["text"] for item in annotations]
