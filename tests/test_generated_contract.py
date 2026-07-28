from __future__ import annotations

from heocr_unified.structured import generate_structured_examples


def test_structured_generator_is_deterministic_unique_and_domain_rich() -> None:
    first = list(generate_structured_examples(400, seed=20260726))
    second = list(generate_structured_examples(400, seed=20260726))
    assert first == second
    assert len(first) == 400
    assert len({row.text_sha256 for row in first}) == 400
    assert all(row.split in {"train", "validation_synthetic", "test_synthetic"} for row in first)
    assert sum(row.mixed_bidi for row in first) >= 120
    assert sum(row.with_digits for row in first) >= 350
    assert any("מ״ר" in row.text for row in first)
    assert any("A-" in row.text or "REV" in row.text for row in first)


def test_structured_generator_has_stable_document_level_splits() -> None:
    rows = list(generate_structured_examples(1000, seed=99))
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row.group_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in groups.values())
    assert {row.split for row in rows} == {"train", "validation_synthetic", "test_synthetic"}
