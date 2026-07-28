from __future__ import annotations

from pathlib import Path

import pandas as pd

from heocr_unified.architecture import ArchitectureCorpus, ArchitectureState


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "arch"
    (root / "txt").mkdir(parents=True)
    pd.DataFrame([
        {"corpus_index":"1", "origin":"Born digital", "year":2020, "title":"א"},
        {"corpus_index":"2", "origin":"Scanned", "year":1980, "title":"ב"},
    ]).to_csv(root / "full_IIA_corpus.csv", index=False)
    (root / "txt" / "1.txt").write_text("שורה תקינה בעברית\nשורה תקינה בעברית\n", encoding="utf-8")
    (root / "txt" / "2.txt").write_text("טקסט שעבר OCR ואינו זהב\n", encoding="utf-8")
    return root


def test_born_digital_is_gold_and_scanned_is_quarantine(tmp_path: Path) -> None:
    corpus = ArchitectureCorpus(_fixture(tmp_path))
    rows = list(corpus.iter_accounted_segments())
    states = {(r.document_id, r.state) for r in rows}
    assert ("1", ArchitectureState.GOLD) in states
    assert ("2", ArchitectureState.QUARANTINE) in states


def test_duplicate_gold_text_is_accounted_not_rendered_twice(tmp_path: Path) -> None:
    rows = list(ArchitectureCorpus(_fixture(tmp_path)).iter_accounted_segments())
    gold = [r for r in rows if r.document_id == "1"]
    assert [r.state for r in gold] == [ArchitectureState.GOLD, ArchitectureState.DUPLICATE]


def test_document_split_is_deterministic(tmp_path: Path) -> None:
    corpus = ArchitectureCorpus(_fixture(tmp_path))
    a = corpus.document_split("1")
    b = corpus.document_split("1")
    assert a == b
    assert a in {"train", "validation_synthetic", "test_synthetic"}


def test_split_filter_excludes_other_documents_before_dedup(tmp_path: Path) -> None:
    corpus = ArchitectureCorpus(_fixture(tmp_path))
    target = corpus.document_split("1")
    rows = list(corpus.iter_accounted_segments(splits={target}))
    assert rows
    assert all(row.split == target for row in rows)
