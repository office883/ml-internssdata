from __future__ import annotations

from pathlib import Path

import pandas as pd

from heocr_unified.architecture import ArchitectureCorpus, ArchitectureState, ArchitectureTextResolver


def _fixture(tmp_path: Path) -> ArchitectureCorpus:
    root = tmp_path / "arch"
    (root / "txt").mkdir(parents=True)
    pd.DataFrame([
        {"corpus_index": "1", "origin": "Born digital"},
        {"corpus_index": "2", "origin": "Born digital"},
        {"corpus_index": "3", "origin": "Born digital"},
        {"corpus_index": "4", "origin": "Scanned"},
    ]).to_csv(root / "full_IIA_corpus.csv", index=False)
    for doc in ("1", "2", "3"):
        (root / "txt" / f"{doc}.txt").write_text("אותה שורת אדריכלות 17\n", encoding="utf-8")
    (root / "txt" / "4.txt").write_text("טקסט סרוק חשוד\n", encoding="utf-8")
    corpus = ArchitectureCorpus(root)
    split_map = {"1": "train", "2": "validation_synthetic", "3": "test_synthetic", "4": "train"}
    corpus.document_split = lambda document_id: split_map[document_id]  # type: ignore[method-assign]
    return corpus


def test_global_resolver_assigns_one_canonical_owner_by_split_priority(tmp_path: Path) -> None:
    corpus = _fixture(tmp_path)
    resolver = ArchitectureTextResolver(
        tmp_path / "resolver.sqlite",
        source_revision="a" * 40,
        policy="test_synthetic>validation_synthetic>train",
    )
    summary = resolver.build(corpus)
    rows = list(corpus.iter_accounted_segments(resolver=resolver))
    gold = [row for row in rows if row.state == ArchitectureState.GOLD]
    duplicates = [row for row in rows if row.state == ArchitectureState.DUPLICATE]
    quarantine = [row for row in rows if row.state == ArchitectureState.QUARANTINE]
    assert len(gold) == 1
    assert gold[0].document_id == "3"
    assert gold[0].split == "test_synthetic"
    assert len(duplicates) == 2
    assert len(quarantine) == 1
    assert summary["canonical_gold_texts"] == 1
    assert summary["duplicate_gold_occurrences"] == 2
    resolver.close()


def test_resolver_is_deterministic_across_reopen(tmp_path: Path) -> None:
    corpus = _fixture(tmp_path)
    path = tmp_path / "resolver.sqlite"
    first = ArchitectureTextResolver(path, source_revision="a" * 40)
    summary_a = first.build(corpus)
    first.close()
    second = ArchitectureTextResolver(path, source_revision="a" * 40)
    summary_b = second.build(corpus)
    assert summary_a == summary_b
    second.close()


def test_architecture_resolver_rebuilds_if_cached_owner_table_is_incomplete(tmp_path: Path) -> None:
    corpus = _fixture(tmp_path)
    path = tmp_path / "resolver.sqlite"
    first = ArchitectureTextResolver(path, source_revision="a" * 40)
    assert first.build(corpus)["canonical_gold_texts"] == 1
    first.db.execute("DELETE FROM owners")
    first.db.commit()
    first.close()

    second = ArchitectureTextResolver(path, source_revision="a" * 40)
    summary = second.build(corpus)
    assert summary["canonical_gold_texts"] == 1
    assert second.owner(next(corpus.iter_raw_segments()).text_sha256) is not None
    second.close()
