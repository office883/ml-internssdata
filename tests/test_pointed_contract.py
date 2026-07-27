from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from heocr_unified.pointed import (
    PointedTextResolver,
    PointedTextEntry,
    render_verified_pointed_row,
)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "source_dataset": "samaritan-ai/hebrew_synth_lines",
                "source_license": "mit",
                "rtl_text_order": "logical_unicode",
                "recommended": True,
                "standalone_selected": True,
                **row,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def test_pointed_resolver_is_global_deterministic_and_prefers_test(tmp_path: Path) -> None:
    manifest = tmp_path / "strict.jsonl.gz"
    pointed = "בְּרֵאשִׁית בָּרָא"
    other = "וַיֹּאמֶר אֱלֹהִים"
    _write_manifest(
        manifest,
        [
            {"id": "z", "curated_config": "biblical_pointed_lines", "split": "train", "text": pointed, "recommended": True},
            {"id": "b", "curated_config": "biblical_pointed_lines", "split": "test", "text": pointed, "recommended": True},
            {"id": "a", "curated_config": "biblical_pointed_lines", "split": "test", "text": pointed, "recommended": True},
            {"id": "c", "curated_config": "biblical_pointed_lines", "split": "validation", "text": other, "recommended": True},
            {"id": "plain", "curated_config": "biblical_pointed_lines", "split": "train", "text": "בראשית", "recommended": True},
            {"id": "wrong", "curated_config": "modern_print_lines", "split": "train", "text": pointed, "recommended": True},
        ],
    )
    resolver = PointedTextResolver(
        tmp_path / "pointed.sqlite",
        source_revision="a" * 40,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    summary = resolver.build(manifest)
    assert summary["canonical_texts"] == 2
    assert summary["duplicate_occurrences"] == 2
    assert summary["without_combining_marks"] == 1
    test_rows = list(resolver.iter_entries("test_synthetic"))
    assert len(test_rows) == 1
    assert test_rows[0].source_id == "a"
    assert test_rows[0].text == pointed
    assert [row.text for row in resolver.iter_entries("validation_synthetic")] == [other]
    assert list(resolver.iter_entries("train")) == []
    resolver.close()


class _Rendered:
    def __init__(self, text: str):
        self.text = text
        self.visibility_fraction = 1.0
        self.image = Image.new("RGB", (80, 24), "white")
        self.font = SimpleNamespace(family="Alef", style="Regular", sha256="c" * 64)
        self.metadata = {"profile": "clean_digital", "visibility_fraction": 1.0}

    def to_bytes(self) -> bytes:
        import io
        buffer = io.BytesIO()
        self.image.save(buffer, "WEBP", lossless=True)
        return buffer.getvalue()


class _Renderer:
    def render_line(self, text: str, *, profile: str, seed: int, split: str, rashi: bool = False):
        assert split in {"train", "validation_synthetic", "test_synthetic"}
        assert not rashi
        return _Rendered(text)


def test_render_verified_pointed_row_is_gold_and_auditable() -> None:
    entry = PointedTextEntry(
        text="שָׁלוֹם",
        text_sha256="d" * 64,
        split="train",
        source_id="source-1",
        source_line=17,
        source_dataset="samaritan-ai/hebrew_synth_lines",
        source_license="mit",
    )
    row = render_verified_pointed_row(
        entry,
        variant=1,
        renderer=_Renderer(),
        source_revision="e" * 40,
        manifest_sha256="f" * 64,
    )
    assert row["data_tier"] == "gold"
    assert row["label_trust"] == "gold"
    assert row["sample_origin"] == "synthetic"
    assert row["label_source"] == "verified_text_rerender"
    assert row["split"] == "train"
    assert row["source_path"] == "manifests/strict_all.jsonl.gz#biblical_pointed_lines"
    assert json.loads(row["provenance_json"])["variant"] == 1
    assert json.loads(row["provenance_json"])["manifest_sha256"] == "f" * 64


def test_iter_verified_pointed_rows_emits_exact_variant_count() -> None:
    from heocr_unified.pointed import iter_verified_pointed_rows

    entries = [
        PointedTextEntry(
            text="שָׁלוֹם", text_sha256="1" * 64, split="test_synthetic",
            source_id="one", source_line=1,
            source_dataset="samaritan-ai/hebrew_synth_lines", source_license="mit",
            original_split="test",
        ),
        PointedTextEntry(
            text="בָּרוּךְ", text_sha256="2" * 64, split="test_synthetic",
            source_id="two", source_line=2,
            source_dataset="samaritan-ai/hebrew_synth_lines", source_license="mit",
            original_split="test",
        ),
    ]
    rows = list(iter_verified_pointed_rows(
        entries,
        variants=2,
        renderer=_Renderer(),
        source_revision="e" * 40,
        manifest_sha256="f" * 64,
    ))
    assert len(rows) == 4
    assert len({row["sample_id"] for row in rows}) == 4
    assert all(row["split"] == "test_synthetic" for row in rows)


def test_pointed_resolver_rebuilds_if_cached_entry_table_is_incomplete(tmp_path: Path) -> None:
    manifest = tmp_path / "strict.jsonl.gz"
    _write_manifest(
        manifest,
        [
            {"id": "one", "curated_config": "biblical_pointed_lines", "split": "train", "text": "שָׁלוֹם", "recommended": True},
            {"id": "two", "curated_config": "biblical_pointed_lines", "split": "validation", "text": "בָּרוּךְ", "recommended": True},
        ],
    )
    db = tmp_path / "pointed.sqlite"
    kwargs = {
        "source_revision": "a" * 40,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    first = PointedTextResolver(db, **kwargs)
    assert first.build(manifest)["canonical_texts"] == 2
    first.db.execute("DELETE FROM entries WHERE split='train'")
    first.db.commit()
    first.close()

    second = PointedTextResolver(db, **kwargs)
    summary = second.build(manifest)
    assert summary["canonical_texts"] == 2
    assert len(list(second.iter_entries("train"))) == 1
    second.close()


def test_pointed_resolver_rebuilds_if_cached_entry_content_changes_without_count_change(tmp_path: Path) -> None:
    manifest = tmp_path / "strict.jsonl.gz"
    _write_manifest(manifest, [
        {"id": "one", "curated_config": "biblical_pointed_lines", "split": "train", "text": "שָׁלוֹם", "recommended": True, "standalone_selected": True},
    ])
    db = tmp_path / "pointed.sqlite"
    kwargs = {
        "source_revision": "a" * 40,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    first = PointedTextResolver(db, **kwargs)
    expected = first.build(manifest)
    first.db.execute("UPDATE entries SET source_id='tampered'")
    first.db.commit()
    first.close()

    second = PointedTextResolver(db, **kwargs)
    rebuilt = second.build(manifest)
    assert rebuilt == expected
    assert list(second.iter_entries("train"))[0].source_id == "one"
    second.close()


def test_pointed_resolver_requires_explicit_recommendation_and_standalone_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "strict.jsonl.gz"
    _write_manifest(manifest, [
        {"id": "missing-standalone", "curated_config": "biblical_pointed_lines", "split": "train", "text": "שָׁלוֹם", "recommended": True, "standalone_selected": None},
        {"id": "missing-recommended", "curated_config": "biblical_pointed_lines", "split": "train", "text": "בָּרוּךְ", "recommended": None, "standalone_selected": True},
    ])
    resolver = PointedTextResolver(
        tmp_path / "pointed.sqlite",
        source_revision="a" * 40,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    summary = resolver.build(manifest)
    assert summary["canonical_texts"] == 0
    assert summary["not_recommended"] == 2
    resolver.close()
