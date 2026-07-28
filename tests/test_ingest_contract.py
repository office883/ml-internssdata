from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
import pytest

from heocr_unified.ingest import ImageIntegrityError, make_sample_row


def _png() -> bytes:
    image = Image.new("L", (30, 10), 255)
    image.putpixel((10, 5), 0)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def test_make_sample_row_recomputes_image_and_text_identity() -> None:
    row = make_sample_row(
        image_bytes=_png(), image_path="x.png", text="שלום A-1", sample_id="s1",
        split="train", task="line_recognition", granularity="line", modality="print",
        data_tier="gold", is_synthetic=False, sample_origin="real", label_source="source_transcription", label_trust="gold", quality_tier="A", source_repo="r", source_revision="a"*40,
        source_path="p", source_split="train", source_id="i",
    )
    assert row["width"] == 30 and row["height"] == 10
    assert len(row["image_sha256"]) == 64
    assert len(row["visual_sha256"]) == 64
    assert len(row["text_sha256"]) == 64
    assert row["text"] == "שלום A-1"


def test_declared_image_hash_mismatch_is_fatal() -> None:
    with pytest.raises(ImageIntegrityError, match="declared"):
        make_sample_row(
            image_bytes=_png(), image_path="x.png", text="שלום", sample_id="s1",
            split="train", task="line_recognition", granularity="line", modality="print",
            data_tier="gold", is_synthetic=False, sample_origin="real", label_source="source_transcription", label_trust="gold", quality_tier="A", source_repo="r", source_revision="a"*40,
            source_path="p", source_split="train", source_id="i", declared_image_sha256="0"*64,
        )


def test_make_sample_row_fails_closed_on_unknown_gold_provenance() -> None:
    with pytest.raises(ValueError, match="provenance"):
        make_sample_row(
            image_bytes=_png(), image_path="x.png", text="שלום", sample_id="s2",
            split="train", task="line_recognition", granularity="line", modality="print",
            data_tier="gold", is_synthetic=False, source_repo="r", source_revision="a"*40,
            source_path="p", source_split="train", source_id="i",
        )


def test_quarantine_constructor_forces_zero_training_weight() -> None:
    row = make_sample_row(
        image_bytes=_png(), image_path="x.png", text="שלום", sample_id="q1",
        split="train", task="line_recognition", granularity="line", modality="print",
        data_tier="quarantine", is_synthetic=False, sample_origin="unknown",
        label_source="unknown", label_trust="quarantine", source_repo="r",
        source_revision="a"*40, source_path="p", source_split="train", source_id="i",
        recommended_sampling_weight=0.0,
    )
    assert row["recommended_sampling_weight"] == 0.0


def test_htr_sample_identity_is_namespaced_by_repo_file_and_row() -> None:
    from heocr_unified.ingest import _htr_sample_id
    a = _htr_sample_id("repo/a", "stage/train-000.parquet", "same", 7)
    b = _htr_sample_id("repo/a", "stage/train-001.parquet", "same", 7)
    c = _htr_sample_id("repo/a", "stage/train-000.parquet", "same", 8)
    assert a != b != c
    assert a.startswith("htr-")


def test_htr_provenance_retains_upstream_source_chain() -> None:
    from heocr_unified.ingest import _htr_provenance
    metadata = {
        "source_repo": "upstream/repo",
        "source_revision": "b" * 40,
        "source_split": "test",
        "source_file": "data/test.parquet",
        "source_row_index": 41,
        "text_group_id": "group-17",
        "human_source": "matan_primary",
    }
    provenance = _htr_provenance(metadata, curated_row_index=9, classification="curated_human_htr")
    assert provenance["curated_row_index"] == 9
    assert provenance["upstream_source_repo"] == "upstream/repo"
    assert provenance["upstream_source_revision"] == "b" * 40
    assert provenance["upstream_source_split"] == "test"
    assert provenance["upstream_source_file"] == "data/test.parquet"
    assert provenance["upstream_source_row_index"] == 41
    assert provenance["text_group_id"] == "group-17"
    assert provenance["human_source"] == "matan_primary"
