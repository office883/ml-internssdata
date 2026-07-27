from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
import pytest

from heocr_unified.identity import byte_sha256, canonical_visual_sha256
from heocr_unified.registry import DedupRegistry, RegistryConflict


def _encoded(fmt: str) -> bytes:
    image = Image.new("L", (40, 12), 255)
    for x in range(8, 30):
        image.putpixel((x, 6), 0)
    buf = io.BytesIO()
    image.save(buf, fmt)
    return buf.getvalue()


def test_visual_hash_collides_across_lossless_encodings() -> None:
    png = _encoded("PNG")
    webp = _encoded("WEBP")
    assert byte_sha256(png) != byte_sha256(webp)
    assert canonical_visual_sha256(png) == canonical_visual_sha256(webp)


def test_evaluation_reservation_rejects_later_train_text(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    reg.reserve_evaluation_entity(
        split="validation",
        task="line_recognition",
        text_sha256="t1",
        visual_sha256="v1",
        writer_key="repo:writer:7",
        document_key="repo:doc:9",
        sample_id="eval-1",
    )
    decision = reg.accept_sample(
        sample_id="train-1",
        split="train",
        task="line_recognition",
        byte_sha256="b2",
        visual_sha256="v2",
        text_sha256="t1",
        writer_key="",
        document_key="",
        source_key="s",
        text_cap=10,
    )
    assert not decision.accepted
    assert decision.reason == "train_text_reserved_by_evaluation"


def test_word_labels_may_repeat_without_entity_or_image_overlap(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    first = reg.accept_sample(
        sample_id="word-a",
        split="validation",
        task="word_recognition",
        byte_sha256="b1",
        visual_sha256="v1",
        text_sha256="same-word",
        writer_key="repo:w1",
        document_key="repo:d1",
        source_key="a",
        text_cap=100,
    )
    second = reg.accept_sample(
        sample_id="word-b",
        split="test",
        task="word_recognition",
        byte_sha256="b2",
        visual_sha256="v2",
        text_sha256="same-word",
        writer_key="repo:w2",
        document_key="repo:d2",
        source_key="b",
        text_cap=100,
    )
    assert first.accepted and second.accepted


def test_same_visual_with_different_label_is_fatal(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    assert reg.accept_sample(
        sample_id="a", split="train", task="line_recognition",
        byte_sha256="b1", visual_sha256="v", text_sha256="t1",
        writer_key="", document_key="", source_key="a", text_cap=10,
    ).accepted
    with pytest.raises(RegistryConflict, match="visual-label"):
        reg.accept_sample(
            sample_id="b", split="train", task="line_recognition",
            byte_sha256="b2", visual_sha256="v", text_sha256="t2",
            writer_key="", document_key="", source_key="b", text_cap=10,
        )


def test_resume_rejects_build_fingerprint_change(tmp_path: Path) -> None:
    DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp1").close()
    with pytest.raises(RegistryConflict, match="fingerprint"):
        DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp2")


def test_source_transaction_rolls_back_samples_and_completion(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    with pytest.raises(RuntimeError):
        with reg.source_transaction("source-a"):
            assert reg.accept_sample(
                sample_id="a", split="train", task="line_recognition",
                byte_sha256="b", visual_sha256="v", text_sha256="t",
                writer_key="", document_key="", source_key="source-a", text_cap=10,
            ).accepted
            raise RuntimeError("boom")
    assert reg.sample_count() == 0
    assert not reg.source_is_complete("source-a")


def test_architecture_ledger_is_complete_and_idempotent(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    reg.record_architecture_segment(
        segment_key="doc:1:0", document_id="doc", source_line=1, segment_index=0,
        text_sha256="t1", source_state="gold", outcome="accepted", reason="born_digital",
        split="train", sample_id="s1",
    )
    reg.record_architecture_segment(
        segment_key="doc:2:0", document_id="doc", source_line=2, segment_index=0,
        text_sha256="t2", source_state="quarantine", outcome="quarantined", reason="origin_scanned",
        split="train", sample_id="",
    )
    # Exact replay is safe.
    reg.record_architecture_segment(
        segment_key="doc:1:0", document_id="doc", source_line=1, segment_index=0,
        text_sha256="t1", source_state="gold", outcome="accepted", reason="born_digital",
        split="train", sample_id="s1",
    )
    summary = reg.architecture_ledger_summary()
    assert summary["total"] == 2
    assert summary["outcomes"] == {"accepted": 1, "quarantined": 1}
    assert summary["gold_total"] == 1
    assert summary["gold_accounted"] == 1


def test_architecture_ledger_rejects_changed_replay(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    kwargs = dict(
        segment_key="doc:1:0", document_id="doc", source_line=1, segment_index=0,
        text_sha256="t1", source_state="gold", outcome="accepted", reason="born_digital",
        split="train", sample_id="s1",
    )
    reg.record_architecture_segment(**kwargs)
    with pytest.raises(RegistryConflict, match="architecture ledger conflict"):
        reg.record_architecture_segment(**{**kwargs, "outcome": "rejected"})


def test_evaluation_visual_reservation_blocks_later_train_even_for_word_task(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    reg.reserve_evaluation_entity(
        split="test", task="word_recognition", text_sha256="word-a",
        visual_sha256="visual-same", writer_key="", document_key="",
        sample_id="eval-word", data_tier="gold",
    )
    decision = reg.accept_sample(
        sample_id="train-word", split="train", task="word_recognition",
        byte_sha256="bytes", visual_sha256="visual-same", text_sha256="word-a",
        writer_key="", document_key="", source_key="train", text_cap=100,
        data_tier="gold", sample_origin="synthetic",
    )
    assert not decision.accepted
    assert decision.reason == "train_visual_reserved_by_evaluation"


def test_evaluation_visual_reservation_rejects_conflicting_label(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    reg.reserve_evaluation_entity(
        split="test", task="word_recognition", text_sha256="word-a",
        visual_sha256="visual-same", writer_key="", document_key="",
        sample_id="eval-a", data_tier="gold",
    )
    with pytest.raises(RegistryConflict, match="visual-label"):
        reg.reserve_evaluation_entity(
            split="validation", task="word_recognition", text_sha256="word-b",
            visual_sha256="visual-same", writer_key="", document_key="",
            sample_id="eval-b", data_tier="gold",
        )
