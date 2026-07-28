from __future__ import annotations

import pytest

from heocr_unified.registry import DedupRegistry, RegistryConflict


def _accept(registry: DedupRegistry, *, sample: str, split: str, text: str, visual: str, writer: str = "", document: str = ""):
    return registry.accept_sample(
        sample_id=sample,
        split=split,
        task="line_recognition",
        byte_sha256=(sample[0] * 64),
        visual_sha256=visual,
        text_sha256=text,
        writer_key=writer,
        document_key=document,
        page_key="",
        source_key="source",
        text_cap=32,
        data_tier="gold",
        sample_origin="human" if split in {"test", "validation"} else "synthetic",
    )


def test_test_owns_text_over_validation_and_train(tmp_path) -> None:
    registry = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    assert _accept(registry, sample="a", split="test", text="1" * 64, visual="a" * 64).accepted
    decision = _accept(registry, sample="b", split="validation", text="1" * 64, visual="b" * 64)
    assert not decision.accepted
    assert decision.reason == "evaluation_text_owned_by_test"
    decision = _accept(registry, sample="c", split="train", text="1" * 64, visual="c" * 64)
    assert not decision.accepted
    assert decision.reason == "train_text_reserved_by_evaluation"
    registry.close()


def test_higher_priority_evaluation_arriving_late_fails_closed(tmp_path) -> None:
    registry = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    assert _accept(registry, sample="a", split="validation", text="1" * 64, visual="a" * 64).accepted
    with pytest.raises(RegistryConflict, match="higher-priority"):
        _accept(registry, sample="b", split="test", text="1" * 64, visual="b" * 64)
    registry.close()


def test_multiple_lines_same_writer_same_split_are_allowed(tmp_path) -> None:
    registry = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    assert _accept(registry, sample="a", split="test", text="1" * 64, visual="a" * 64, writer="w").accepted
    assert _accept(registry, sample="b", split="test", text="2" * 64, visual="b" * 64, writer="w").accepted
    registry.close()


def test_writer_owned_by_test_rejects_validation_and_train(tmp_path) -> None:
    registry = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    assert _accept(registry, sample="a", split="test", text="1" * 64, visual="a" * 64, writer="w").accepted
    decision = _accept(registry, sample="b", split="validation", text="2" * 64, visual="b" * 64, writer="w")
    assert not decision.accepted
    assert decision.reason == "evaluation_writer_owned_by_test"
    decision = _accept(registry, sample="c", split="train", text="3" * 64, visual="c" * 64, writer="w")
    assert not decision.accepted
    assert decision.reason == "train_writer_reserved_by_evaluation"
    registry.close()
