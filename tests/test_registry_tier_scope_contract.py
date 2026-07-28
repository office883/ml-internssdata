from __future__ import annotations

from pathlib import Path

from heocr_unified.registry import DedupRegistry


def _accept(reg: DedupRegistry, *, sample_id: str, split: str, tier: str, text: str, visual: str, label: str | None = None):
    return reg.accept_sample(
        sample_id=sample_id, split=split, task="line_recognition",
        byte_sha256="b-" + sample_id, visual_sha256=visual,
        text_sha256=label or text, writer_key="", document_key="", page_key="",
        source_key="s", text_cap=4, data_tier=tier,
        sample_origin="unknown" if tier == "quarantine" else "synthetic",
    )


def test_lower_trust_evaluation_never_blocks_gold_train(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    assert _accept(reg, sample_id="qe", split="test", tier="quarantine", text="t", visual="vq").accepted
    assert _accept(reg, sample_id="ee", split="validation", tier="extended", text="t", visual="ve").accepted
    assert _accept(reg, sample_id="gt", split="train", tier="gold", text="t", visual="vg").accepted


def test_gold_evaluation_blocks_extended_and_gold_train(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    assert _accept(reg, sample_id="ge", split="test", tier="gold", text="t", visual="v1").accepted
    assert not _accept(reg, sample_id="et", split="train", tier="extended", text="t", visual="v2").accepted
    assert not _accept(reg, sample_id="gt", split="train", tier="gold", text="t", visual="v3").accepted


def test_cross_tier_same_visual_does_not_let_quarantine_preempt_gold(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    assert _accept(reg, sample_id="q", split="test", tier="quarantine", text="t", visual="same").accepted
    assert _accept(reg, sample_id="g", split="train", tier="gold", text="t", visual="same").accepted


def test_text_variant_cap_is_isolated_by_tier(tmp_path: Path) -> None:
    reg = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    for tier in ("gold", "extended", "quarantine"):
        decision = reg.accept_sample(
            sample_id=tier, split="train", task="word_recognition",
            byte_sha256="b" + tier, visual_sha256="v" + tier,
            text_sha256="same", writer_key="", document_key="", page_key="",
            source_key="s", text_cap=1, data_tier=tier,
            sample_origin="unknown" if tier == "quarantine" else "synthetic",
        )
        assert decision.accepted
