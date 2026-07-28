from __future__ import annotations

from heocr_unified.orchestrator import reserve_evaluation_candidates
from heocr_unified.registry import DedupRegistry


def _candidate(*, sample_id: str, split: str, text: str, visual: str = "") -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "task": "line_recognition",
        "text_sha256": text,
        "visual_sha256": visual,
        "writer_id": "",
        "source_document": "",
        "source_page": "",
        "data_tier": "gold",
        "sample_origin": "synthetic" if split.endswith("_synthetic") else "real",
    }


def test_global_reservation_sorts_by_priority_not_input_order(tmp_path) -> None:
    registry = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    report = reserve_evaluation_candidates(
        registry,
        [
            _candidate(sample_id="validation", split="validation_synthetic", text="t"),
            _candidate(sample_id="test", split="test_synthetic", text="t"),
        ],
    )
    assert report["candidates"] == 2
    assert report["reserved"] == 1
    assert report["rejected"] == 1
    assert report["rejects"] == {"evaluation_text_owned_by_test_synthetic": 1}

    validation = registry.accept_sample(
        sample_id="validation", split="validation_synthetic", task="line_recognition",
        byte_sha256="b1", visual_sha256="v1", text_sha256="t",
        writer_key="", document_key="", page_key="", source_key="validation",
        text_cap=8, data_tier="gold", sample_origin="synthetic",
    )
    test = registry.accept_sample(
        sample_id="test", split="test_synthetic", task="line_recognition",
        byte_sha256="b2", visual_sha256="v2", text_sha256="t",
        writer_key="", document_key="", page_key="", source_key="test",
        text_cap=8, data_tier="gold", sample_origin="synthetic",
    )
    assert not validation.accepted
    assert test.accepted


def test_global_reservation_is_idempotent_and_does_not_inflate_reject_counters(tmp_path) -> None:
    registry = DedupRegistry(tmp_path / "r.sqlite", build_fingerprint="fp")
    candidates = [
        _candidate(sample_id="test", split="test", text="t", visual="v"),
        _candidate(sample_id="validation", split="validation", text="t", visual="v"),
    ]
    first = reserve_evaluation_candidates(registry, candidates)
    rejects_before = registry.summary()["rejects"]
    second = reserve_evaluation_candidates(registry, candidates)
    assert first == second
    assert registry.summary()["rejects"] == rejects_before


def _trusted_candidate(*, sample_id: str, split: str, text: str, origin: str, tier: str = "gold") -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "task": "line_recognition",
        "text_sha256": text,
        "visual_sha256": "",
        "writer_id": "",
        "source_document": "",
        "source_page": "",
        "data_tier": tier,
        "sample_origin": origin,
    }


def test_human_validation_owns_duplicate_over_other_real_test(tmp_path) -> None:
    registry = DedupRegistry(tmp_path / "human-priority.sqlite", build_fingerprint="fp")
    report = reserve_evaluation_candidates(registry, [
        _trusted_candidate(sample_id="real-test", split="test", text="same", origin="real"),
        _trusted_candidate(sample_id="human-validation", split="validation", text="same", origin="human"),
    ])
    assert report["reserved"] == 1
    assert report["rejects"] == {"evaluation_text_owned_by_validation": 1}
    owner = registry.db.execute(
        "SELECT split,sample_id FROM evaluation_text_owners WHERE text_sha256=? AND data_tier='gold'",
        ("same",),
    ).fetchone()
    assert owner == ("validation", "human-validation")


def test_real_validation_owns_duplicate_over_synthetic_test(tmp_path) -> None:
    registry = DedupRegistry(tmp_path / "real-priority.sqlite", build_fingerprint="fp")
    report = reserve_evaluation_candidates(registry, [
        _trusted_candidate(sample_id="synthetic-test", split="test_synthetic", text="same", origin="synthetic"),
        _trusted_candidate(sample_id="real-validation", split="validation", text="same", origin="real"),
    ])
    assert report["reserved"] == 1
    assert report["rejects"] == {"evaluation_text_owned_by_validation": 1}
