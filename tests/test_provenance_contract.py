from __future__ import annotations

from heocr_unified.provenance import classify_row_provenance
from heocr_unified.sources import SourceTask, classify_source_task


def _task(family: str, path: str, split: str = "train") -> SourceTask:
    return SourceTask(family, f"fixture/{family}", path, split, "a" * 40, 1)


def test_human_htr_is_real_gold() -> None:
    task = _task("htr", "stage3_human_finetune/train-00000.parquet")
    decision = classify_row_provenance(task, classify_source_task(task), {"human_source": "matan_primary"})
    assert decision.sample_origin == "human"
    assert decision.label_source == "human_transcription"
    assert decision.data_tier == "gold"
    assert decision.is_synthetic is False


def test_diffusion_htr_is_extended() -> None:
    task = _task("htr", "stage2_diffusion_augmentation/train-00000.parquet")
    decision = classify_row_provenance(task, classify_source_task(task), {"cer": 0.0})
    assert decision.sample_origin == "diffusion"
    assert decision.data_tier == "extended"
    assert decision.is_synthetic is True


def test_known_real_historical_handwriting_is_gold() -> None:
    task = _task("ocr", "webdataset/historical_handwriting_lines/train/a.tar")
    decision = classify_row_provenance(
        task,
        classify_source_task(task),
        {
            "modality": "historical_hebrew_handwritten_line",
            "source_dataset": "zenodo/pinkas_dataset",
            "image_kind": "materialized_online",
            "quality_tier": "A",
            "source_license": "cc-by-4.0",
        },
    )
    assert decision.sample_origin == "real"
    assert decision.data_tier == "gold"
    assert decision.is_synthetic is False


def test_quality_b_synthetic_material_is_extended() -> None:
    task = _task("ocr", "webdataset/modern_print_lines/train/a.tar")
    decision = classify_row_provenance(
        task,
        classify_source_task(task),
        {
            "modality": "printed",
            "image_kind": "materialized_synthetic",
            "quality_tier": "B",
            "source_dataset": "hebrew_ocr_1m_foundation_v1",
            "render_profile": "print",
        },
    )
    assert decision.sample_origin == "synthetic"
    assert decision.data_tier == "extended"
    assert decision.is_synthetic is True


def test_unknown_existing_rendered_or_scanned_is_quarantine() -> None:
    task = _task("ocr", "webdataset/modern_print_lines/train/a.tar")
    decision = classify_row_provenance(
        task,
        classify_source_task(task),
        {
            "modality": "existing_rendered_or_scanned",
            "image_kind": "source_archive_uri",
            "quality_tier": "B",
            "source_dataset": "unknown_source",
        },
    )
    assert decision.data_tier == "quarantine"
    assert decision.recommended_sampling_weight == 0.0


def test_foundation_is_synthetic_gold() -> None:
    task = _task("foundation", "shards/train-modern-natural-000.tar")
    decision = classify_row_provenance(task, classify_source_task(task), {"profile": "modern_natural"})
    assert decision.sample_origin == "synthetic"
    assert decision.label_source == "synthetic_ground_truth"
    assert decision.data_tier == "gold"


def test_samaritan_handwriting_is_isolated_in_extended_not_gold() -> None:
    task = _task("ocr", "webdataset/historical_handwriting_lines/train/a.tar")
    for metadata in (
        {
            "modality": "historical_samaritan_handwritten_line",
            "source_dataset": "johnlockejrr/samaritan_v1",
            "image_kind": "materialized_online",
            "quality_tier": "A",
            "source_license": "mit",
        },
        {
            "modality": "historical_samaritan_lightonocr_line",
            "source_dataset": "samaritan-ai/samaritan_hebrew_LightOnOcr",
            "image_kind": "materialized_online",
            "quality_tier": "A",
            "source_license": "cc-by-4.0",
        },
    ):
        decision = classify_row_provenance(task, classify_source_task(task), metadata)
        assert decision.sample_origin == "real"
        assert decision.data_tier == "extended"
        assert decision.is_synthetic is False
        assert decision.reason == "samaritan_script_opt_in_only"
        assert 0.0 < decision.recommended_sampling_weight < 1.0


def test_hebrew_historical_handwriting_remains_gold_after_samaritan_isolation() -> None:
    task = _task("ocr", "webdataset/historical_handwriting_lines/train/a.tar")
    decision = classify_row_provenance(
        task,
        classify_source_task(task),
        {
            "modality": "historical_hebrew_handwritten_line",
            "source_dataset": "zenodo/pinkas_dataset",
            "image_kind": "materialized_online",
            "quality_tier": "A",
        },
    )
    assert decision.data_tier == "gold"
    assert decision.reason == "known_real_source"
