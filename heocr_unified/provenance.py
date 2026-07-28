from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .sources import MappedSource, SourceTask


@dataclass(frozen=True)
class ProvenanceDecision:
    is_synthetic: bool
    data_tier: str
    sample_origin: str
    label_source: str
    label_trust: str
    reason: str
    recommended_sampling_weight: float


_SAMARITAN_DATASETS = {
    "johnlockejrr/samaritan_v1",
    "samaritan-ai/samaritan_hebrew_LightOnOcr",
}

_SAMARITAN_MODALITIES = {
    "historical_samaritan_handwritten_line",
    "historical_samaritan_lightonocr_line",
}

_REAL_DATASETS = {
    "zenodo/pinkas_dataset",
    "sivan22/hebrew-handwritten-dataset",
}

_REAL_MODALITIES = {
    "handwriting_real_character",
    "historical_hebrew_handwritten_line",
}

_SYNTHETIC_MODALITY_TOKENS = (
    "synthetic",
    "printed_online",
    "handwriting_synthetic",
)

_SYNTHETIC_IMAGE_KINDS = {
    "materialized_synthetic",
    "generated",
    "rendered_synthetic",
}


def _quality(metadata: Mapping[str, Any]) -> str:
    value = str(metadata.get("quality_tier") or metadata.get("data_tier") or "").strip().upper()
    return value or "UNKNOWN"


def _synthetic_hint(metadata: Mapping[str, Any]) -> bool:
    modality = str(metadata.get("modality") or "").lower()
    image_kind = str(metadata.get("image_kind") or "").lower()
    source_dataset = str(metadata.get("source_dataset") or "").lower()
    return (
        any(token in modality for token in _SYNTHETIC_MODALITY_TOKENS)
        or image_kind in _SYNTHETIC_IMAGE_KINDS
        or "synth" in source_dataset
        or bool(metadata.get("render_profile"))
        or bool(metadata.get("template_signature"))
    )


def _decision(
    *,
    is_synthetic: bool,
    data_tier: str,
    sample_origin: str,
    label_source: str,
    reason: str,
    weight: float,
) -> ProvenanceDecision:
    if data_tier not in {"gold", "extended", "quarantine"}:
        raise ValueError(f"invalid data tier: {data_tier}")
    return ProvenanceDecision(
        is_synthetic=is_synthetic,
        data_tier=data_tier,
        sample_origin=sample_origin,
        label_source=label_source,
        label_trust=data_tier,
        reason=reason,
        recommended_sampling_weight=float(weight),
    )


def classify_row_provenance(
    task: SourceTask,
    mapped: MappedSource,
    metadata: Mapping[str, Any],
) -> ProvenanceDecision:
    """Classify one source row conservatively.

    The source configuration is only a hint. The row-level provenance fields are
    authoritative when present. Unknown or contradictory provenance is never
    promoted into the default gold training configuration.
    """

    if task.family == "foundation":
        return _decision(
            is_synthetic=True,
            data_tier="gold",
            sample_origin="synthetic",
            label_source="synthetic_ground_truth",
            reason="foundation_verified_renderer",
            weight=1.0,
        )

    if task.family == "htr":
        stage = task.path.split("/", 1)[0]
        if stage == "stage3_human_finetune":
            return _decision(
                is_synthetic=False,
                data_tier="gold",
                sample_origin="human",
                label_source="human_transcription",
                reason="curated_human_htr",
                weight=float(metadata.get("recommended_sampling_weight") or 1.0),
            )
        if stage == "stage2_diffusion_augmentation":
            return _decision(
                is_synthetic=True,
                data_tier="extended",
                sample_origin="diffusion",
                label_source="prompt_conditioned_label",
                reason="diffusion_not_pixel_verified",
                weight=min(float(metadata.get("recommended_sampling_weight") or 0.35), 0.5),
            )
        return _decision(
            is_synthetic=True,
            data_tier="gold",
            sample_origin="synthetic",
            label_source="synthetic_ground_truth",
            reason="curated_synthetic_htr",
            weight=float(metadata.get("recommended_sampling_weight") or 0.75),
        )

    if task.family != "ocr":
        return _decision(
            is_synthetic=bool(mapped.synthetic),
            data_tier="quarantine",
            sample_origin="unknown",
            label_source="unknown",
            reason="unsupported_source_family",
            weight=0.0,
        )

    modality = str(metadata.get("modality") or mapped.modality).strip()
    modality_lower = modality.lower()
    source_dataset = str(metadata.get("source_dataset") or "").strip()
    image_kind = str(metadata.get("image_kind") or "").strip().lower()
    quality = _quality(metadata)

    # Samaritan-script images use Hebrew Unicode labels, but their glyph shapes are
    # not ordinary square Hebrew. They are useful for opt-in historical research,
    # yet must never silently teach the default Hebrew OCR model that Samaritan
    # glyphs are modern Hebrew print or handwriting.
    if source_dataset in _SAMARITAN_DATASETS or modality in _SAMARITAN_MODALITIES:
        if quality in {"C", "Q"}:
            return _decision(
                is_synthetic=False,
                data_tier="quarantine",
                sample_origin="real",
                label_source="source_transcription",
                reason=f"samaritan_script_quality_{quality.lower()}",
                weight=0.0,
            )
        return _decision(
            is_synthetic=False,
            data_tier="extended",
            sample_origin="real",
            label_source="source_transcription",
            reason="samaritan_script_opt_in_only",
            weight=0.35 if quality in {"A", "UNKNOWN"} else 0.25,
        )

    known_real = source_dataset in _REAL_DATASETS or modality in _REAL_MODALITIES
    if known_real:
        if quality in {"C", "Q"}:
            return _decision(
                is_synthetic=False,
                data_tier="quarantine",
                sample_origin="real",
                label_source="source_transcription",
                reason=f"real_quality_{quality.lower()}",
                weight=0.0,
            )
        return _decision(
            is_synthetic=False,
            data_tier="gold" if quality in {"A", "UNKNOWN"} else "extended",
            sample_origin="real",
            label_source="source_transcription",
            reason="known_real_source",
            weight=1.0 if quality in {"A", "UNKNOWN"} else 0.6,
        )

    row_synthetic_hint = _synthetic_hint(metadata)
    if (
        modality_lower == "existing_rendered_or_scanned"
        or image_kind == "source_archive_uri"
    ) and not row_synthetic_hint:
        return _decision(
            is_synthetic=False,
            data_tier="quarantine",
            sample_origin="unknown",
            label_source="unknown",
            reason="ambiguous_rendered_or_scanned_provenance",
            weight=0.0,
        )

    synthetic = row_synthetic_hint or bool(mapped.synthetic)
    if synthetic:
        if quality in {"C", "Q"}:
            tier = "quarantine"
            weight = 0.0
        elif quality == "B":
            tier = "extended"
            weight = 0.5
        else:
            tier = "gold" if mapped.data_tier == "gold" else "extended"
            weight = 0.8 if tier == "gold" else 0.4
        return _decision(
            is_synthetic=True,
            data_tier=tier,
            sample_origin="synthetic",
            label_source="synthetic_ground_truth",
            reason=f"synthetic_provenance_quality_{quality.lower()}",
            weight=weight,
        )

    return _decision(
        is_synthetic=False,
        data_tier="quarantine",
        sample_origin="unknown",
        label_source="unknown",
        reason="unclassified_provenance",
        weight=0.0,
    )
