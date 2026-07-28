from __future__ import annotations

import collections
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .architecture import ArchitectureSegment, ArchitectureState
from .augment import LINE_PROFILES
from .builder import _relative_artifact_path
from .identity import stable_token
from .ingest import make_sample_row
from .registry import DedupRegistry
from .unicode_utils import namespace_key


def _seed(*parts: object) -> int:
    return int(hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def _segment_key(segment: ArchitectureSegment) -> str:
    return f"{segment.document_id}:{segment.source_line}:{segment.segment_index}"


def _extra_variant_selected(*, key: str, text_sha256: str, rate: float) -> bool:
    """Choose extra train renders deterministically, independent of processing order."""
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    token = _seed("architecture-extra-variant-selection-v12", key, text_sha256)
    return token / float((1 << 64) - 1) < rate


def _profile_for_variant(*, primary_seed: int, variant_seed: int, variant_index: int) -> str:
    primary_index = primary_seed % len(LINE_PROFILES)
    if variant_index == 0 or len(LINE_PROFILES) == 1:
        return LINE_PROFILES[primary_index]
    # Force an alternate profile instead of accidentally drawing the same variant twice.
    offset = 1 + variant_seed % (len(LINE_PROFILES) - 1)
    return LINE_PROFILES[(primary_index + offset) % len(LINE_PROFILES)]


def _render_visible_line(
    *, renderer, segment: ArchitectureSegment, key: str, profile: str, seed: int, variant_index: int
):
    try:
        rendered = renderer.render_line(
            segment.text, profile=profile, seed=seed, split=segment.split, rashi=False
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"gold architecture segment cannot be rendered: {key}:v{variant_index}: {exc}"
        ) from exc
    if rendered.visibility_fraction < 0.42:
        rendered = renderer.render_line(
            segment.text,
            profile="clean_digital",
            seed=seed,
            split=segment.split,
            rashi=False,
        )
    if rendered.visibility_fraction < 0.42:
        raise RuntimeError(
            f"gold architecture segment cannot be rendered visibly: {key}:v{variant_index}: "
            f"visibility={rendered.visibility_fraction:.4f}"
        )
    return rendered


def _sample_row(
    *,
    segment: ArchitectureSegment,
    key: str,
    rendered,
    architecture_revision: str,
    variant_index: int,
) -> dict[str, Any]:
    if variant_index == 0:
        sample_id = f"arch-{stable_token(key, segment.text_sha256)}"
        source_id = key
        variant_role = "primary"
    else:
        sample_id = f"arch-{stable_token(key, segment.text_sha256, 'variant', variant_index)}"
        source_id = f"{key}:variant:{variant_index}"
        variant_role = "extra_train"

    augmentation = dict(rendered.metadata)
    augmentation.update({"variant_index": variant_index, "variant_role": variant_role})
    return make_sample_row(
        image_bytes=rendered.to_bytes(),
        image_path=f"{sample_id}.webp",
        text=segment.text,
        sample_id=sample_id,
        split=segment.split,
        task="line_recognition",
        granularity="line",
        modality="print",
        data_tier="gold",
        is_synthetic=True,
        sample_origin="synthetic",
        label_source="synthetic_ground_truth",
        label_trust="gold",
        provenance_reason="born_digital_text_rendered_by_pinned_pipeline",
        quality_tier="A",
        source_repo="ssdataanalysis/hebrew-architecture-corpus",
        source_revision=architecture_revision,
        source_path=f"txt/{segment.document_id}.txt",
        source_split=segment.split,
        source_id=source_id,
        source_document=namespace_key(
            "ssdataanalysis/hebrew-architecture-corpus", "document", segment.document_id
        ),
        font_family=rendered.font.family,
        font_style=rendered.font.style,
        font_sha256=rendered.font.sha256,
        augmentation=augmentation,
        provenance={
            "document_id": segment.document_id,
            "source_line": segment.source_line,
            "segment_index": segment.segment_index,
            "origin": segment.origin,
            "metadata": segment.metadata,
            "variant_index": variant_index,
            "variant_role": variant_role,
        },
    )


def _accept_architecture_row(
    *,
    row: dict[str, Any],
    source_key: str,
    registry: DedupRegistry,
    writer,
    config: dict[str, Any],
) -> tuple[bool, str]:
    decision = registry.accept_sample(
        sample_id=row["sample_id"],
        split=row["split"],
        task=row["task"],
        byte_sha256=row["image_sha256"],
        visual_sha256=row["visual_sha256"],
        text_sha256=row["text_sha256"],
        writer_key="",
        document_key=row["source_document"],
        source_key=source_key,
        text_cap=int(config["text_variant_caps"]["architecture"]),
        data_tier="gold",
        sample_origin="synthetic",
    )
    if decision.accepted:
        writer.add(row)
        return True, "accepted"
    if decision.reason == "train_text_reserved_by_evaluation":
        return False, "evaluation_reserved"
    return False, "duplicate"


def process_architecture_chunk(
    *,
    source_key: str,
    segments: Iterable[ArchitectureSegment],
    renderer,
    registry: DedupRegistry,
    writer,
    output_root: str | Path,
    config: dict[str, Any],
    architecture_revision: str,
) -> dict[str, Any]:
    output_root = Path(output_root)
    if registry.source_is_complete(source_key):
        report = registry.source_report(source_key)
        if report is None:
            raise RuntimeError(f"complete architecture source lacks report: {source_key}")
        return report

    counts: collections.Counter[str] = collections.Counter()
    extra_counts: collections.Counter[str] = collections.Counter()
    accepted = 0
    primary_variants = 0
    extra_variants = 0
    extra_variant_selected = 0
    segment_count = 0
    artifacts: list[Any] = []
    extra_rate = float(config["architecture_extra_variant_rate"])
    try:
        with registry.source_transaction(source_key):
            for segment in segments:
                segment_count += 1
                key = _segment_key(segment)
                if segment.state != ArchitectureState.GOLD:
                    outcome = {
                        ArchitectureState.QUARANTINE: "quarantined",
                        ArchitectureState.DUPLICATE: "duplicate",
                    }.get(segment.state, "rejected")
                    counts[outcome] += 1
                    registry.record_architecture_segment(
                        segment_key=key,
                        document_id=segment.document_id,
                        source_line=segment.source_line,
                        segment_index=segment.segment_index,
                        text_sha256=segment.text_sha256,
                        source_state=segment.state.value,
                        outcome=outcome,
                        reason=segment.reason,
                        split=segment.split,
                        sample_id="",
                    )
                    continue

                primary_seed = _seed("architecture-natural-v12", key, segment.text_sha256)
                primary_profile = _profile_for_variant(
                    primary_seed=primary_seed,
                    variant_seed=primary_seed,
                    variant_index=0,
                )
                primary_render = _render_visible_line(
                    renderer=renderer,
                    segment=segment,
                    key=key,
                    profile=primary_profile,
                    seed=primary_seed,
                    variant_index=0,
                )
                primary_row = _sample_row(
                    segment=segment,
                    key=key,
                    rendered=primary_render,
                    architecture_revision=architecture_revision,
                    variant_index=0,
                )
                primary_accepted, outcome = _accept_architecture_row(
                    row=primary_row,
                    source_key=source_key,
                    registry=registry,
                    writer=writer,
                    config=config,
                )
                if primary_accepted:
                    accepted += 1
                    primary_variants += 1
                counts[outcome] += 1
                registry.record_architecture_segment(
                    segment_key=key,
                    document_id=segment.document_id,
                    source_line=segment.source_line,
                    segment_index=segment.segment_index,
                    text_sha256=segment.text_sha256,
                    source_state=segment.state.value,
                    outcome=outcome,
                    reason="accepted_primary_render" if primary_accepted else outcome,
                    split=segment.split,
                    sample_id=primary_row["sample_id"] if primary_accepted else "",
                )

                # Evaluation sets remain one canonical image per exact line. Extra
                # synthetic diversity is training-only and can never inflate a benchmark.
                if not primary_accepted or segment.split != "train":
                    continue
                if not _extra_variant_selected(
                    key=key, text_sha256=segment.text_sha256, rate=extra_rate
                ):
                    continue

                extra_variant_selected += 1
                extra_seed = _seed(
                    "architecture-natural-v12-extra", key, segment.text_sha256, 1
                )
                extra_profile = _profile_for_variant(
                    primary_seed=primary_seed,
                    variant_seed=extra_seed,
                    variant_index=1,
                )
                extra_render = _render_visible_line(
                    renderer=renderer,
                    segment=segment,
                    key=key,
                    profile=extra_profile,
                    seed=extra_seed,
                    variant_index=1,
                )
                extra_row = _sample_row(
                    segment=segment,
                    key=key,
                    rendered=extra_render,
                    architecture_revision=architecture_revision,
                    variant_index=1,
                )
                extra_accepted, extra_outcome = _accept_architecture_row(
                    row=extra_row,
                    source_key=source_key,
                    registry=registry,
                    writer=writer,
                    config=config,
                )
                extra_counts[extra_outcome] += 1
                if extra_accepted:
                    accepted += 1
                    extra_variants += 1

            artifacts = writer.finish()
            written_rows = sum(int(item.rows) for item in artifacts)
            if written_rows != accepted:
                raise RuntimeError(f"architecture accepted/written mismatch: {accepted} != {written_rows}")
            artifact_rows = []
            for item in artifacts:
                rel = _relative_artifact_path(Path(item.path), output_root)
                registry.register_artifact(
                    source_key=source_key,
                    path=rel,
                    sha256=str(item.sha256),
                    rows=int(item.rows),
                    bytes_count=int(item.bytes),
                )
                artifact_rows.append({
                    "path": rel,
                    "sha256": str(item.sha256),
                    "rows": int(item.rows),
                    "bytes": int(item.bytes),
                })
            report = {
                "source_key": source_key,
                "family": "architecture",
                "segments": segment_count,
                "accepted": accepted,
                "primary_variants": primary_variants,
                "extra_variant_selected": extra_variant_selected,
                "extra_variants": extra_variants,
                "written_rows": written_rows,
                "outcomes": dict(sorted(counts.items())),
                "extra_variant_outcomes": dict(sorted(extra_counts.items())),
                "artifacts": artifact_rows,
            }
            registry.mark_source_complete(source_key, report)
        return report
    except Exception:
        writer.cleanup()
        raise
