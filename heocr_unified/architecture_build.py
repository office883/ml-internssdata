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
    accepted = 0
    segment_count = 0
    artifacts: list[Any] = []
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

                seed = _seed("architecture-natural-v9", key, segment.text_sha256)
                profile = LINE_PROFILES[seed % len(LINE_PROFILES)]
                try:
                    rendered = renderer.render_line(
                        segment.text, profile=profile, seed=seed, split=segment.split, rashi=False
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"gold architecture segment cannot be rendered: {key}: {exc}"
                    ) from exc
                if rendered.visibility_fraction < 0.42:
                    rendered = renderer.render_line(
                        segment.text, profile="clean_digital", seed=seed, split=segment.split, rashi=False
                    )
                if rendered.visibility_fraction < 0.42:
                    raise RuntimeError(
                        f"gold architecture segment cannot be rendered visibly: {key}: "
                        f"visibility={rendered.visibility_fraction:.4f}"
                    )

                sample_id = f"arch-{stable_token(key, segment.text_sha256)}"
                image_bytes = rendered.to_bytes()
                row = make_sample_row(
                    image_bytes=image_bytes,
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
                    source_id=key,
                    source_document=namespace_key(
                        "ssdataanalysis/hebrew-architecture-corpus", "document", segment.document_id
                    ),
                    font_family=rendered.font.family,
                    font_style=rendered.font.style,
                    font_sha256=rendered.font.sha256,
                    augmentation=rendered.metadata,
                    provenance={
                        "document_id": segment.document_id,
                        "source_line": segment.source_line,
                        "segment_index": segment.segment_index,
                        "origin": segment.origin,
                        "metadata": segment.metadata,
                    },
                )
                decision = registry.accept_sample(
                    sample_id=sample_id,
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
                    accepted += 1
                    outcome = "accepted"
                elif decision.reason == "train_text_reserved_by_evaluation":
                    outcome = "evaluation_reserved"
                else:
                    outcome = "duplicate"
                counts[outcome] += 1
                registry.record_architecture_segment(
                    segment_key=key,
                    document_id=segment.document_id,
                    source_line=segment.source_line,
                    segment_index=segment.segment_index,
                    text_sha256=segment.text_sha256,
                    source_state=segment.state.value,
                    outcome=outcome,
                    reason=decision.reason,
                    split=segment.split,
                    sample_id=sample_id if decision.accepted else "",
                )

            artifacts = writer.finish()
            written_rows = sum(int(item.rows) for item in artifacts)
            if written_rows != accepted:
                raise RuntimeError(f"architecture accepted/written mismatch: {accepted} != {written_rows}")
            artifact_rows = []
            for item in artifacts:
                rel = _relative_artifact_path(Path(item.path), output_root)
                registry.register_artifact(
                    source_key=source_key, path=rel, sha256=str(item.sha256),
                    rows=int(item.rows), bytes_count=int(item.bytes),
                )
                artifact_rows.append({
                    "path": rel, "sha256": str(item.sha256), "rows": int(item.rows), "bytes": int(item.bytes)
                })
            report = {
                "source_key": source_key,
                "family": "architecture",
                "segments": segment_count,
                "accepted": accepted,
                "written_rows": written_rows,
                "outcomes": dict(sorted(counts.items())),
                "artifacts": artifact_rows,
            }
            registry.mark_source_complete(source_key, report)
        return report
    except Exception:
        writer.cleanup()
        raise
