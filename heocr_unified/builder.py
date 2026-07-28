from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterable, Protocol

from .registry import DedupRegistry


class WriterProtocol(Protocol):
    artifacts: list[Any]

    def add(self, row: dict[str, Any]) -> None: ...
    def finish(self) -> list[Any]: ...
    def cleanup(self) -> None: ...


def _text_cap(row: dict[str, Any], config: dict[str, Any]) -> int:
    caps = config["text_variant_caps"]
    source_path = str(row.get("source_path", "")).lower()
    source_repo = str(row.get("source_repo", "")).lower()
    if not bool(row.get("is_synthetic")):
        key = "human" if "htr" in source_repo or row.get("writer_id") else "real"
    elif "diffusion" in source_path:
        key = "diffusion"
    elif "architecture" in source_repo:
        key = "architecture"
    else:
        key = "synthetic"
    return int(caps[key])


def _relative_artifact_path(path: Path, output_root: Path) -> str:
    try:
        return path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact is outside output root: {path}") from exc


def process_source_rows(
    *,
    source_key: str,
    rows: Iterable[dict[str, Any]],
    registry: DedupRegistry,
    writer: WriterProtocol,
    output_root: str | Path,
    config: dict[str, Any],
    report_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process one immutable source unit atomically.

    Registry changes commit only after every accepted row has been written and all
    artifacts have been produced. On any error both registry changes and filesystem
    artifacts created by this writer are removed.
    """
    output_root = Path(output_root)
    if registry.source_is_complete(source_key):
        report = registry.source_report(source_key)
        if report is None:
            raise RuntimeError(f"complete source has no report: {source_key}")
        return report

    scanned = 0
    accepted = 0
    rejects: collections.Counter[str] = collections.Counter()
    artifacts: list[Any] = []
    try:
        with registry.source_transaction(source_key):
            for row in rows:
                scanned += 1
                decision = registry.accept_sample(
                    sample_id=str(row["sample_id"]),
                    split=str(row["split"]),
                    task=str(row["task"]),
                    byte_sha256=str(row["image_sha256"]),
                    visual_sha256=str(row["visual_sha256"]),
                    text_sha256=str(row["text_sha256"]),
                    writer_key=str(row.get("writer_id") or ""),
                    document_key=str(row.get("source_document") or ""),
                    page_key=str(row.get("source_page") or ""),
                    source_key=source_key,
                    text_cap=_text_cap(row, config),
                    data_tier=str(row.get("data_tier") or "quarantine"),
                    sample_origin=str(row.get("sample_origin") or "unknown"),
                )
                if not decision.accepted:
                    rejects[decision.reason] += 1
                    continue
                writer.add(row)
                accepted += 1

            artifacts = writer.finish()
            written_rows = sum(int(item.rows) for item in artifacts)
            if written_rows != accepted:
                raise RuntimeError(
                    f"accepted/written row mismatch for {source_key}: {accepted} != {written_rows}"
                )
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
                artifact_rows.append(
                    {"path": rel, "sha256": str(item.sha256), "rows": int(item.rows), "bytes": int(item.bytes)}
                )
            report = {
                **(report_metadata or {}),
                "source_key": source_key,
                "scanned": scanned,
                "accepted": accepted,
                "rejected": scanned - accepted,
                "rejects": dict(sorted(rejects.items())),
                "written_rows": written_rows,
                "artifacts": artifact_rows,
            }
            registry.mark_source_complete(source_key, report)
        return report
    except Exception:
        writer.cleanup()
        raise
