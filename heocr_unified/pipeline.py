from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from .builder import process_source_rows
from .identity import stable_token
from .ingest import iter_htr_parquet, iter_webdataset_source
from .registry import DedupRegistry
from .sources import SourceTask, classify_source_task
from .writer import AtomicParquetWriter, TieredParquetRouter, verify_parquet_artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_completed_source(
    source_key: str,
    *,
    registry: DedupRegistry,
    output_root: str | Path,
    artifact_verifier: Callable[..., Any] = verify_parquet_artifact,
) -> dict[str, Any]:
    report = registry.source_report(source_key)
    if report is None or not registry.source_is_complete(source_key):
        raise RuntimeError(f"source is not complete: {source_key}")
    artifacts = registry.artifacts_for_source(source_key)
    if artifacts != report.get("artifacts", []):
        raise RuntimeError(f"source artifact registry/report mismatch: {source_key}")
    written = 0
    root = Path(output_root)
    for row in artifacts:
        path = root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"artifact size mismatch: {row['path']}")
        artifact_verifier(path, expected_rows=int(row["rows"]), expected_sha256=str(row["sha256"]))
        written += int(row["rows"])
    if written != int(report.get("written_rows", -1)):
        raise RuntimeError(f"source written-row mismatch: {source_key}")
    return report


def process_downloaded_task(
    *,
    task: SourceTask,
    local_path: str | Path,
    registry: DedupRegistry,
    output_root: str | Path,
    config: dict[str, Any],
    limit: int | None = None,
    writer_factory=AtomicParquetWriter,
) -> dict[str, Any]:
    output_root = Path(output_root)
    if registry.source_is_complete(task.source_key):
        return verify_completed_source(task.source_key, registry=registry, output_root=output_root)
    source = Path(local_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    if task.size and size != int(task.size):
        raise ValueError(f"downloaded source size mismatch: {task.path}: {size} != {task.size}")
    source_sha = _sha256(source)
    mapped = classify_source_task(task)
    token = stable_token(task.source_key, length=20)
    if writer_factory is AtomicParquetWriter:
        writer = TieredParquetRouter(
            output_root,
            base_config_name=mapped.output_config,
            split=mapped.output_split,
            source_token=token,
            rows_per_shard=int(config["rows_per_shard"]),
        )
    else:
        writer = writer_factory(
            output_root=output_root,
            config_name=mapped.output_config,
            split=mapped.output_split,
            source_token=token,
            rows_per_shard=int(config["rows_per_shard"]),
        )
    if task.family == "htr":
        rows = iter_htr_parquet(str(source), task=task, mapped=mapped, limit=limit)
    elif task.family in {"ocr", "foundation"}:
        rows = iter_webdataset_source(str(source), task=task, mapped=mapped, limit=limit)
    else:
        raise ValueError(f"unsupported visual source family: {task.family}")
    return process_source_rows(
        source_key=task.source_key,
        rows=rows,
        registry=registry,
        writer=writer,
        output_root=output_root,
        config=config,
        report_metadata={
            "family": task.family,
            "repo_id": task.repo_id,
            "revision": task.revision,
            "source_path": task.path,
            "source_split": task.split,
            "source_bytes": size,
            "source_sha256": source_sha,
            "output_config": mapped.output_config,
            "output_split": mapped.output_split,
            "data_tier": mapped.data_tier,
        },
    )
