from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParquetArtifact:
    path: Path
    rows: int
    bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("task", pa.string(), nullable=False),
        pa.field("granularity", pa.string(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("data_tier", pa.string(), nullable=False),
        pa.field("is_synthetic", pa.bool_(), nullable=False),
        pa.field("sample_origin", pa.string(), nullable=False),
        pa.field("label_source", pa.string(), nullable=False),
        pa.field("label_trust", pa.string(), nullable=False),
        pa.field("provenance_reason", pa.string(), nullable=False),
        pa.field("quality_tier", pa.string(), nullable=False),
        pa.field("source_repo", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_document", pa.string(), nullable=False),
        pa.field("source_page", pa.string(), nullable=False),
        pa.field("writer_id", pa.string(), nullable=False),
        pa.field("image_sha256", pa.string(), nullable=False),
        pa.field("visual_sha256", pa.string(), nullable=False),
        pa.field("text_sha256", pa.string(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        pa.field("image_format", pa.string(), nullable=False),
        pa.field("font_family", pa.string(), nullable=False),
        pa.field("font_style", pa.string(), nullable=False),
        pa.field("font_sha256", pa.string(), nullable=False),
        pa.field("augmentation_json", pa.string(), nullable=False),
        pa.field("annotations_json", pa.string(), nullable=False),
        pa.field("provenance_json", pa.string(), nullable=False),
        pa.field("recommended_sampling_weight", pa.float32(), nullable=False),
    ])


_DEFAULTS: dict[str, Any] = {
    "split": "train", "task": "line_recognition", "granularity": "line", "modality": "print",
    "data_tier": "gold", "is_synthetic": False, "sample_origin": "unknown",
    "label_source": "unknown", "label_trust": "gold", "provenance_reason": "",
    "quality_tier": "", "source_repo": "", "source_revision": "",
    "source_path": "", "source_split": "", "source_id": "", "source_document": "", "source_page": "",
    "writer_id": "", "image_sha256": "", "visual_sha256": "", "text_sha256": "", "width": 0,
    "height": 0, "image_format": "", "font_family": "", "font_style": "", "font_sha256": "",
    "augmentation_json": "{}", "annotations_json": "[]", "provenance_json": "{}",
    "recommended_sampling_weight": 1.0,
}


def _canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(_DEFAULTS)
    output.update(row)
    if "sample_id" not in output or "text" not in output or "image" not in output:
        raise ValueError("sample_id, image, and text are required")
    image = output["image"]
    if isinstance(image, (bytes, bytearray, memoryview)):
        output["image"] = {"bytes": bytes(image), "path": "image"}
    elif isinstance(image, dict):
        output["image"] = {"bytes": bytes(image.get("bytes") or b""), "path": str(image.get("path") or "image")}
    else:
        raise ValueError("invalid image value")
    return output


def verify_parquet_artifact(path: str | Path, *, expected_rows: int, expected_sha256: str) -> ParquetArtifact:
    import pyarrow.parquet as pq
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256(path)
    if digest != expected_sha256:
        raise ValueError("parquet hash mismatch")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError("parquet row-count mismatch")
    expected_names = _schema().names
    if parquet.schema_arrow.names != expected_names:
        raise ValueError("parquet schema mismatch")
    return ParquetArtifact(path, expected_rows, path.stat().st_size, digest)


class AtomicParquetWriter:
    def __init__(self, output_root: str | Path, *, config_name: str, split: str, source_token: str, rows_per_shard: int = 1500):
        self.output_root = Path(output_root)
        self.config_name = config_name
        self.split = split
        self.source_token = source_token
        self.rows_per_shard = int(rows_per_shard)
        self.rows: list[dict[str, Any]] = []
        self.index = 0
        self.artifacts: list[ParquetArtifact] = []
        self.final_dir = self.output_root / "data" / config_name / split
        self.partial_dir = self.output_root / ".partial" / config_name / split
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.partial_dir.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(_canonical_row(row))
        if len(self.rows) >= self.rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        name = f"{self.source_token}-{self.index:05d}.parquet"
        final = self.final_dir / name
        temp = self.partial_dir / f"{name}.tmp"
        if final.exists():
            raise FileExistsError(final)
        if temp.exists():
            temp.unlink()
        table = pa.Table.from_pylist(self.rows, schema=_schema())
        pq.write_table(table, temp, compression="zstd", compression_level=6, row_group_size=min(512, len(self.rows)))
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = _sha256(temp)
        rows = len(self.rows)
        verify_parquet_artifact(temp, expected_rows=rows, expected_sha256=digest)
        os.replace(temp, final)
        self.artifacts.append(ParquetArtifact(final, rows, final.stat().st_size, digest))
        self.rows.clear()
        self.index += 1

    def finish(self) -> list[ParquetArtifact]:
        self._flush()
        return list(self.artifacts)

    def cleanup(self) -> None:
        for artifact in list(self.artifacts):
            artifact.path.unlink(missing_ok=True)
        self.artifacts.clear()
        for path in self.partial_dir.glob(f"{self.source_token}-*.parquet.tmp"):
            path.unlink(missing_ok=True)


class TieredParquetRouter:
    """Route rows into isolated gold, extended and quarantine configs.

    Gold keeps the canonical base config name for convenient training. Lower
    trust material is physically separated so a glob for the default config can
    never ingest it by accident.
    """

    _SUFFIX = {
        "gold": "",
        "extended": "_extended",
        "quarantine": "_quarantine",
    }

    def __init__(
        self,
        output_root: str | Path,
        *,
        base_config_name: str,
        split: str,
        source_token: str,
        rows_per_shard: int = 1500,
        writer_cls=AtomicParquetWriter,
    ):
        self.output_root = Path(output_root)
        self.base_config_name = str(base_config_name)
        self.split = str(split)
        self.source_token = str(source_token)
        self.rows_per_shard = int(rows_per_shard)
        self.writer_cls = writer_cls
        self._writers: dict[tuple[str, str], Any] = {}
        self.artifacts: list[ParquetArtifact] = []

    def _writer(self, tier: str, split: str):
        if tier not in self._SUFFIX:
            raise ValueError(f"unknown data tier: {tier}")
        key = (tier, split)
        writer = self._writers.get(key)
        if writer is None:
            writer = self.writer_cls(
                self.output_root,
                config_name=self.base_config_name + self._SUFFIX[tier],
                split=split,
                source_token=f"{self.source_token}-{tier}-{split}",
                rows_per_shard=self.rows_per_shard,
            )
            self._writers[key] = writer
        return writer

    def add(self, row: dict[str, Any]) -> None:
        tier = str(row.get("data_tier") or "quarantine")
        split = str(row.get("split") or self.split)
        self._writer(tier, split).add(row)

    def finish(self) -> list[ParquetArtifact]:
        artifacts: list[ParquetArtifact] = []
        for key in sorted(self._writers):
            writer = self._writers[key]
            artifacts.extend(writer.finish())
        self.artifacts = artifacts
        return list(artifacts)

    def cleanup(self) -> None:
        for writer in self._writers.values():
            writer.cleanup()
        self.artifacts.clear()
