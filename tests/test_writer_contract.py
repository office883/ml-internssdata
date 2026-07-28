from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow unavailable")

from heocr_unified.writer import AtomicParquetWriter, verify_parquet_artifact


def test_atomic_writer_round_trips_and_refuses_overwrite(tmp_path: Path) -> None:
    writer = AtomicParquetWriter(tmp_path, config_name="modern_print_lines", split="train", source_token="src", rows_per_shard=2)
    writer.add({"sample_id":"a", "image":{"bytes":b"x", "path":"a.png"}, "text":"שלום"})
    writer.add({"sample_id":"b", "image":{"bytes":b"y", "path":"b.png"}, "text":"עולם"})
    artifacts = writer.finish()
    assert len(artifacts) == 1
    verify_parquet_artifact(artifacts[0].path, expected_rows=2, expected_sha256=artifacts[0].sha256)
    writer2 = AtomicParquetWriter(tmp_path, config_name="modern_print_lines", split="train", source_token="src", rows_per_shard=2)
    writer2.add({"sample_id":"c", "image":{"bytes":b"z", "path":"c.png"}, "text":"בדיקה"})
    with pytest.raises(FileExistsError):
        writer2.finish()
