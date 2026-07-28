from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow unavailable")


def test_corruption_suite_catches_parquet_row_and_manifest_tampering(tmp_path: Path) -> None:
    import io
    import json
    from PIL import Image
    from heocr_unified.corruption import run_corruption_suite
    from heocr_unified.ingest import make_sample_row
    from heocr_unified.writer import AtomicParquetWriter

    image = Image.new("L", (40, 16), 255)
    image.putpixel((10, 8), 0)
    buffer = io.BytesIO(); image.save(buffer, "PNG")
    revision = "a" * 40
    row = make_sample_row(
        image_bytes=buffer.getvalue(), image_path="x.png", text="שלום A-1", sample_id="s",
        split="train", task="line_recognition", granularity="line", modality="print",
        data_tier="gold", is_synthetic=True, sample_origin="synthetic",
        label_source="synthetic_ground_truth", label_trust="gold", quality_tier="A",
        source_repo="repo/source", source_revision=revision, source_path="source",
        source_split="train", source_id="1",
    )
    writer = AtomicParquetWriter(
        tmp_path, config_name="modern_print_lines", split="train", source_token="s", rows_per_shard=10
    )
    writer.add(row); writer.finish()
    (tmp_path / "BUILD_FINGERPRINT").write_text("f" * 64)
    (tmp_path / "FONT_MANIFEST.json").write_text(json.dumps({"fonts": []}))
    report = run_corruption_suite(tmp_path, source_revisions={"repo/source": revision})
    assert report["status"] == "PASS"
    assert report["test_count"] >= 6
    assert all(item["status"] == "PASS" for item in report["tests"])
