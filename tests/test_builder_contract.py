from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import pytest

from heocr_unified.builder import process_source_rows
from heocr_unified.config import load_config
from heocr_unified.ingest import make_sample_row
from heocr_unified.registry import DedupRegistry


def _png(seed: int) -> bytes:
    image = Image.new("L", (32, 12), 255)
    image.putpixel((4 + seed % 20, 5), 0)
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def _row(sample_id: str, text: str, seed: int) -> dict:
    return make_sample_row(
        image_bytes=_png(seed), image_path=f"{sample_id}.png", text=text, sample_id=sample_id,
        split="train", task="line_recognition", granularity="line", modality="print",
        data_tier="gold", is_synthetic=True, sample_origin="synthetic", label_source="synthetic_ground_truth", label_trust="gold", quality_tier="A", source_repo="fixture/repo",
        source_revision="a" * 40, source_path="fixture.tar", source_split="train", source_id=sample_id,
    )


@dataclass(frozen=True)
class _Artifact:
    path: Path
    rows: int
    bytes: int
    sha256: str


class _MemoryWriter:
    def __init__(self, root: Path, fail: bool = False):
        self.root = root
        self.fail = fail
        self.rows: list[dict] = []
        self.artifacts: list[_Artifact] = []

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if self.fail and len(self.rows) == 2:
            raise RuntimeError("writer failure")

    def finish(self) -> list[_Artifact]:
        path = self.root / "artifact.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(row["sample_id"] for row in self.rows).encode()
        path.write_bytes(payload)
        artifact = _Artifact(path, len(self.rows), len(payload), hashlib.sha256(payload).hexdigest())
        self.artifacts.append(artifact)
        return list(self.artifacts)

    def cleanup(self) -> None:
        for artifact in self.artifacts:
            artifact.path.unlink(missing_ok=True)


def test_process_source_rows_commits_report_artifact_and_samples(tmp_path: Path) -> None:
    config = load_config(None, overrides={"work_dir": str(tmp_path), "upload": False})
    registry = DedupRegistry(tmp_path / "registry.sqlite", build_fingerprint="fp")
    writer = _MemoryWriter(tmp_path / "out")
    report = process_source_rows(
        source_key="fixture-source", rows=[_row("a", "שלום 1", 1), _row("b", "שלום 2", 2)],
        registry=registry, writer=writer, output_root=tmp_path, config=config,
    )
    assert report["scanned"] == 2
    assert report["accepted"] == 2
    assert report["written_rows"] == 2
    assert registry.sample_count() == 2
    assert registry.source_is_complete("fixture-source")
    assert len(registry.artifacts_for_source("fixture-source")) == 1


def test_process_source_rows_rolls_back_registry_and_deletes_files_on_failure(tmp_path: Path) -> None:
    config = load_config(None, overrides={"work_dir": str(tmp_path), "upload": False})
    registry = DedupRegistry(tmp_path / "registry.sqlite", build_fingerprint="fp")
    writer = _MemoryWriter(tmp_path / "out", fail=True)
    with pytest.raises(RuntimeError, match="writer failure"):
        process_source_rows(
            source_key="fixture-source", rows=[_row("a", "שלום 1", 1), _row("b", "שלום 2", 2)],
            registry=registry, writer=writer, output_root=tmp_path, config=config,
        )
    assert registry.sample_count() == 0
    assert not registry.source_is_complete("fixture-source")
    assert not list((tmp_path / "out").glob("*"))


def test_process_source_rows_rejects_later_train_text_reserved_by_eval(tmp_path: Path) -> None:
    config = load_config(None, overrides={"work_dir": str(tmp_path), "upload": False})
    registry = DedupRegistry(tmp_path / "registry.sqlite", build_fingerprint="fp")
    eval_row = _row("eval", "טקסט שמור", 10)
    eval_row["split"] = "validation"
    process_source_rows(
        source_key="eval-source", rows=[eval_row], registry=registry,
        writer=_MemoryWriter(tmp_path / "eval"), output_root=tmp_path, config=config,
    )
    train_row = _row("train", "טקסט שמור", 11)
    report = process_source_rows(
        source_key="train-source", rows=[train_row], registry=registry,
        writer=_MemoryWriter(tmp_path / "train"), output_root=tmp_path, config=config,
    )
    assert report["accepted"] == 0
    assert report["rejects"] == {"train_text_reserved_by_evaluation": 1}
    assert registry.sample_count() == 1
