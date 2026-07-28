from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

from PIL import Image
import pytest

from heocr_unified.config import load_config
from heocr_unified.pipeline import process_downloaded_task, verify_completed_source
from heocr_unified.registry import DedupRegistry
from heocr_unified.sources import SourceTask


def _webdataset(path: Path) -> None:
    image = Image.new("L", (40, 14), 255)
    image.putpixel((20, 7), 0)
    buf = io.BytesIO(); image.save(buf, "PNG")
    metadata = json.dumps({"id":"one", "text":"שלום 17", "modality":"printed"}, ensure_ascii=False).encode()
    with tarfile.open(path, "w") as tf:
        for name, data in (("one.png", buf.getvalue()), ("one.json", metadata)):
            info = tarfile.TarInfo(name); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


class _Artifact:
    def __init__(self, path: Path, rows: int):
        self.path = path
        self.rows = rows
        self.bytes = path.stat().st_size
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()


class _Writer:
    def __init__(self, root: Path):
        self.root = root; self.rows=[]; self.artifacts=[]
    def add(self, row): self.rows.append(row)
    def finish(self):
        p=self.root/"a.bin"; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes("\n".join(r["sample_id"] for r in self.rows).encode())
        self.artifacts=[_Artifact(p,len(self.rows))]; return self.artifacts
    def cleanup(self):
        for a in self.artifacts: a.path.unlink(missing_ok=True)


def test_process_downloaded_webdataset_task_uses_real_strict_parser(tmp_path: Path) -> None:
    tar = tmp_path / "source.tar"; _webdataset(tar)
    task = SourceTask(
        "ocr", "fixture/ocr", "webdataset/modern_print_lines/train/a.tar", "train",
        revision="a"*40, size=tar.stat().st_size,
    )
    config = load_config(None, overrides={"work_dir":str(tmp_path), "upload":False})
    registry = DedupRegistry(tmp_path/"registry.sqlite", build_fingerprint="fp")
    report = process_downloaded_task(
        task=task, local_path=tar, registry=registry, output_root=tmp_path,
        config=config, writer_factory=lambda **_: _Writer(tmp_path/"out"),
    )
    assert report["accepted"] == 1
    assert report["source_bytes"] == tar.stat().st_size
    assert report["source_sha256"] == hashlib.sha256(tar.read_bytes()).hexdigest()


def test_verify_completed_source_fails_when_artifact_disappears(tmp_path: Path) -> None:
    tar = tmp_path / "source.tar"; _webdataset(tar)
    task = SourceTask("ocr","fixture/ocr","webdataset/modern_print_lines/train/a.tar","train",revision="a"*40,size=tar.stat().st_size)
    config = load_config(None, overrides={"work_dir":str(tmp_path), "upload":False})
    registry = DedupRegistry(tmp_path/"registry.sqlite", build_fingerprint="fp")
    process_downloaded_task(
        task=task, local_path=tar, registry=registry, output_root=tmp_path,
        config=config, writer_factory=lambda **_: _Writer(tmp_path/"out"),
    )
    (tmp_path/"out"/"a.bin").unlink()
    with pytest.raises(FileNotFoundError):
        verify_completed_source(task.source_key, registry=registry, output_root=tmp_path, artifact_verifier=lambda *a, **k: None)
