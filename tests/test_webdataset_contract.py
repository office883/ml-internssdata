from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from heocr_unified.webdataset import WebDatasetError, iter_webdataset_strict


def _tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_yields_exact_image_json_pair(tmp_path: Path) -> None:
    path = tmp_path / "a.tar"
    _tar(path, [("x.png", b"image"), ("x.json", json.dumps({"text":"שלום"}).encode())])
    rows = list(iter_webdataset_strict(path))
    assert rows == [{"key":"x", "image_extension":"png", "image_bytes":b"image", "metadata":{"text":"שלום"}}]


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "a.tar"
    _tar(path, [("x.png", b"image"), ("x.json", b"{")])
    with pytest.raises(WebDatasetError, match="JSON"):
        list(iter_webdataset_strict(path))


def test_rejects_missing_pair(tmp_path: Path) -> None:
    path = tmp_path / "a.tar"
    _tar(path, [("x.png", b"image")])
    with pytest.raises(WebDatasetError, match="unpaired"):
        list(iter_webdataset_strict(path))


def test_rejects_duplicate_member(tmp_path: Path) -> None:
    path = tmp_path / "a.tar"
    _tar(path, [("x.png", b"one"), ("dir/x.png", b"two"), ("x.json", b"{}")])
    with pytest.raises(WebDatasetError, match="duplicate"):
        list(iter_webdataset_strict(path))
