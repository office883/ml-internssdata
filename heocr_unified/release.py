from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_release_manifest(root: str | Path, *, exclude_names: set[str] | None = None) -> dict:
    root = Path(root)
    excluded = {"RELEASE_MANIFEST.json", "LOCAL_READY.json", "REMOTE_READY.json"}
    if exclude_names:
        excluded.update(exclude_names)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        rel = path.relative_to(root).as_posix()
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "version": 1,
        "excluded_names": sorted(excluded),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def verify_release_manifest(root: str | Path, manifest: dict) -> None:
    root = Path(root)
    if int(manifest.get("version", -1)) != 1:
        raise ValueError("unsupported release manifest version")
    rows = manifest.get("files", [])
    if not isinstance(rows, list):
        raise ValueError("manifest files must be a list")
    expected_paths = {str(row["path"]) for row in rows}
    if len(expected_paths) != len(rows) or len(expected_paths) != int(manifest.get("file_count", -1)):
        raise ValueError("manifest file count mismatch")
    excluded = set(manifest.get("excluded_names") or {
        "RELEASE_MANIFEST.json", "LOCAL_READY.json", "REMOTE_READY.json"
    })
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing:
        raise ValueError(f"missing release files: {missing[:10]}")
    if extra:
        raise ValueError(f"unexpected release files: {extra[:10]}")
    actual_total = 0
    for row in rows:
        path = root / row["path"]
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"size mismatch: {row['path']}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"hash mismatch: {row['path']}")
        actual_total += path.stat().st_size
    if actual_total != int(manifest.get("total_bytes", -1)):
        raise ValueError("manifest total bytes mismatch")
