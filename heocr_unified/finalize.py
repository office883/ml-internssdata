from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .metadata import write_json_atomic
from .corruption import run_corruption_suite
from .previews import generate_previews
from .release import build_release_manifest, verify_release_manifest
from .verifier import verify_output_dataset

_EXCLUDED = {"CHECKSUMS.sha256", "RELEASE_MANIFEST.json", "LOCAL_READY.json", "REMOTE_READY.json"}


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(root: str | Path) -> list[dict[str, Any]]:
    root=Path(root)
    rows=[]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in _EXCLUDED:
            continue
        rows.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":_sha256(path)})
    (root/"CHECKSUMS.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),encoding="utf-8"
    )
    return rows


def finalize_local_release(
    output_root: str | Path,
    *,
    registry_path: str | Path,
    config: dict[str, Any],
    mini: bool,
) -> dict[str, Any]:
    root=Path(output_root)
    (root/"LOCAL_READY.json").unlink(missing_ok=True)
    (root/"REMOTE_READY.json").unlink(missing_ok=True)
    (root/"RELEASE_MANIFEST.json").unlink(missing_ok=True)
    qa=verify_output_dataset(root,registry_path=registry_path,config=config,mini=mini)
    previews=generate_previews(root)
    source_revisions = {
        str(item["repo_id"]): str(item["revision"])
        for item in config["sources"].values()
    }
    corruption = run_corruption_suite(root, source_revisions=source_revisions)
    write_json_atomic(root / "qa" / "CORRUPTION_REPORT.json", corruption)
    checksums=write_checksums(root)
    manifest=build_release_manifest(root)
    write_json_atomic(root/"RELEASE_MANIFEST.json",manifest)
    verify_release_manifest(root,manifest)
    manifest_sha=_sha256(root/"RELEASE_MANIFEST.json")
    ready={
        "status":"PASS","mode":"mini" if mini else "full",
        "build_fingerprint":(root/"BUILD_FINGERPRINT").read_text(encoding="ascii").strip(),
        "release_manifest_sha256":manifest_sha,
        "release_files":manifest["file_count"],"release_bytes":manifest["total_bytes"],
        "qa_report":"qa/QA_REPORT.json","corruption_report":"qa/CORRUPTION_REPORT.json",
        "corruption_tests":corruption["test_count"],
        "preview_inventory":"previews/PREVIEW_INVENTORY.json",
        "preview_sheets":len(previews["sheets"]),"checksummed_files":len(checksums),
        "total_rows":qa["all_rows"],"gold_rows":qa["gold_rows"],
        "extended_rows":qa.get("extended_rows",0),"quarantine_rows":qa.get("quarantine_rows",0),
        "integrity_errors":0,"leakage_errors":0,
    }
    write_json_atomic(root/"LOCAL_READY.json",ready)
    return ready
