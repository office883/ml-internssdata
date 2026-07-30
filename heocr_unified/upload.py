from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

from .metadata import write_json_atomic
from .release import verify_release_manifest


class UploadVerificationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def compare_remote_inventory(expected: Mapping[str,int], actual: Mapping[str,int]) -> None:
    missing=sorted(set(expected)-set(actual))
    if missing: raise UploadVerificationError(f"remote inventory missing files: {missing[:10]}")
    unexpected=sorted(set(actual)-set(expected))
    if unexpected: raise UploadVerificationError(f"remote inventory has unexpected files: {unexpected[:10]}")
    wrong=[path for path in expected if int(expected[path])!=int(actual[path])]
    if wrong: raise UploadVerificationError(f"remote inventory size mismatch: {wrong[:10]}")


def _result_with_attestation_commit(attestation: Mapping[str, Any], commit_sha: str) -> dict[str, Any]:
    """Return API result metadata without changing the already-uploaded attestation file."""
    return {**dict(attestation), "commit_sha_with_attestation": commit_sha}


def _local_inventory(root: Path, *, exclude_remote_ready: bool=True) -> dict[str,int]:
    return {
        path.relative_to(root).as_posix():path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file() and not (exclude_remote_ready and path.name=="REMOTE_READY.json")
    }


def _remote_inventory(api: HfApi, repo_id: str, revision: str | None=None) -> dict[str,int]:
    result={}
    for item in api.list_repo_tree(repo_id=repo_id,repo_type="dataset",revision=revision,recursive=True,expand=True):
        if item.__class__.__name__=="RepoFile": result[str(item.path)]=int(getattr(item,"size",0) or 0)
    return result


def _ensure_remote_safe(api: HfApi, repo_id: str, root: Path, token: str) -> None:
    try:
        info=api.dataset_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id,repo_type="dataset",private=True,exist_ok=False)
        return
    if not info.private:
        raise UploadVerificationError("refusing to upload dataset into a public repository")
    files=_remote_inventory(api,repo_id)
    if not files: return
    if "BUILD_FINGERPRINT" not in files:
        raise UploadVerificationError("existing remote repository has no build fingerprint")
    remote=Path(hf_hub_download(repo_id=repo_id,repo_type="dataset",filename="BUILD_FINGERPRINT",token=token,force_download=True))
    local=(root/"BUILD_FINGERPRINT").read_text(encoding="ascii").strip()
    if remote.read_text(encoding="ascii").strip()!=local:
        raise UploadVerificationError("existing remote repository belongs to another build fingerprint")



def probe_private_write_access(
    *,
    output_repo: str,
    token: str,
) -> dict[str, Any]:
    """Fail-fast check that the active token can create, write, read and delete a private dataset repo."""
    namespace = str(output_repo).split("/", 1)[0].strip()
    if not namespace or "/" not in str(output_repo):
        raise UploadVerificationError("output_repo must be in namespace/name form")
    probe_repo_id = f"{namespace}/heocr-write-probe-{uuid.uuid4().hex[:12]}"
    payload = json.dumps(
        {
            "purpose": "hebrew-ocr-unified-builder-write-probe",
            "output_repo": str(output_repo),
            "nonce": uuid.uuid4().hex,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    api = HfApi(token=token)
    created = False
    primary_error: Exception | None = None
    cleanup_error: Exception | None = None
    result: dict[str, Any] | None = None
    try:
        api.create_repo(
            repo_id=probe_repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=False,
        )
        created = True
        info = api.dataset_info(probe_repo_id)
        if not bool(info.private):
            raise UploadVerificationError("write probe repository was not private")
        api.upload_file(
            repo_id=probe_repo_id,
            repo_type="dataset",
            path_or_fileobj=payload,
            path_in_repo="WRITE_PROBE.json",
            commit_message="Verify private dataset write access",
        )
        committed = api.dataset_info(probe_repo_id)
        if not bool(committed.private):
            raise UploadVerificationError("write probe repository became public")
        with tempfile.TemporaryDirectory(prefix="heocr-write-probe-") as cache_dir:
            downloaded = Path(
                hf_hub_download(
                    repo_id=probe_repo_id,
                    repo_type="dataset",
                    revision=committed.sha,
                    filename="WRITE_PROBE.json",
                    token=token,
                    cache_dir=cache_dir,
                    force_download=True,
                )
            )
            if downloaded.read_bytes() != payload:
                raise UploadVerificationError("write probe download did not match uploaded bytes")
        result = {
            "status": "PASS",
            "namespace": namespace,
            "probe_repo_id": probe_repo_id,
            "private": True,
            "uploaded_bytes": len(payload),
            "download_verified": True,
            "deleted": True,
        }
    except Exception as exc:  # deliberately normalize all network/auth failures
        primary_error = exc
    finally:
        if created:
            try:
                api.delete_repo(repo_id=probe_repo_id, repo_type="dataset")
            except Exception as exc:
                cleanup_error = exc
    if primary_error is not None:
        message = f"Hugging Face private write probe failed: {primary_error}"
        if cleanup_error is not None:
            message += f"; cleanup also failed: {cleanup_error}"
        if isinstance(primary_error, UploadVerificationError):
            raise UploadVerificationError(message) from primary_error
        raise UploadVerificationError(message) from primary_error
    if cleanup_error is not None:
        raise UploadVerificationError(
            f"Hugging Face private write probe cleanup failed: {cleanup_error}"
        ) from cleanup_error
    assert result is not None
    return result


def upload_private_release(
    output_root: str|Path,
    *,
    repo_id: str,
    token: str,
    deep_verify: bool=True,
) -> dict[str,Any]:
    root=Path(output_root).resolve()
    ready=json.loads((root/"LOCAL_READY.json").read_text(encoding="utf-8"))
    if ready.get("status")!="PASS" or ready.get("mode")!="full":
        raise UploadVerificationError("only a fully verified full build may be uploaded")
    manifest=json.loads((root/"RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    verify_release_manifest(root,manifest)
    (root/"REMOTE_READY.json").unlink(missing_ok=True)
    api=HfApi(token=token)
    _ensure_remote_safe(api,repo_id,root,token)
    hf=shutil.which("hf")
    if not hf: raise UploadVerificationError("Hugging Face CLI `hf` is not installed")
    command=[hf,"upload",repo_id,str(root),".","--repo-type","dataset","--private"]
    subprocess.run(command,check=True,env={**os.environ,"HF_TOKEN":token})
    info=api.dataset_info(repo_id)
    if not info.private: raise UploadVerificationError("remote repository became public")
    expected=_local_inventory(root)
    actual=_remote_inventory(api,repo_id,revision=info.sha)
    compare_remote_inventory(expected,actual)

    verification_cache=root.parent/"remote-verification-cache"
    if verification_cache.exists(): shutil.rmtree(verification_cache)
    verification_cache.mkdir(parents=True)
    manifest_hashes={row["path"]:row["sha256"] for row in manifest["files"]}
    critical={"RELEASE_MANIFEST.json","BUILD_FINGERPRINT","LOCAL_READY.json","qa/QA_REPORT.json","previews/PREVIEW_INVENTORY.json"}
    paths=sorted(expected) if deep_verify else sorted(critical & set(expected))
    checked=0
    for rel in paths:
        remote_path=Path(hf_hub_download(
            repo_id=repo_id,repo_type="dataset",revision=info.sha,filename=rel,token=token,
            cache_dir=verification_cache,force_download=True,
        ))
        local_hash=_sha256(root/rel)
        if _sha256(remote_path)!=local_hash:
            raise UploadVerificationError(f"remote hash mismatch: {rel}")
        if rel in manifest_hashes and manifest_hashes[rel]!=local_hash:
            raise UploadVerificationError(f"local release-manifest hash mismatch: {rel}")
        checked+=1
    remote_ready={
        "status":"PASS","repo_id":repo_id,"commit_sha":info.sha,"private":True,
        "build_fingerprint":ready["build_fingerprint"],"remote_files":len(actual),
        "remote_bytes":sum(actual.values()),"download_verified_files":checked,
        "deep_verify":bool(deep_verify),"release_manifest_sha256":_sha256(root/"RELEASE_MANIFEST.json"),
    }
    write_json_atomic(root/"REMOTE_READY.json",remote_ready)
    api.upload_file(
        repo_id=repo_id,repo_type="dataset",path_or_fileobj=str(root/"REMOTE_READY.json"),
        path_in_repo="REMOTE_READY.json",commit_message="Add verified remote-ready attestation",
    )
    final=api.dataset_info(repo_id)
    downloaded=Path(hf_hub_download(
        repo_id=repo_id,repo_type="dataset",revision=final.sha,filename="REMOTE_READY.json",
        token=token,cache_dir=verification_cache,force_download=True,
    ))
    if _sha256(downloaded)!=_sha256(root/"REMOTE_READY.json"):
        raise UploadVerificationError("REMOTE_READY attestation verification failed")
    return _result_with_attestation_commit(remote_ready, final.sha)
