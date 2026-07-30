from __future__ import annotations

import pytest

from heocr_unified.upload import UploadVerificationError, compare_remote_inventory


def test_remote_inventory_requires_every_file_and_exact_size() -> None:
    expected={"a":1,"b":2}
    compare_remote_inventory(expected,{"a":1,"b":2})
    with pytest.raises(UploadVerificationError,match="missing"):
        compare_remote_inventory(expected,{"a":1})
    with pytest.raises(UploadVerificationError,match="size"):
        compare_remote_inventory(expected,{"a":1,"b":3})
    with pytest.raises(UploadVerificationError,match="unexpected"):
        compare_remote_inventory(expected,{"a":1,"b":2,"c":3})


def test_result_commit_does_not_mutate_uploaded_remote_attestation() -> None:
    from heocr_unified.upload import _result_with_attestation_commit

    uploaded={"status":"PASS","commit_sha":"before"}
    result=_result_with_attestation_commit(uploaded,"after")

    assert uploaded == {"status":"PASS","commit_sha":"before"}
    assert result == {
        "status":"PASS",
        "commit_sha":"before",
        "commit_sha_with_attestation":"after",
    }



def test_private_write_probe_creates_uploads_verifies_and_deletes(monkeypatch, tmp_path: Path) -> None:
    import heocr_unified.upload as upload

    events: list[tuple] = []
    remote_payload = tmp_path / "remote-probe.json"

    class Info:
        private = True
        sha = "remote-commit"

    class FakeApi:
        def __init__(self, token: str):
            events.append(("init", token))

        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            events.append(("create", repo_id, repo_type, private, exist_ok))

        def dataset_info(self, repo_id):
            events.append(("info", repo_id))
            return Info()

        def upload_file(self, *, repo_id, repo_type, path_or_fileobj, path_in_repo, commit_message):
            payload = bytes(path_or_fileobj)
            remote_payload.write_bytes(payload)
            events.append(("upload", repo_id, repo_type, path_in_repo, commit_message, payload))

        def delete_repo(self, *, repo_id, repo_type):
            events.append(("delete", repo_id, repo_type))

    monkeypatch.setattr(upload, "HfApi", FakeApi)
    monkeypatch.setattr(upload, "hf_hub_download", lambda **kwargs: str(remote_payload))
    monkeypatch.setattr(upload.uuid, "uuid4", lambda: type("U", (), {"hex": "0123456789abcdef"})())

    result = upload.probe_private_write_access(
        output_repo="ssdataanalysis/hebrew-ocr-unified-sota-v1",
        token="write-token",
    )

    assert result == {
        "status": "PASS",
        "namespace": "ssdataanalysis",
        "probe_repo_id": "ssdataanalysis/heocr-write-probe-0123456789ab",
        "private": True,
        "uploaded_bytes": len(events[3][-1]),
        "download_verified": True,
        "deleted": True,
    }
    assert events[0] == ("init", "write-token")
    assert events[1] == (
        "create",
        "ssdataanalysis/heocr-write-probe-0123456789ab",
        "dataset",
        True,
        False,
    )
    assert events[-1] == (
        "delete",
        "ssdataanalysis/heocr-write-probe-0123456789ab",
        "dataset",
    )


def test_private_write_probe_fails_closed_when_cleanup_fails(monkeypatch, tmp_path: Path) -> None:
    import heocr_unified.upload as upload

    remote_payload = tmp_path / "remote-probe.json"

    class Info:
        private = True
        sha = "remote-commit"

    class FakeApi:
        def __init__(self, token: str):
            pass

        def create_repo(self, **kwargs):
            pass

        def dataset_info(self, repo_id):
            return Info()

        def upload_file(self, **kwargs):
            remote_payload.write_bytes(bytes(kwargs["path_or_fileobj"]))

        def delete_repo(self, **kwargs):
            raise RuntimeError("cannot delete")

    monkeypatch.setattr(upload, "HfApi", FakeApi)
    monkeypatch.setattr(upload, "hf_hub_download", lambda **kwargs: str(remote_payload))

    with pytest.raises(UploadVerificationError, match="cleanup"):
        upload.probe_private_write_access(
            output_repo="ssdataanalysis/hebrew-ocr-unified-sota-v1",
            token="write-token",
        )
