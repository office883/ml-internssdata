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
