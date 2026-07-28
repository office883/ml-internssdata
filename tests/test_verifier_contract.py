from __future__ import annotations

import io
import json
from PIL import Image
import pytest

from heocr_unified.ingest import make_sample_row
from heocr_unified.verifier import (
    VerificationError, validate_row, verify_evaluation_reservations, verify_pointed_audit,
)


def _row() -> dict:
    image=Image.new("L",(60,20),255); image.putpixel((20,10),0)
    buf=io.BytesIO(); image.save(buf,"PNG")
    return make_sample_row(
        image_bytes=buf.getvalue(),image_path="a.png",text="שלום A-17",sample_id="x",
        split="train",task="line_recognition",granularity="line",modality="print",
        data_tier="gold",is_synthetic=True,sample_origin="synthetic",label_source="synthetic_ground_truth",label_trust="gold",quality_tier="A",source_repo="ssdataanalysis/hebrew-ocr-foundation-v1",
        source_revision="1"*40,source_path="shards/a.tar",source_split="train",source_id="a",
    )


def test_validate_row_recomputes_all_identities() -> None:
    row=_row()
    result=validate_row(row,source_revisions={row["source_repo"]:row["source_revision"]})
    assert result.text == "שלום A-17"
    assert result.mixed_bidi
    assert result.digits == 2


def test_validate_row_rejects_image_and_text_tampering() -> None:
    row=_row(); row["text_sha256"]="0"*64
    with pytest.raises(VerificationError,match="text SHA"):
        validate_row(row,source_revisions={row["source_repo"]:row["source_revision"]})
    row=_row(); row["image_sha256"]="0"*64
    with pytest.raises(VerificationError,match="image SHA"):
        validate_row(row,source_revisions={row["source_repo"]:row["source_revision"]})


def test_validate_page_rejects_out_of_bounds_or_wrong_reading_order() -> None:
    row=_row(); row["task"]="page_transcription"; row["granularity"]="page"
    row["annotations_json"]=json.dumps([
        {"text":"שלום A-17","reading_order":1,"bbox":[0,0,999,5],"polygon":[[0,0],[999,0],[999,5],[0,5]],"baseline":[[0,4],[999,4]]}
    ])
    with pytest.raises(VerificationError):
        validate_row(row,source_revisions={row["source_repo"]:row["source_revision"]})


def test_validate_row_accepts_explicit_quarantine_but_not_bad_provenance_contract() -> None:
    row = _row()
    row.update({
        "data_tier": "quarantine",
        "label_trust": "quarantine",
        "sample_origin": "unknown",
        "recommended_sampling_weight": 0.0,
    })
    validate_row(row, source_revisions={row["source_repo"]: row["source_revision"]})
    row["label_trust"] = "gold"
    with pytest.raises(VerificationError, match="label trust"):
        validate_row(row, source_revisions={row["source_repo"]: row["source_revision"]})


def test_verify_pointed_audit_binds_manifest_inventory_and_fingerprint(tmp_path: Path) -> None:
    audit = {
        "status": "PASS",
        "manifest_rows": 10,
        "matching_rows": 8,
        "eligible_occurrences": 6,
        "canonical_texts": 5,
        "duplicate_occurrences": 1,
        "by_split": {"train": 3, "validation_synthetic": 1, "test_synthetic": 1},
        "source_revision": "c" * 40,
        "manifest_sha256": "d" * 64,
        "policy": "test_synthetic>validation_synthetic>train",
        "trusted_datasets": ["samaritan-ai/hebrew_synth_lines"],
        "trusted_licenses": ["mit"],
        "max_graphemes": 160,
        "entries_fingerprint": "f" * 64,
    }
    payload = json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    audit["fingerprint"] = __import__("hashlib").sha256(payload.encode()).hexdigest()
    (tmp_path / "VERIFIED_POINTED_AUDIT.json").write_text(json.dumps(audit), encoding="utf-8")
    (tmp_path / "SOURCE_INVENTORY.json").write_text(json.dumps({
        "verified_pointed": {
            "repo_id": "ssdataanalysis/hebrew-ocr-corpus",
            "revision": "c" * 40,
            "path": "manifests/strict_all.jsonl.gz",
            "sha256": "d" * 64,
        }
    }), encoding="utf-8")
    config = {
        "pointed_manifest_path": "manifests/strict_all.jsonl.gz",
        "pointed_max_graphemes": 160,
        "sources": {"ocr": {"repo_id": "ssdataanalysis/hebrew-ocr-corpus", "revision": "c" * 40}},
        "acceptance": {"minimum_pointed_canonical_texts": 5},
    }
    checked = verify_pointed_audit(tmp_path, config=config)
    assert checked["canonical_texts"] == 5

    audit["manifest_sha256"] = "e" * 64
    (tmp_path / "VERIFIED_POINTED_AUDIT.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(VerificationError, match="manifest SHA"):
        verify_pointed_audit(tmp_path, config=config)


def test_verify_evaluation_reservations_requires_complete_accounting(tmp_path: Path) -> None:
    report = {
        "status": "PASS", "candidates": 7, "reserved": 5, "rejected": 2,
        "visual_candidates": 4, "generated_candidates": 3,
        "rejects": {"evaluation_text_owned_by_test": 2},
        "fingerprint": "a" * 64,
    }
    (tmp_path / "EVALUATION_RESERVATIONS.json").write_text(json.dumps(report), encoding="utf-8")
    assert verify_evaluation_reservations(tmp_path)["reserved"] == 5
    report["candidates"] = 8
    (tmp_path / "EVALUATION_RESERVATIONS.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(VerificationError, match="accounting"):
        verify_evaluation_reservations(tmp_path)


def test_validate_row_rejects_samaritan_script_inside_gold() -> None:
    row = _row()
    row.update({
        "modality": "historical_samaritan_handwritten_line",
        "sample_origin": "real",
        "is_synthetic": False,
        "label_source": "source_transcription",
        "provenance_reason": "known_real_source",
        "source_repo": "ssdataanalysis/hebrew-ocr-corpus",
        "source_revision": "c" * 40,
        "source_path": "webdataset/historical_handwriting_lines/train/a.tar",
        "provenance_json": json.dumps({
            "source_metadata": {"source_dataset": "johnlockejrr/samaritan_v1"}
        }),
    })
    with pytest.raises(VerificationError, match="Samaritan script"):
        validate_row(row, source_revisions={row["source_repo"]: row["source_revision"]})


def test_validate_row_allows_samaritan_script_in_extended_opt_in() -> None:
    row = _row()
    row.update({
        "modality": "historical_samaritan_lightonocr_line",
        "data_tier": "extended",
        "label_trust": "extended",
        "sample_origin": "real",
        "is_synthetic": False,
        "label_source": "source_transcription",
        "provenance_reason": "samaritan_script_opt_in_only",
        "recommended_sampling_weight": 0.35,
        "source_repo": "ssdataanalysis/hebrew-ocr-corpus",
        "source_revision": "c" * 40,
        "source_path": "webdataset/historical_handwriting_lines/train/a.tar",
        "provenance_json": json.dumps({
            "source_metadata": {"source_dataset": "samaritan-ai/samaritan_hebrew_LightOnOcr"}
        }),
    })
    validate_row(row, source_revisions={row["source_repo"]: row["source_revision"]})
