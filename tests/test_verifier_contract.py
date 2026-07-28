from __future__ import annotations

import io
import json
from PIL import Image
import pytest

from heocr_unified.ingest import make_sample_row
from heocr_unified.verifier import VerificationError, validate_row


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
