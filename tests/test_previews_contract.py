from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from heocr_unified.previews import PreviewCandidate, write_contact_sheet


def _image(value: int) -> bytes:
    image=Image.new("RGB",(80,30),(255,255,255))
    image.putpixel((10+value,15),(0,0,0))
    buf=io.BytesIO(); image.save(buf,"PNG"); return buf.getvalue()


def test_contact_sheet_and_sidecar_are_deterministic(tmp_path: Path) -> None:
    rows=[PreviewCandidate(category=f"c{i}",sample_id=f"s{i}",text=f"שלום {i}",image_bytes=_image(i),metadata={"i":i}) for i in range(4)]
    first=write_contact_sheet(tmp_path/"a",title="test",candidates=rows)
    second=write_contact_sheet(tmp_path/"b",title="test",candidates=rows)
    assert first["image_sha256"] == second["image_sha256"]
    assert first["sidecar_sha256"] == second["sidecar_sha256"]
    assert (tmp_path/"a.png").is_file()
    assert (tmp_path/"a.json").is_file()
