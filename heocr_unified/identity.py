from __future__ import annotations

import hashlib
import io
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def byte_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_visual_sha256(data: bytes) -> str:
    """Hash decoded raster content, robust to metadata and lossless encoding changes."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        gray = ImageOps.autocontrast(image.convert("L"))
        array = np.asarray(gray, dtype=np.uint8)
    # Quantization makes minor codec noise non-authoritative while retaining glyph shape.
    quantized = ((array.astype(np.uint16) + 8) // 16).astype(np.uint8)
    header = f"{quantized.shape[1]}x{quantized.shape[0]}|L4|".encode("ascii")
    return hashlib.sha256(header + quantized.tobytes(order="C")).hexdigest()


def stable_token(*parts: Any, length: int = 24) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
