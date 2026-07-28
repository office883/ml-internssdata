from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Iterator

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"}


class WebDatasetError(RuntimeError):
    pass


def _key_ext(name: str) -> tuple[str, str]:
    base = Path(name).name
    if "." not in base:
        return base, ""
    key, ext = base.rsplit(".", 1)
    return key, ext.lower()


def iter_webdataset_strict(path: str | Path) -> Iterator[dict]:
    pending: dict[str, dict[str, bytes]] = {}
    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                key, ext = _key_ext(member.name)
                if not ext:
                    continue
                if ext not in _IMAGE_EXTENSIONS and ext != "json":
                    continue
                bucket = pending.setdefault(key, {})
                if ext in bucket:
                    raise WebDatasetError(f"duplicate member for key={key} ext={ext}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise WebDatasetError(f"cannot extract member {member.name}")
                bucket[ext] = handle.read()
                images = [candidate for candidate in _IMAGE_EXTENSIONS if candidate in bucket]
                if "json" in bucket and images:
                    if len(images) != 1:
                        raise WebDatasetError(f"multiple image members for key={key}")
                    try:
                        metadata = json.loads(bucket["json"].decode("utf-8", errors="strict"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise WebDatasetError(f"invalid JSON for key={key}") from exc
                    if not isinstance(metadata, dict):
                        raise WebDatasetError(f"JSON is not an object for key={key}")
                    image_ext = images[0]
                    yield {
                        "key": key,
                        "image_extension": image_ext,
                        "image_bytes": bucket[image_ext],
                        "metadata": metadata,
                    }
                    pending.pop(key, None)
    except WebDatasetError:
        raise
    except (tarfile.TarError, OSError) as exc:
        raise WebDatasetError(f"invalid or truncated TAR: {path}") from exc
    if pending:
        example = next(iter(pending))
        raise WebDatasetError(f"unpaired WebDataset members remain, e.g. key={example}")
