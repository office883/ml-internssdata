from __future__ import annotations

import io
import json
from typing import Any, Iterator

from PIL import Image

from .identity import byte_sha256, canonical_visual_sha256, stable_token
from .sources import MappedSource, SourceTask, normal_output_split
from .provenance import classify_row_provenance
from .unicode_utils import namespace_key, normalize_label_strict
from .webdataset import iter_webdataset_strict


class ImageIntegrityError(RuntimeError):
    pass


def _image_info(data: bytes) -> tuple[int, int, str]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = int(image.width), int(image.height)
            fmt = str(image.format or "unknown").lower()
    except Exception as exc:
        raise ImageIntegrityError("image cannot be decoded") from exc
    if width < 1 or height < 1:
        raise ImageIntegrityError("image has invalid dimensions")
    return width, height, fmt


def make_sample_row(
    *,
    image_bytes: bytes,
    image_path: str,
    text: str,
    sample_id: str,
    split: str,
    task: str,
    granularity: str,
    modality: str,
    data_tier: str,
    is_synthetic: bool,
    sample_origin: str = "unknown",
    label_source: str = "unknown",
    label_trust: str = "",
    provenance_reason: str = "",
    quality_tier: str = "",
    source_repo: str,
    source_revision: str,
    source_path: str,
    source_split: str,
    source_id: str,
    source_document: str = "",
    source_page: str = "",
    writer_id: str = "",
    declared_image_sha256: str = "",
    declared_width: int | None = None,
    declared_height: int | None = None,
    font_family: str = "",
    font_style: str = "",
    font_sha256: str = "",
    augmentation: dict[str, Any] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    recommended_sampling_weight: float = 1.0,
) -> dict[str, Any]:
    data_tier = str(data_tier)
    sample_origin = str(sample_origin)
    label_source = str(label_source)
    label_trust = str(label_trust or data_tier)
    if data_tier not in {"gold", "extended", "quarantine"}:
        raise ValueError("invalid data tier")
    if label_trust != data_tier:
        raise ValueError("label trust must match data tier")
    if sample_origin not in {"human", "real", "synthetic", "diffusion", "unknown"}:
        raise ValueError("invalid provenance origin")
    if data_tier != "quarantine" and (sample_origin == "unknown" or label_source == "unknown"):
        raise ValueError("gold/extended rows require explicit provenance")
    if sample_origin in {"synthetic", "diffusion"} and not bool(is_synthetic):
        raise ValueError("synthetic provenance must set is_synthetic")
    if sample_origin in {"human", "real"} and bool(is_synthetic):
        raise ValueError("real provenance cannot set is_synthetic")
    weight = float(recommended_sampling_weight)
    if not (weight >= 0.0) or weight in {float("inf"), float("-inf")} or weight != weight:
        raise ValueError("invalid recommended sampling weight")
    if data_tier == "quarantine" and weight != 0.0:
        raise ValueError("quarantine rows require zero sampling weight")
    if data_tier != "quarantine" and weight <= 0.0:
        raise ValueError("gold/extended rows require positive sampling weight")

    label = normalize_label_strict(text)
    actual_sha = byte_sha256(image_bytes)
    if declared_image_sha256 and len(declared_image_sha256) == 64 and declared_image_sha256.lower() != actual_sha:
        raise ImageIntegrityError("declared image SHA-256 does not match actual bytes")
    width, height, fmt = _image_info(image_bytes)
    if declared_width not in (None, 0) and int(declared_width) != width:
        raise ImageIntegrityError("declared width does not match decoded image")
    if declared_height not in (None, 0) and int(declared_height) != height:
        raise ImageIntegrityError("declared height does not match decoded image")
    return {
        "sample_id": sample_id,
        "image": {"bytes": image_bytes, "path": image_path},
        "text": label.text,
        "split": split,
        "task": task,
        "granularity": granularity,
        "modality": modality,
        "data_tier": data_tier,
        "is_synthetic": bool(is_synthetic),
        "sample_origin": sample_origin,
        "label_source": label_source,
        "label_trust": label_trust,
        "provenance_reason": str(provenance_reason),
        "quality_tier": str(quality_tier),
        "source_repo": source_repo,
        "source_revision": source_revision,
        "source_path": source_path,
        "source_split": source_split,
        "source_id": source_id,
        "source_document": source_document,
        "source_page": source_page,
        "writer_id": writer_id,
        "image_sha256": actual_sha,
        "visual_sha256": canonical_visual_sha256(image_bytes),
        "text_sha256": label.text_sha256,
        "width": width,
        "height": height,
        "image_format": fmt,
        "font_family": font_family,
        "font_style": font_style,
        "font_sha256": font_sha256,
        "augmentation_json": json.dumps(augmentation or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "annotations_json": json.dumps(annotations or [], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "provenance_json": json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "recommended_sampling_weight": weight,
    }


def _image_value(value: Any) -> tuple[bytes, str]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value), "image"
    if isinstance(value, dict):
        data = value.get("bytes")
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data), str(value.get("path") or "image")
    raise ImageIntegrityError("source row has no embedded image bytes")


def _first_text(metadata: dict[str, Any]) -> str:
    for key in ("text", "text_logical", "transcription", "label", "ground_truth", "caption"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("source row has no text label")


def iter_htr_parquet(
    path: str,
    *,
    task: SourceTask,
    mapped: MappedSource,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    columns = [name for name in [
        "image", "text", "sample_id", "image_sha256", "width", "height", "source_repo",
        "source_revision", "source_split", "source_file", "source_row_index", "writer",
        "source_doc", "human_source", "recommended_sampling_weight", "text_group_id",
    ] if name in parquet.schema_arrow.names]
    produced = 0
    source_row = 0
    for batch in parquet.iter_batches(batch_size=512, columns=columns):
        for metadata in batch.to_pylist():
            if limit is not None and produced >= limit:
                return
            image_bytes, image_path = _image_value(metadata.get("image"))
            source_id = str(metadata.get("sample_id") or stable_token(task.source_key, source_row))
            writer = namespace_key(task.repo_id, "writer", metadata.get("writer"))
            document = namespace_key(task.repo_id, "document", metadata.get("source_doc"))
            decision = classify_row_provenance(task, mapped, metadata)
            yield make_sample_row(
                image_bytes=image_bytes,
                image_path=image_path,
                text=_first_text(metadata),
                sample_id=f"htr-{source_id}",
                split=mapped.output_split,
                task=mapped.task,
                granularity=mapped.granularity,
                modality=mapped.modality,
                data_tier=decision.data_tier,
                is_synthetic=decision.is_synthetic,
                sample_origin=decision.sample_origin,
                label_source=decision.label_source,
                label_trust=decision.label_trust,
                provenance_reason=decision.reason,
                quality_tier=str(metadata.get("quality_tier") or ""),
                source_repo=task.repo_id,
                source_revision=task.revision,
                source_path=task.path,
                source_split=task.split,
                source_id=source_id,
                source_document=document,
                writer_id=writer,
                declared_image_sha256=str(metadata.get("image_sha256") or ""),
                declared_width=metadata.get("width"),
                declared_height=metadata.get("height"),
                provenance={
                    "source_row_index": source_row,
                    "text_group_id": metadata.get("text_group_id"),
                    "human_source": metadata.get("human_source"),
                    "classification": decision.reason,
                },
                recommended_sampling_weight=decision.recommended_sampling_weight,
            )
            produced += 1
            source_row += 1


def iter_webdataset_source(
    path: str,
    *,
    task: SourceTask,
    mapped: MappedSource,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    produced = 0
    for source_row, item in enumerate(iter_webdataset_strict(path)):
        if limit is not None and produced >= limit:
            return
        metadata = item["metadata"]
        source_id = str(metadata.get("id") or metadata.get("sample_id") or item["key"])
        writer = namespace_key(task.repo_id, "writer", metadata.get("writer") or metadata.get("writer_id"))
        document_value = (
            metadata.get("source_doc")
            or metadata.get("source_document")
            or metadata.get("source_image")
            or metadata.get("source_page")
            or metadata.get("curation_group_id")
        )
        document = namespace_key(task.repo_id, "document", document_value)
        page = namespace_key(task.repo_id, "page", metadata.get("source_page") or metadata.get("source_image"))
        decision = classify_row_provenance(task, mapped, metadata)
        row_split = normal_output_split(
            task.split,
            synthetic=decision.is_synthetic,
            human=decision.sample_origin == "human",
        )
        yield make_sample_row(
            image_bytes=item["image_bytes"],
            image_path=f"{item['key']}.{item['image_extension']}",
            text=_first_text(metadata),
            sample_id=f"wds-{stable_token(task.repo_id, task.path, source_id)}",
            split=row_split,
            task=mapped.task,
            granularity=mapped.granularity,
            modality=str(metadata.get("modality") or mapped.modality),
            data_tier=decision.data_tier,
            is_synthetic=decision.is_synthetic,
            sample_origin=decision.sample_origin,
            label_source=decision.label_source,
            label_trust=decision.label_trust,
            provenance_reason=decision.reason,
            quality_tier=str(metadata.get("quality_tier") or ""),
            source_repo=task.repo_id,
            source_revision=task.revision,
            source_path=task.path,
            source_split=task.split,
            source_id=source_id,
            source_document=document,
            source_page=page,
            writer_id=writer,
            declared_image_sha256=str(metadata.get("image_sha256") or ""),
            declared_width=metadata.get("width") or metadata.get("image_width"),
            declared_height=metadata.get("height") or metadata.get("image_height"),
            font_family=str(metadata.get("font_family") or (metadata.get("font") or {}).get("family") or ""),
            font_style=str((metadata.get("font") or {}).get("style") or ""),
            font_sha256=str((metadata.get("font") or {}).get("sha256") or ""),
            augmentation=(metadata.get("render") or {}).get("augmentation") or {"noise_profile": metadata.get("noise_profile")},
            provenance={
                "source_row_index": source_row,
                "source_metadata": metadata,
                "classification": decision.reason,
            },
            recommended_sampling_weight=decision.recommended_sampling_weight,
        )
        produced += 1
