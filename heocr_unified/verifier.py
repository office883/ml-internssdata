from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .identity import byte_sha256, canonical_visual_sha256
from .unicode_utils import NormalizedLabel, normalize_label_strict


class VerificationError(RuntimeError):
    pass



_SAMARITAN_DATASETS = {
    "johnlockejrr/samaritan_v1",
    "samaritan-ai/samaritan_hebrew_LightOnOcr",
}
_SAMARITAN_MODALITIES = {
    "historical_samaritan_handwritten_line",
    "historical_samaritan_lightonocr_line",
}

_REQUIRED = {
    "sample_id", "image", "text", "split", "task", "granularity", "modality",
    "data_tier", "is_synthetic", "sample_origin", "label_source", "label_trust",
    "provenance_reason", "quality_tier", "source_repo", "source_revision",
    "source_path", "source_split", "source_id", "source_document", "source_page",
    "writer_id", "image_sha256", "visual_sha256", "text_sha256", "width",
    "height", "image_format", "font_family", "font_style", "font_sha256",
    "augmentation_json", "annotations_json", "provenance_json",
    "recommended_sampling_weight",
}


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
        raise VerificationError("missing embedded image bytes")
    return bytes(value)


def _json_object(value: Any, *, field: str, expected: type) -> Any:
    if not isinstance(value, str):
        raise VerificationError(f"{field} is not a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid {field}") from exc
    if not isinstance(parsed, expected):
        raise VerificationError(f"{field} has wrong JSON type")
    return parsed


def _point_in_bounds(point: Any, width: int, height: int) -> bool:
    return (
        isinstance(point, (list, tuple)) and len(point) == 2
        and 0 <= float(point[0]) <= width and 0 <= float(point[1]) <= height
    )


def base_config_name(config_name: str) -> str:
    value = str(config_name)
    for suffix in ("_quarantine", "_extended"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def required_config_families_present(required: set[str], present: set[str]) -> bool:
    families = {base_config_name(name) for name in present}
    return set(required).issubset(families)


def _read_json_file(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 2:
        raise VerificationError(f"missing {description}: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid {description}: {path.name}") from exc
    if not isinstance(payload, dict):
        raise VerificationError(f"{description} must be a JSON object")
    return payload


def verify_evaluation_reservations(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    report = _read_json_file(
        root / "EVALUATION_RESERVATIONS.json", description="evaluation reservation report"
    )
    if str(report.get("status") or "") != "PASS":
        raise VerificationError("evaluation reservation report is not PASS")
    try:
        candidates = int(report["candidates"])
        reserved = int(report["reserved"])
        rejected = int(report["rejected"])
        visual = int(report["visual_candidates"])
        generated = int(report["generated_candidates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError("evaluation reservation counts are invalid") from exc
    if candidates < 1 or reserved < 1:
        raise VerificationError("evaluation reservation report is empty")
    if candidates != reserved + rejected or candidates != visual + generated:
        raise VerificationError("evaluation reservation accounting mismatch")
    rejects = report.get("rejects")
    if not isinstance(rejects, dict) or sum(int(value) for value in rejects.values()) != rejected:
        raise VerificationError("evaluation reservation reject accounting mismatch")
    fingerprint = str(report.get("fingerprint") or "")
    if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
        raise VerificationError("evaluation reservation fingerprint is invalid")
    return report


def verify_pointed_audit(output_root: str | Path, *, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(output_root)
    audit = _read_json_file(root / "VERIFIED_POINTED_AUDIT.json", description="pointed audit")
    inventory = _read_json_file(root / "SOURCE_INVENTORY.json", description="source inventory")
    source = dict(config["sources"]["ocr"])
    item = inventory.get("verified_pointed")
    if not isinstance(item, dict):
        raise VerificationError("source inventory lacks verified_pointed")

    expected_path = str(config.get("pointed_manifest_path") or "manifests/strict_all.jsonl.gz")
    if str(item.get("repo_id") or "") != str(source["repo_id"]):
        raise VerificationError("pointed inventory repo mismatch")
    if str(item.get("revision") or "") != str(source["revision"]):
        raise VerificationError("pointed inventory revision mismatch")
    if str(item.get("path") or "") != expected_path:
        raise VerificationError("pointed inventory path mismatch")
    inventory_sha = str(item.get("sha256") or "")
    if len(inventory_sha) != 64:
        raise VerificationError("pointed inventory lacks manifest SHA-256")
    if str(audit.get("manifest_sha256") or "") != inventory_sha:
        raise VerificationError("pointed audit manifest SHA mismatch")
    if str(audit.get("source_revision") or "") != str(source["revision"]):
        raise VerificationError("pointed audit source revision mismatch")
    if str(audit.get("status") or "") != "PASS":
        raise VerificationError("pointed audit is not PASS")
    if str(audit.get("policy") or "") != "test_synthetic>validation_synthetic>train":
        raise VerificationError("pointed audit policy mismatch")
    if int(audit.get("max_graphemes", -1)) != int(config["pointed_max_graphemes"]):
        raise VerificationError("pointed audit grapheme limit mismatch")

    by_split_raw = audit.get("by_split")
    if not isinstance(by_split_raw, dict):
        raise VerificationError("pointed audit lacks split counts")
    required_splits = ("train", "validation_synthetic", "test_synthetic")
    try:
        by_split = {split: int(by_split_raw[split]) for split in required_splits}
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError("pointed audit split counts are invalid") from exc
    if any(value < 1 for value in by_split.values()):
        raise VerificationError("pointed audit has an empty canonical split")
    canonical = int(audit.get("canonical_texts", -1))
    eligible = int(audit.get("eligible_occurrences", -1))
    duplicates = int(audit.get("duplicate_occurrences", -1))
    if len(str(audit.get("entries_fingerprint") or "")) != 64:
        raise VerificationError("pointed audit lacks entries fingerprint")
    if canonical != sum(by_split.values()):
        raise VerificationError("pointed audit canonical count mismatch")
    if eligible != canonical + duplicates:
        raise VerificationError("pointed audit duplicate accounting mismatch")
    if not (int(audit.get("manifest_rows", -1)) >= int(audit.get("matching_rows", -1)) >= eligible):
        raise VerificationError("pointed audit source count ordering is invalid")
    minimum = int(config["acceptance"].get("minimum_pointed_canonical_texts", 1))
    if canonical < minimum:
        raise VerificationError(f"pointed audit canonical_texts={canonical} < {minimum}")

    fingerprint = str(audit.get("fingerprint") or "")
    payload = {key: value for key, value in audit.items() if key != "fingerprint"}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != fingerprint:
        raise VerificationError("pointed audit fingerprint mismatch")
    return audit


def validate_verified_pointed_row(
    row: Mapping[str, Any], *, config_name: str, audit: Mapping[str, Any], label: NormalizedLabel
) -> None:
    if str(row.get("label_source") or "") != "verified_text_rerender":
        return
    if base_config_name(config_name) != "verified_pointed_rerender":
        raise VerificationError("verified pointed row is in the wrong config")
    if str(row.get("data_tier") or "") != "gold" or str(row.get("label_trust") or "") != "gold":
        raise VerificationError("verified pointed row is not gold")
    if not bool(row.get("is_synthetic")) or str(row.get("sample_origin") or "") != "synthetic":
        raise VerificationError("verified pointed row has the wrong origin")
    if str(row.get("modality") or "") != "pointed_print" or label.combining_marks < 1:
        raise VerificationError("verified pointed row is not actually pointed print")
    if str(row.get("source_repo") or "") != "ssdataanalysis/hebrew-ocr-corpus":
        raise VerificationError("verified pointed row source repo mismatch")
    if str(row.get("source_path") or "") != "manifests/strict_all.jsonl.gz#biblical_pointed_lines":
        raise VerificationError("verified pointed row source path mismatch")
    provenance = _json_object(row.get("provenance_json"), field="provenance_json", expected=dict)
    if str(provenance.get("generator") or "") != "verified-pointed-v12":
        raise VerificationError("verified pointed generator mismatch")
    if str(provenance.get("manifest_sha256") or "") != str(audit.get("manifest_sha256") or ""):
        raise VerificationError("verified pointed row manifest SHA mismatch")
    if str(provenance.get("source_dataset") or "") != "samaritan-ai/hebrew_synth_lines":
        raise VerificationError("verified pointed source dataset mismatch")
    if str(provenance.get("source_license") or "").lower() != "mit":
        raise VerificationError("verified pointed source license mismatch")


def _expected_tier_for_config(config_name: str) -> str:
    if config_name.endswith("_quarantine"):
        return "quarantine"
    if config_name.endswith("_extended"):
        return "extended"
    return "gold"


def enforce_config_tier(config_name: str, row: Mapping[str, Any]) -> None:
    """Prove that trust tiers are physically isolated by config path."""
    expected = _expected_tier_for_config(str(config_name))
    tier = str(row.get("data_tier") or "")
    if tier != expected:
        raise VerificationError(
            f"config/tier mismatch: config={config_name} expects {expected}, row={tier}"
        )
    if str(row.get("label_trust") or "") != tier:
        raise VerificationError("label trust does not match data tier")
    try:
        weight = float(row.get("recommended_sampling_weight"))
    except (TypeError, ValueError) as exc:
        raise VerificationError("invalid sampling weight") from exc
    if weight != weight or weight in {float("inf"), float("-inf")} or weight < 0:
        raise VerificationError("invalid sampling weight")
    if tier == "quarantine":
        if weight != 0.0:
            raise VerificationError("quarantine sampling weight must be exactly zero")
    elif weight <= 0.0:
        raise VerificationError("gold/extended rows require a positive sampling weight")


def validate_row(row: Mapping[str, Any], *, source_revisions: Mapping[str, str]) -> NormalizedLabel:
    missing = _REQUIRED - set(row)
    if missing:
        raise VerificationError(f"missing required fields: {sorted(missing)}")
    label = normalize_label_strict(str(row["text"]))
    if label.text != row["text"]:
        raise VerificationError("text is not canonical NFC/logical form")
    if label.text_sha256 != row["text_sha256"]:
        raise VerificationError("text SHA mismatch")

    repo = str(row["source_repo"])
    if repo in source_revisions and str(row["source_revision"]) != str(source_revisions[repo]):
        raise VerificationError("source revision mismatch")
    revision = str(row["source_revision"])
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise VerificationError("source revision is not a pinned commit")

    data = _image_bytes(row["image"])
    if byte_sha256(data) != row["image_sha256"]:
        raise VerificationError("image SHA mismatch")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            fmt = str(image.format or "unknown").lower()
    except Exception as exc:
        raise VerificationError("image cannot be decoded") from exc
    if width != int(row["width"]) or height != int(row["height"]):
        raise VerificationError("image dimensions mismatch")
    if fmt != str(row["image_format"]).lower():
        raise VerificationError("image format mismatch")
    if canonical_visual_sha256(data) != row["visual_sha256"]:
        raise VerificationError("visual SHA mismatch")

    _json_object(row["augmentation_json"], field="augmentation_json", expected=dict)
    annotations = _json_object(row["annotations_json"], field="annotations_json", expected=list)
    provenance = _json_object(row["provenance_json"], field="provenance_json", expected=dict)

    split = str(row["split"])
    if split not in {"train", "validation", "test", "validation_synthetic", "test_synthetic"}:
        raise VerificationError(f"unknown split: {split}")
    tier = str(row["data_tier"])
    if tier not in {"gold", "extended", "quarantine"}:
        raise VerificationError("unknown data tier")
    if str(row["label_trust"]) != tier:
        raise VerificationError("label trust does not match data tier")
    origin = str(row["sample_origin"])
    if origin not in {"human", "real", "synthetic", "diffusion", "unknown"}:
        raise VerificationError("unknown sample origin")
    if origin in {"synthetic", "diffusion"} and not bool(row["is_synthetic"]):
        raise VerificationError("synthetic origin is marked real")
    if origin in {"human", "real"} and bool(row["is_synthetic"]):
        raise VerificationError("real origin is marked synthetic")
    if tier != "quarantine" and origin == "unknown":
        raise VerificationError("unknown provenance outside quarantine")
    if tier != "quarantine" and str(row["label_source"]) == "unknown":
        raise VerificationError("unknown label source outside quarantine")

    source_metadata = provenance.get("source_metadata")
    source_dataset = (
        str(source_metadata.get("source_dataset") or "")
        if isinstance(source_metadata, dict) else ""
    )
    modality = str(row.get("modality") or "")
    if tier == "gold" and (
        source_dataset in _SAMARITAN_DATASETS or modality in _SAMARITAN_MODALITIES
    ):
        raise VerificationError("Samaritan script is opt-in and cannot appear in gold")
    weight = float(row["recommended_sampling_weight"])
    if tier == "quarantine" and weight != 0.0:
        raise VerificationError("quarantine sampling weight must be exactly zero")
    if tier != "quarantine" and weight <= 0.0:
        raise VerificationError("gold/extended rows require a positive sampling weight")

    if row["task"] == "page_transcription":
        if not annotations:
            raise VerificationError("page has no annotations")
        ordered = sorted(annotations, key=lambda item: int(item.get("reading_order", -1)))
        if [int(item.get("reading_order", -1)) for item in ordered] != list(range(len(ordered))):
            raise VerificationError("page reading order is not contiguous from zero")
        if label.text.splitlines() != [str(item.get("text", "")) for item in ordered]:
            raise VerificationError("page transcription differs from annotation text")
        for item in ordered:
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise VerificationError("page annotation has invalid bbox")
            x0, y0, x1, y1 = map(float, bbox)
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise VerificationError("page annotation bbox is out of bounds")
            for field in ("polygon", "baseline"):
                points = item.get(field)
                if not isinstance(points, list) or not points or not all(
                    _point_in_bounds(point, width, height) for point in points
                ):
                    raise VerificationError(f"page annotation {field} is out of bounds")
    elif annotations:
        # Non-page samples may later gain glyph boxes, but only when explicitly typed.
        if not all(isinstance(item, dict) for item in annotations):
            raise VerificationError("non-page annotations are malformed")

    return label


def enforce_acceptance(summary: Mapping[str, Any], *, config: Mapping[str, Any], mini: bool) -> None:
    if int(summary.get("leakage_errors", 0)) != 0:
        raise VerificationError("leakage_errors must be zero")
    if int(summary.get("integrity_errors", 0)) != 0:
        raise VerificationError("integrity_errors must be zero")
    if int(summary.get("architecture_gold_accounted", -1)) != int(summary.get("architecture_gold_total", -2)):
        raise VerificationError("architecture gold coverage is incomplete")

    actual_keys = {
        "gold_rows": "minimum_total_rows",
        "gold_train_rows": "minimum_train_rows",
        "gold_recognition_lines": "minimum_recognition_lines",
        "gold_unique_texts": "minimum_unique_texts",
        "human_train": "minimum_human_train",
        "human_validation": "minimum_human_validation",
        "human_test": "minimum_human_test",
        "architecture_primary_lines": "minimum_architecture_natural_lines",
        "architecture_extra_variants": "minimum_architecture_extra_variants",
        "architecture_structured_lines": "minimum_architecture_structured_lines",
        "pages": "minimum_pages",
        "mixed_bidi": "minimum_mixed_bidi",
        "with_digits": "minimum_with_digits",
        "with_combining_marks": "minimum_with_combining_marks",
        "verified_pointed_rerender": "minimum_verified_pointed_rerender",
    }
    if mini:
        for key in actual_keys:
            if int(summary.get(key, 0)) < 1:
                raise VerificationError(f"mini acceptance failed: {key}")
        if not summary.get("required_configs_present"):
            raise VerificationError("mini acceptance failed: required_configs_present")
        if not summary.get("required_source_families_present"):
            raise VerificationError("mini acceptance failed: required_source_families_present")
        return

    acceptance = config["acceptance"]
    for actual_key, minimum_key in actual_keys.items():
        actual = int(summary.get(actual_key, 0))
        minimum = int(acceptance[minimum_key])
        if actual < minimum:
            raise VerificationError(f"acceptance failed: {actual_key}={actual} < {minimum}")


def _split_group(split: str) -> str:
    if split == "train":
        return "train"
    if split.startswith("validation"):
        return "validation"
    if split.startswith("test"):
        return "test"
    raise VerificationError(f"unknown split group: {split}")


def _init_verification_db(path: Path):
    import sqlite3
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(
        """
        CREATE TABLE sample_ids(value TEXT PRIMARY KEY);
        CREATE TABLE visuals(
          tier TEXT NOT NULL, value TEXT NOT NULL, text_sha TEXT NOT NULL,
          split_group TEXT NOT NULL, sample_id TEXT NOT NULL,
          PRIMARY KEY(tier,value)
        );
        CREATE INDEX visuals_value_idx ON visuals(value);
        CREATE TABLE text_groups(
          tier TEXT NOT NULL, task TEXT NOT NULL, text_sha TEXT NOT NULL, split_group TEXT NOT NULL,
          PRIMARY KEY(tier,task,text_sha,split_group)
        );
        CREATE INDEX text_groups_lookup_idx ON text_groups(task,text_sha,tier,split_group);
        CREATE TABLE entities(
          tier TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL, split_group TEXT NOT NULL,
          PRIMARY KEY(tier,kind,value,split_group)
        );
        CREATE INDEX entities_lookup_idx ON entities(kind,value,tier,split_group);
        CREATE TABLE unique_texts(value TEXT PRIMARY KEY);
        CREATE TABLE gold_unique_texts(value TEXT PRIMARY KEY);
        """
    )
    return db


def verify_output_dataset(
    output_root: str | Path,
    *,
    registry_path: str | Path,
    config: Mapping[str, Any],
    mini: bool,
    progress_every: int = 10000,
) -> dict[str, Any]:
    """Exhaustively verify every Parquet row and every embedded image."""
    import collections
    import sqlite3
    import time
    import pyarrow.parquet as pq

    from .registry import DedupRegistry, owner_scopes, train_tiers_blocked_by_evaluation
    from .writer import verify_parquet_artifact
    from .metadata import write_json_atomic

    root = Path(output_root)
    qa_dir = root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = (root / "BUILD_FINGERPRINT").read_text(encoding="ascii").strip()
    registry = DedupRegistry(registry_path, build_fingerprint=fingerprint)
    source_revisions = {
        str(item["repo_id"]): str(item["revision"])
        for item in config["sources"].values()
    }
    reservation_report = verify_evaluation_reservations(root)
    pointed_audit = verify_pointed_audit(root, config=config)
    verifier_db = _init_verification_db(qa_dir / "verification.sqlite")
    counts = collections.Counter()
    split_counts = collections.Counter()
    config_counts = collections.Counter()
    source_families: set[str] = set()
    gold_source_families: set[str] = set()
    configs_present: set[str] = set()
    started = time.time()
    leakage_errors = 0
    integrity_errors = 0

    try:
        registered = {
            row["path"]: row for source in [r[0] for r in registry.db.execute("SELECT source_key FROM source_tasks")]
            for row in registry.artifacts_for_source(source)
        }
        actual_parquets = {
            path.relative_to(root).as_posix(): path for path in sorted((root / "data").rglob("*.parquet"))
        }
        if set(registered) != set(actual_parquets):
            missing = sorted(set(registered) - set(actual_parquets))[:10]
            extra = sorted(set(actual_parquets) - set(registered))[:10]
            raise VerificationError(f"artifact inventory mismatch missing={missing} extra={extra}")

        for rel, artifact in sorted(registered.items()):
            path = actual_parquets[rel]
            verify_parquet_artifact(
                path, expected_rows=int(artifact["rows"]), expected_sha256=str(artifact["sha256"])
            )
            parts = Path(rel).parts
            if len(parts) < 4 or parts[0] != "data":
                raise VerificationError(f"invalid data path: {rel}")
            config_name, path_split = parts[1], parts[2]
            configs_present.add(config_name)
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    counts["all_rows"] += 1
                    try:
                        label = validate_row(row, source_revisions=source_revisions)
                        enforce_config_tier(config_name, row)
                        validate_verified_pointed_row(
                            row, config_name=config_name, audit=pointed_audit, label=label
                        )
                    except Exception:
                        integrity_errors += 1
                        raise
                    if row["split"] != path_split:
                        raise VerificationError(f"row/path split mismatch: {rel}")
                    sample_id = str(row["sample_id"])
                    try:
                        verifier_db.execute("INSERT INTO sample_ids VALUES(?)", (sample_id,))
                    except sqlite3.IntegrityError as exc:
                        leakage_errors += 1
                        raise VerificationError(f"duplicate sample id: {sample_id}") from exc
                    tier = str(row["data_tier"])
                    group = _split_group(str(row["split"]))
                    visual_sha = str(row["visual_sha256"])
                    text_sha = str(row["text_sha256"])
                    existing_same_tier = verifier_db.execute(
                        "SELECT text_sha,split_group FROM visuals WHERE tier=? AND value=?",
                        (tier, visual_sha),
                    ).fetchone()
                    if existing_same_tier:
                        leakage_errors += 1
                        raise VerificationError("duplicate visual inside one trust tier")
                    for existing_tier, existing_text, existing_group in verifier_db.execute(
                        "SELECT tier,text_sha,split_group FROM visuals WHERE value=?", (visual_sha,)
                    ):
                        existing_tier = str(existing_tier)
                        existing_group = str(existing_group)
                        if str(existing_text) != text_sha and "quarantine" not in {tier, existing_tier}:
                            integrity_errors += 1
                            raise VerificationError("cross-tier visual-label conflict")
                        if tier != "quarantine" and existing_tier != "quarantine":
                            if group == "train" and existing_group != "train" and existing_tier in owner_scopes(tier):
                                leakage_errors += 1
                                raise VerificationError("evaluation visual leaked into train")
                            if group != "train" and existing_group == "train" and existing_tier in train_tiers_blocked_by_evaluation(tier):
                                leakage_errors += 1
                                raise VerificationError("train visual leaked into evaluation")
                    verifier_db.execute(
                        "INSERT INTO visuals VALUES(?,?,?,?,?)",
                        (tier, visual_sha, text_sha, group, sample_id),
                    )

                    # Quarantine is preserved for audit, but it is not part of any
                    # train/evaluation universe and therefore cannot contaminate
                    # leakage or acceptance statistics.
                    if tier != "quarantine":
                        scopes = owner_scopes(tier)
                        blocked_train_tiers = train_tiers_blocked_by_evaluation(tier)
                        if row["task"] in {"line_recognition", "page_transcription"}:
                            same_tier_groups = {
                                str(item[0]) for item in verifier_db.execute(
                                    "SELECT split_group FROM text_groups WHERE tier=? AND task=? AND text_sha=?",
                                    (tier, row["task"], text_sha),
                                )
                            }
                            if same_tier_groups and group not in same_tier_groups:
                                leakage_errors += 1
                                raise VerificationError("exact text leakage across split groups")
                            if group == "train" and scopes:
                                placeholders = ",".join("?" for _ in scopes)
                                prior_eval = verifier_db.execute(
                                    f"SELECT 1 FROM text_groups WHERE task=? AND text_sha=? "
                                    f"AND tier IN ({placeholders}) AND split_group!='train' LIMIT 1",
                                    (row["task"], text_sha, *scopes),
                                ).fetchone()
                                if prior_eval:
                                    leakage_errors += 1
                                    raise VerificationError("evaluation text leaked into train")
                            elif group != "train" and blocked_train_tiers:
                                placeholders = ",".join("?" for _ in blocked_train_tiers)
                                prior_train = verifier_db.execute(
                                    f"SELECT 1 FROM text_groups WHERE task=? AND text_sha=? "
                                    f"AND tier IN ({placeholders}) AND split_group='train' LIMIT 1",
                                    (row["task"], text_sha, *blocked_train_tiers),
                                ).fetchone()
                                if prior_train:
                                    leakage_errors += 1
                                    raise VerificationError("train text leaked into evaluation")
                            verifier_db.execute(
                                "INSERT OR IGNORE INTO text_groups VALUES(?,?,?,?)",
                                (tier, row["task"], text_sha, group),
                            )
                        for kind, key in (("writer", row.get("writer_id")), ("document", row.get("source_document"))):
                            if not key:
                                continue
                            same_tier_groups = {
                                str(item[0]) for item in verifier_db.execute(
                                    "SELECT split_group FROM entities WHERE tier=? AND kind=? AND value=?",
                                    (tier, kind, key),
                                )
                            }
                            if same_tier_groups and group not in same_tier_groups:
                                leakage_errors += 1
                                raise VerificationError(f"{kind} leakage across split groups")
                            if group == "train" and scopes:
                                placeholders = ",".join("?" for _ in scopes)
                                prior_eval = verifier_db.execute(
                                    f"SELECT 1 FROM entities WHERE kind=? AND value=? "
                                    f"AND tier IN ({placeholders}) AND split_group!='train' LIMIT 1",
                                    (kind, key, *scopes),
                                ).fetchone()
                                if prior_eval:
                                    leakage_errors += 1
                                    raise VerificationError(f"evaluation {kind} leaked into train")
                            elif group != "train" and blocked_train_tiers:
                                placeholders = ",".join("?" for _ in blocked_train_tiers)
                                prior_train = verifier_db.execute(
                                    f"SELECT 1 FROM entities WHERE kind=? AND value=? "
                                    f"AND tier IN ({placeholders}) AND split_group='train' LIMIT 1",
                                    (kind, key, *blocked_train_tiers),
                                ).fetchone()
                                if prior_train:
                                    leakage_errors += 1
                                    raise VerificationError(f"train {kind} leaked into evaluation")
                            verifier_db.execute(
                                "INSERT OR IGNORE INTO entities VALUES(?,?,?,?)", (tier, kind, key, group)
                            )
                    verifier_db.execute("INSERT OR IGNORE INTO unique_texts VALUES(?)", (row["text_sha256"],))
                    if tier == "gold":
                        verifier_db.execute(
                            "INSERT OR IGNORE INTO gold_unique_texts VALUES(?)", (row["text_sha256"],)
                        )

                    split_counts[str(row["split"])] += 1
                    config_counts[config_name] += 1
                    counts[f"{tier}_rows"] += 1
                    if tier == "gold":
                        if row["split"] == "train": counts["gold_train_rows"] += 1
                        if row["task"] == "line_recognition": counts["gold_recognition_lines"] += 1
                        if label.mixed_bidi: counts["mixed_bidi"] += 1
                        if label.digits: counts["with_digits"] += 1
                        if label.combining_marks: counts["with_combining_marks"] += 1
                        if str(row.get("label_source")) == "verified_text_rerender":
                            counts["verified_pointed_rerender"] += 1
                    source_repo = str(row["source_repo"])
                    source_family = ""
                    if source_repo.endswith("hebrew-htr-curated-v1"): source_family = "htr"
                    elif source_repo.endswith("hebrew-ocr-foundation-v1"): source_family = "foundation"
                    elif source_repo.endswith("hebrew-ocr-corpus"): source_family = "ocr"
                    elif source_repo.endswith("hebrew-architecture-corpus"): source_family = "architecture"
                    if source_family:
                        source_families.add(source_family)
                        if tier == "gold":
                            gold_source_families.add(source_family)
                    if tier == "gold" and source_repo.endswith("hebrew-htr-curated-v1") and str(row["source_path"]).startswith("stage3_human_finetune/"):
                        if row["split"] == "train": counts["human_train"] += 1
                        elif row["split"] == "validation": counts["human_validation"] += 1
                        elif row["split"] == "test": counts["human_test"] += 1
                    if tier == "gold" and source_repo.endswith("hebrew-architecture-corpus") and str(row["source_path"]).startswith("txt/"):
                        counts["architecture_natural_lines"] += 1
                        provenance = json.loads(str(row["provenance_json"]))
                        if provenance.get("variant_role") == "extra_train":
                            counts["architecture_extra_variants"] += 1
                        elif provenance.get("variant_role") == "primary":
                            counts["architecture_primary_lines"] += 1
                    if tier == "gold" and str(row["source_path"]) == "generated/structured-lines":
                        counts["architecture_structured_lines"] += 1
                    if tier == "gold" and row["task"] == "page_transcription": counts["pages"] += 1
                    if progress_every and counts["all_rows"] % progress_every == 0:
                        verifier_db.commit()
                        print(f"VERIFY rows={counts['all_rows']:,} elapsed={time.time()-started:.1f}s", flush=True)
        verifier_db.commit()
        counts["unique_texts"] = int(verifier_db.execute("SELECT COUNT(*) FROM unique_texts").fetchone()[0])
        counts["gold_unique_texts"] = int(
            verifier_db.execute("SELECT COUNT(*) FROM gold_unique_texts").fetchone()[0]
        )
        ledger = registry.architecture_ledger_summary()
        counts["architecture_gold_total"] = int(ledger["gold_total"])
        counts["architecture_gold_accounted"] = int(ledger["gold_accounted"])
        counts["leakage_errors"] = leakage_errors
        counts["integrity_errors"] = integrity_errors
        for key in ("gold_rows", "extended_rows", "quarantine_rows", "gold_train_rows", "gold_recognition_lines"):
            counts.setdefault(key, 0)
        required_configs = {
            "modern_print_lines", "modern_print_words", "structured_bidi_lines",
            "historical_print_lines", "lexicographic_print_lines", "biblical_pointed_lines",
            "rabbinic_rashi_lines", "handwriting_real_lines", "handwriting_historical_lines",
            "handwriting_synthetic_lines", "handwriting_real_characters",
            "architecture_synthetic_lines", "architecture_structured_lines",
            "architecture_synthetic_pages", "verified_pointed_rerender",
        }
        counts["required_configs_present"] = required_config_families_present(required_configs, configs_present)
        counts["required_source_families_present"] = {"htr","foundation","ocr","architecture"}.issubset(gold_source_families)
        summary = {
            **dict(counts),
            "splits": dict(sorted(split_counts.items())),
            "configs": dict(sorted(config_counts.items())),
            "source_families": sorted(source_families),
            "gold_source_families": sorted(gold_source_families),
            "architecture_ledger": ledger,
            "verified_pointed_audit": pointed_audit,
            "evaluation_reservations": reservation_report,
            "registry_samples": registry.sample_count(),
            "registered_artifacts": len(registered),
            "elapsed_seconds": round(time.time() - started, 3),
            "mode": "mini" if mini else "full",
        }
        if summary["registry_samples"] != summary["all_rows"]:
            raise VerificationError("registry/output row count mismatch")
        enforce_acceptance(summary, config=config, mini=mini)
        write_json_atomic(qa_dir / "QA_REPORT.json", summary)
        return summary
    finally:
        verifier_db.close()
        registry.close()
