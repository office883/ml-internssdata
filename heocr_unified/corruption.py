from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from .release import build_release_manifest, verify_release_manifest
from .verifier import VerificationError, enforce_config_tier, validate_row
from .writer import verify_parquet_artifact


class CorruptionSuiteError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expect_failure(name: str, function: Callable[[], Any], expected: tuple[type[BaseException], ...]) -> dict[str, str]:
    try:
        function()
    except expected as exc:
        return {"name": name, "status": "PASS", "caught": type(exc).__name__, "message": str(exc)}
    except Exception as exc:  # pragma: no cover - indicates the guard failed in an unexpected way
        raise CorruptionSuiteError(f"{name} raised an unexpected exception: {exc!r}") from exc
    raise CorruptionSuiteError(f"{name} was not detected")


def run_corruption_suite(
    output_root: str | Path,
    *,
    source_revisions: Mapping[str, str],
) -> dict[str, Any]:
    """Prove that representative corruption cannot pass the release gates.

    The original output is never modified. Parquet corruption is performed on a
    private copy; row corruptions are in memory; manifest corruptions use copied
    dictionaries or isolated temporary fixtures.
    """
    import pyarrow.parquet as pq

    root = Path(output_root)
    parquets = sorted((root / "data").rglob("*.parquet"))
    if not parquets:
        raise CorruptionSuiteError("no Parquet file exists for corruption testing")
    parquet_path = parquets[0]
    parts = parquet_path.relative_to(root).parts
    if len(parts) < 4:
        raise CorruptionSuiteError("invalid Parquet path")
    config_name = parts[1]
    parquet = pq.ParquetFile(parquet_path)
    if parquet.metadata.num_rows < 1:
        raise CorruptionSuiteError("first Parquet is empty")
    row = next(iter(parquet.iter_batches(batch_size=1))).to_pylist()[0]
    validate_row(row, source_revisions=source_revisions)
    enforce_config_tier(config_name, row)

    tests: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="heocr-corrupt-") as temp_dir:
        temp = Path(temp_dir)
        corrupted_parquet = temp / parquet_path.name
        shutil.copy2(parquet_path, corrupted_parquet)
        payload = bytearray(corrupted_parquet.read_bytes())
        if not payload:
            raise CorruptionSuiteError("Parquet copy is empty")
        index = max(0, min(len(payload) - 1, len(payload) // 2))
        payload[index] ^= 0x01
        corrupted_parquet.write_bytes(payload)
        tests.append(_expect_failure(
            "parquet_byte_corruption",
            lambda: verify_parquet_artifact(
                corrupted_parquet,
                expected_rows=parquet.metadata.num_rows,
                expected_sha256=_sha256(parquet_path),
            ),
            (ValueError, OSError),
        ))

        fixture = temp / "manifest-fixture"
        fixture.mkdir()
        (fixture / "a.txt").write_text("a", encoding="utf-8")
        fixture_manifest = build_release_manifest(fixture)
        (fixture / "extra.txt").write_text("extra", encoding="utf-8")
        tests.append(_expect_failure(
            "unexpected_release_file",
            lambda: verify_release_manifest(fixture, fixture_manifest),
            (ValueError,),
        ))

    image_row = copy.deepcopy(row)
    image_value = dict(image_row["image"])
    image_bytes = bytearray(image_value["bytes"])
    image_bytes[min(len(image_bytes) - 1, max(0, len(image_bytes) // 2))] ^= 0x01
    image_value["bytes"] = bytes(image_bytes)
    image_row["image"] = image_value
    tests.append(_expect_failure(
        "embedded_image_corruption",
        lambda: validate_row(image_row, source_revisions=source_revisions),
        (VerificationError,),
    ))

    text_row = copy.deepcopy(row)
    text_row["text"] = str(text_row["text"]) + "א"
    tests.append(_expect_failure(
        "text_label_corruption",
        lambda: validate_row(text_row, source_revisions=source_revisions),
        (VerificationError,),
    ))

    revision_row = copy.deepcopy(row)
    revision_row["source_revision"] = "0" * 40
    tests.append(_expect_failure(
        "source_revision_corruption",
        lambda: validate_row(revision_row, source_revisions=source_revisions),
        (VerificationError,),
    ))

    tier_row = copy.deepcopy(row)
    tier_row["data_tier"] = "quarantine" if row["data_tier"] != "quarantine" else "gold"
    tier_row["label_trust"] = tier_row["data_tier"]
    tier_row["recommended_sampling_weight"] = 0.0 if tier_row["data_tier"] == "quarantine" else 1.0
    tests.append(_expect_failure(
        "config_tier_corruption",
        lambda: enforce_config_tier(config_name, tier_row),
        (VerificationError,),
    ))

    provisional = build_release_manifest(root)
    if not provisional["files"]:
        raise CorruptionSuiteError("release manifest is empty")
    hash_manifest = copy.deepcopy(provisional)
    target = next(
        (item for item in hash_manifest["files"] if item["path"] == "FONT_MANIFEST.json"),
        hash_manifest["files"][0],
    )
    target["sha256"] = "0" * 64
    tests.append(_expect_failure(
        "release_or_font_manifest_hash_corruption",
        lambda: verify_release_manifest(root, hash_manifest),
        (ValueError,),
    ))

    return {
        "status": "PASS",
        "parquet": parquet_path.relative_to(root).as_posix(),
        "sample_id": str(row["sample_id"]),
        "test_count": len(tests),
        "tests": tests,
    }
