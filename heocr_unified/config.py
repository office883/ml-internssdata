from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from . import __version__

DEFAULT_CONFIG: dict[str, Any] = {
    "builder_version": __version__,
    "output_repo": "ssdataanalysis/hebrew-ocr-unified-sota-v1",
    "work_dir": str(Path.home() / "hebrew-ocr-unified-work-v15"),
    "upload": True,
    "private": True,
    "deep_remote_verify": True,
    "rows_per_shard": 1500,
    "page_rows_per_shard": 100,
    "architecture_chunk_size": 5000,
    "structured_chunk_size": 2500,
    "page_chunk_size": 100,
    "page_pool_limit": 200000,
    "minimum_free_gib": 200,
    "architecture_extra_variant_rate": 0.22,
    "architecture_structured_lines": 300000,
    "architecture_pages": 6000,
    "architecture_max_graphemes": 112,
    "pointed_manifest_path": "manifests/strict_all.jsonl.gz",
    "pointed_variants_per_text": 2,
    "pointed_chunk_size": 2000,
    "pointed_mini_per_split": 12,
    "pointed_max_graphemes": 160,
    "font_repo": {
        "url": "https://github.com/google/fonts.git",
        "revision": "7ff85c87f93ea6cca5f41c69f2e4edcb90240f26",
        "paths": [
            "ofl/alef",
            "ofl/assistant",
            "ofl/heebo",
            "ofl/rubik",
            "ofl/davidlibre",
            "ofl/frankruhllibre",
            "ofl/notosanshebrew",
            "ofl/notoserifhebrew",
            "ofl/notorashihebrew",
            "ofl/miriamlibre",
            "ofl/varelaround",
            "ofl/secularone",
            "ofl/suezone",
            "ofl/bellefair",
            "ofl/amaticsc",
        ],
    },
    "sources": {
        "foundation": {
            "repo_id": "ssdataanalysis/hebrew-ocr-foundation-v1",
            "revision": "1e277f98b17ad2efb9e6b13abbb7a06afe569a03",
        },
        "htr": {
            "repo_id": "ssdataanalysis/hebrew-htr-curated-v1",
            "revision": "ec4c7074ce2b3edc79889b00319e200d129eecf7",
        },
        "ocr": {
            "repo_id": "ssdataanalysis/hebrew-ocr-corpus",
            "revision": "ce4d1c347bd4e8b98a23f23256b0ecf01fa663c5",
        },
        "architecture": {
            "repo_id": "ssdataanalysis/hebrew-architecture-corpus",
            "revision": "58e7dd53a6caa42191252601f97b1dee96c3d765",
        },
    },
    "text_variant_caps": {
        "human": 32,
        "real": 20,
        "diffusion": 4,
        "synthetic": 8,
        "architecture": 4,
    },
    "acceptance": {
        "minimum_total_rows": 2200000,
        "minimum_train_rows": 2000000,
        "minimum_recognition_lines": 2000000,
        "minimum_unique_texts": 1200000,
        "minimum_human_train": 5000,
        "minimum_human_validation": 500,
        "minimum_human_test": 900,
        "minimum_architecture_natural_lines": 950000,
        "minimum_architecture_extra_variants": 150000,
        "minimum_architecture_structured_lines": 250000,
        "minimum_pages": 5000,
        "minimum_mixed_bidi": 100000,
        "minimum_with_digits": 250000,
        "minimum_with_combining_marks": 100000,
        "minimum_verified_pointed_rerender": 100000,
        "minimum_pointed_canonical_texts": 50000,
    },
}

_OPERATIONAL_KEYS = {"work_dir", "upload", "output_repo", "private", "minimum_free_gib", "deep_remote_verify"}


def builder_code_fingerprint(root: str | Path | None = None) -> str:
    """Return a deterministic digest of the builder code and pinned runtime inputs.

    The digest intentionally excludes user configuration and operational paths; those
    are represented separately by :func:`build_fingerprint`.  It covers every Python
    module that can affect dataset bytes plus the pinned dependency/project metadata.
    """

    project_root = (Path(root) if root is not None else Path(__file__).resolve().parent.parent).resolve()
    package_root = project_root / "heocr_unified"
    if not package_root.is_dir():
        raise ValueError(f"builder package directory is missing: {package_root}")

    paths = [path for path in package_root.rglob("*.py") if "__pycache__" not in path.parts]
    for name in ("requirements-lock.txt", "pyproject.toml"):
        path = project_root / name
        if path.is_file():
            paths.append(path)
    if not paths:
        raise ValueError("builder code fingerprint has no source files")

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(project_root).as_posix()):
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: str | Path | None, *, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        user = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(user, dict):
            raise ValueError("config root must be an object")
        _deep_merge(config, user)
    if overrides:
        _deep_merge(config, overrides)
    config["work_dir"] = os.path.abspath(os.path.expanduser(str(config["work_dir"])))
    # Never trust a user-supplied digest: bind the run to the exact local code bytes.
    config["builder_code_sha256"] = builder_code_fingerprint()
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("builder_version") != __version__:
        raise ValueError("builder_version must match package version")

    output_repo = str(config.get("output_repo") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", output_repo):
        raise ValueError("output_repo must be in owner/name form")
    if config.get("private") is not True:
        raise ValueError("private must be true; this builder refuses public OCR releases")
    for key in ("upload", "deep_remote_verify"):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"{key} must be boolean")

    sources = config.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"foundation", "htr", "ocr", "architecture"}:
        raise ValueError("sources must define exactly foundation, htr, ocr, and architecture")
    for name, source in sources.items():
        if not isinstance(source, Mapping):
            raise ValueError(f"source {name} must be an object")
        repo_id = str(source.get("repo_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id):
            raise ValueError(f"source {name} repo_id must be in owner/name form")
        revision = str(source.get("revision", ""))
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError(f"source {name} is not pinned to a 40-character commit")

    font_repo = config.get("font_repo")
    if not isinstance(font_repo, Mapping):
        raise ValueError("font_repo must be an object")
    font_revision = str(font_repo.get("revision") or "")
    if len(font_revision) != 40 or any(ch not in "0123456789abcdef" for ch in font_revision):
        raise ValueError("font repo must be pinned to a lowercase 40-character commit")
    font_url = str(font_repo.get("url") or "")
    if not font_url.startswith("https://"):
        raise ValueError("font_repo.url must use https")
    font_paths = font_repo.get("paths")
    if not isinstance(font_paths, list) or not font_paths:
        raise ValueError("font_repo.paths must be a non-empty list")
    normalized_paths: list[str] = []
    for value in font_paths:
        path = str(value or "")
        parts = path.split("/")
        if not path or path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("font_repo.paths must contain safe relative repository paths")
        normalized_paths.append(path)
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("font_repo.paths must be unique")

    positive_integer_keys = (
        "rows_per_shard", "page_rows_per_shard", "architecture_chunk_size",
        "structured_chunk_size", "page_chunk_size", "page_pool_limit",
        "architecture_structured_lines", "architecture_pages",
        "architecture_max_graphemes", "pointed_variants_per_text",
        "pointed_chunk_size", "pointed_mini_per_split", "pointed_max_graphemes",
    )
    for key in positive_integer_keys:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    free_gib = config.get("minimum_free_gib")
    if isinstance(free_gib, bool):
        raise ValueError("minimum_free_gib must be a finite non-negative number")
    try:
        free_gib_value = float(free_gib)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_free_gib must be a finite non-negative number") from exc
    if not math.isfinite(free_gib_value) or free_gib_value < 0:
        raise ValueError("minimum_free_gib must be a finite non-negative number")

    try:
        extra_variant_rate = float(config["architecture_extra_variant_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("architecture_extra_variant_rate must be a probability") from exc
    if not math.isfinite(extra_variant_rate) or not 0.0 <= extra_variant_rate <= 1.0:
        raise ValueError("architecture_extra_variant_rate must be between 0 and 1")

    acceptance = config.get("acceptance")
    if not isinstance(acceptance, Mapping) or not acceptance:
        raise ValueError("acceptance must be a non-empty object")
    for key, value in acceptance.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"acceptance.{key} must be a positive integer")
    caps = config.get("text_variant_caps")
    if not isinstance(caps, Mapping) or not caps:
        raise ValueError("text_variant_caps must be a non-empty object")
    for key, value in caps.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"text_variant_caps.{key} must be a positive integer")

    code_digest = str(config.get("builder_code_sha256", ""))
    if len(code_digest) != 64 or any(ch not in "0123456789abcdef" for ch in code_digest):
        raise ValueError("builder_code_sha256 must be a lowercase SHA-256 digest")
    if code_digest != builder_code_fingerprint():
        raise ValueError("builder_code_sha256 does not match the installed builder code")


def content_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in config.items() if key not in _OPERATIONAL_KEYS}


def build_fingerprint(config: Mapping[str, Any]) -> str:
    bound = copy.deepcopy(dict(config))
    bound.setdefault("builder_code_sha256", builder_code_fingerprint())
    payload = json.dumps(content_config(bound), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
