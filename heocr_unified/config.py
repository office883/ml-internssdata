from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import __version__

DEFAULT_CONFIG: dict[str, Any] = {
    "builder_version": __version__,
    "output_repo": "ssdataanalysis/hebrew-ocr-unified-sota-v1",
    "work_dir": str(Path.home() / "hebrew-ocr-unified-work-v11"),
    "upload": True,
    "private": True,
    "deep_remote_verify": True,
    "rows_per_shard": 1500,
    "page_rows_per_shard": 100,
    "architecture_chunk_size": 5000,
    "structured_chunk_size": 2500,
    "page_chunk_size": 100,
    "page_pool_limit": 200000,
    "minimum_free_gib": 120,
    "architecture_extra_variant_rate": 0.22,
    "architecture_structured_lines": 120000,
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
        "minimum_architecture_structured_lines": 100000,
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
    for name, source in config["sources"].items():
        revision = str(source.get("revision", ""))
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError(f"source {name} is not pinned to a 40-character commit")
    font_revision = str(config["font_repo"]["revision"])
    if len(font_revision) != 40:
        raise ValueError("font repo must be pinned")
    if int(config["rows_per_shard"]) < 1:
        raise ValueError("rows_per_shard must be positive")
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
