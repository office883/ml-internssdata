from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .augment import LINE_PROFILES
from .identity import stable_token
from .ingest import make_sample_row
from .unicode_utils import grapheme_clusters, normalize_label_strict

_OCR_REPO = "ssdataanalysis/hebrew-ocr-corpus"
_MANIFEST_PATH = "manifests/strict_all.jsonl.gz"
_CONFIG = "biblical_pointed_lines"
_TRUSTED_DATASETS = {"samaritan-ai/hebrew_synth_lines"}
_TRUSTED_LICENSES = {"mit"}
_SPLIT_MAP = {
    "train": "train",
    "validation": "validation_synthetic",
    "val": "validation_synthetic",
    "test": "test_synthetic",
}
_SPLIT_PRIORITY = {"test_synthetic": 0, "validation_synthetic": 1, "train": 2}

# One clean-ish view and one broader domain-randomized view per canonical label.
_CLEAN_PROFILES = ("clean_digital", "office_scan", "grayscale_scan", "yellowed_archive")
_HARD_PROFILES = tuple(
    name for name in LINE_PROFILES
    if name not in set(_CLEAN_PROFILES) | {"fax", "thermal_receipt", "dark_ui", "blueprint", "extreme"}
)


@dataclass(frozen=True)
class PointedTextEntry:
    text: str
    text_sha256: str
    split: str
    source_id: str
    source_line: int
    source_dataset: str
    source_license: str
    original_split: str = "train"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


class PointedTextResolver:
    """Build a global, deterministic owner index for verified pointed labels.

    The source images remain in their original trust tier. Only the normalized
    logical-order text is used to create new images with the pinned renderer.
    Exact text is owned by one split globally (test > validation > train).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_revision: str,
        manifest_sha256: str,
        max_graphemes: int = 160,
    ):
        if len(source_revision) != 40:
            raise ValueError("source_revision must be a pinned commit")
        if len(manifest_sha256) != 64:
            raise ValueError("manifest_sha256 must be a SHA-256")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source_revision = source_revision
        self.manifest_sha256 = manifest_sha256
        self.max_graphemes = int(max_graphemes)
        self.db = sqlite3.connect(self.path, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS entries(
              text_sha256 TEXT PRIMARY KEY,
              text TEXT NOT NULL,
              split TEXT NOT NULL,
              priority INTEGER NOT NULL,
              source_id TEXT NOT NULL,
              source_line INTEGER NOT NULL,
              source_dataset TEXT NOT NULL,
              source_license TEXT NOT NULL,
              original_split TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS pointed_split_idx ON entries(split,text_sha256);
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def _metadata(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _reset(self) -> None:
        self.db.execute("DELETE FROM entries")
        self.db.execute("DELETE FROM metadata")
        self.db.commit()

    def _entries_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for row in self.db.execute(
            "SELECT text_sha256,text,split,priority,source_id,source_line,source_dataset,"
            "source_license,original_split FROM entries ORDER BY text_sha256"
        ):
            digest.update("\x1f".join(map(str, row)).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _summary_fingerprint(summary: dict[str, Any]) -> str:
        payload = {key: value for key, value in summary.items() if key != "fingerprint"}
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _stored_summary(self) -> dict[str, Any] | None:
        if self._metadata("status") != "complete":
            return None
        if self._metadata("source_revision") != self.source_revision:
            return None
        if self._metadata("manifest_sha256") != self.manifest_sha256:
            return None
        raw = self._metadata("summary")
        if raw is None:
            return None
        summary = json.loads(raw)
        if not isinstance(summary, dict) or summary.get("status") != "PASS":
            return None
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("pointed resolver SQLite integrity check failed")
        actual = int(self.db.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
        by_split = {
            split: int(self.db.execute("SELECT COUNT(*) FROM entries WHERE split=?", (split,)).fetchone()[0])
            for split in ("train", "validation_synthetic", "test_synthetic")
        }
        if actual != int(summary.get("canonical_texts", -1)):
            return None
        if by_split != {key: int(value) for key, value in dict(summary.get("by_split") or {}).items()}:
            return None
        if self._entries_fingerprint() != str(summary.get("entries_fingerprint") or ""):
            return None
        if self._summary_fingerprint(summary) != str(summary.get("fingerprint") or ""):
            return None
        return summary

    def build(self, manifest_path: str | Path) -> dict[str, Any]:
        manifest = Path(manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        if _sha256_file(manifest) != self.manifest_sha256:
            raise ValueError("pointed manifest SHA-256 mismatch")
        cached = self._stored_summary()
        if cached is not None:
            return cached

        self._reset()
        counts: dict[str, int] = {
            "manifest_rows": 0,
            "matching_rows": 0,
            "eligible_occurrences": 0,
            "invalid_labels": 0,
            "without_combining_marks": 0,
            "too_long": 0,
            "not_recommended": 0,
            "wrong_text_order": 0,
            "untrusted_source": 0,
            "untrusted_license": 0,
        }

        with gzip.open(manifest, "rt", encoding="utf-8", errors="strict") as handle:
            for source_line, raw in enumerate(handle, start=1):
                counts["manifest_rows"] += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in pointed manifest at line {source_line}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row in pointed manifest at line {source_line}")
                if str(row.get("curated_config") or "") != _CONFIG:
                    continue
                counts["matching_rows"] += 1
                if row.get("recommended") is not True or row.get("standalone_selected") is not True:
                    counts["not_recommended"] += 1
                    continue
                order = str(row.get("rtl_text_order") or "")
                if order != "logical_unicode":
                    counts["wrong_text_order"] += 1
                    continue
                source_dataset = str(row.get("source_dataset") or "")
                if source_dataset not in _TRUSTED_DATASETS:
                    counts["untrusted_source"] += 1
                    continue
                source_license = str(row.get("source_license") or "").strip().lower()
                if source_license not in _TRUSTED_LICENSES:
                    counts["untrusted_license"] += 1
                    continue
                original_split = str(row.get("split") or row.get("original_split") or "").lower()
                split = _SPLIT_MAP.get(original_split)
                if split is None:
                    raise ValueError(f"unknown pointed split at line {source_line}: {original_split!r}")
                try:
                    label = normalize_label_strict(str(row.get("text") or ""))
                except Exception:
                    counts["invalid_labels"] += 1
                    continue
                if label.combining_marks < 1:
                    counts["without_combining_marks"] += 1
                    continue
                if len(grapheme_clusters(label.text)) > self.max_graphemes:
                    counts["too_long"] += 1
                    continue

                counts["eligible_occurrences"] += 1
                source_id = str(row.get("id") or row.get("source_id") or f"line-{source_line}")
                candidate = (
                    _SPLIT_PRIORITY[split],
                    source_id,
                    source_line,
                )
                existing = self.db.execute(
                    "SELECT priority,source_id,source_line FROM entries WHERE text_sha256=?",
                    (label.text_sha256,),
                ).fetchone()
                if existing is None or candidate < (int(existing[0]), str(existing[1]), int(existing[2])):
                    self.db.execute(
                        """
                        INSERT INTO entries(
                          text_sha256,text,split,priority,source_id,source_line,
                          source_dataset,source_license,original_split
                        ) VALUES(?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(text_sha256) DO UPDATE SET
                          text=excluded.text, split=excluded.split, priority=excluded.priority,
                          source_id=excluded.source_id, source_line=excluded.source_line,
                          source_dataset=excluded.source_dataset,
                          source_license=excluded.source_license,
                          original_split=excluded.original_split
                        """,
                        (
                            label.text_sha256,
                            label.text,
                            split,
                            _SPLIT_PRIORITY[split],
                            source_id,
                            source_line,
                            source_dataset,
                            source_license,
                            original_split,
                        ),
                    )
                if counts["manifest_rows"] % 10000 == 0:
                    self.db.commit()
        self.db.commit()

        canonical = int(self.db.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
        by_split = {
            split: int(self.db.execute("SELECT COUNT(*) FROM entries WHERE split=?", (split,)).fetchone()[0])
            for split in ("train", "validation_synthetic", "test_synthetic")
        }
        summary: dict[str, Any] = {
            "status": "PASS",
            **counts,
            "canonical_texts": canonical,
            "duplicate_occurrences": counts["eligible_occurrences"] - canonical,
            "by_split": by_split,
            "source_revision": self.source_revision,
            "manifest_sha256": self.manifest_sha256,
            "policy": "test_synthetic>validation_synthetic>train",
            "trusted_datasets": sorted(_TRUSTED_DATASETS),
            "trusted_licenses": sorted(_TRUSTED_LICENSES),
            "max_graphemes": self.max_graphemes,
            "entries_fingerprint": self._entries_fingerprint(),
        }
        summary["fingerprint"] = self._summary_fingerprint(summary)
        for key, value in (
            ("status", "complete"),
            ("source_revision", self.source_revision),
            ("manifest_sha256", self.manifest_sha256),
            ("summary", json.dumps(summary, ensure_ascii=False, sort_keys=True)),
        ):
            self.db.execute(
                "INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
        self.db.commit()
        return summary

    def iter_entries(self, split: str, *, limit: int | None = None) -> Iterator[PointedTextEntry]:
        if split not in _SPLIT_PRIORITY:
            raise ValueError(f"unknown pointed output split: {split}")
        query = (
            "SELECT text,text_sha256,split,source_id,source_line,source_dataset,source_license,original_split "
            "FROM entries WHERE split=? ORDER BY text_sha256,source_id,source_line"
        )
        params: tuple[Any, ...] = (split,)
        if limit is not None:
            query += " LIMIT ?"
            params = (split, int(limit))
        for row in self.db.execute(query, params):
            yield PointedTextEntry(
                text=str(row[0]),
                text_sha256=str(row[1]),
                split=str(row[2]),
                source_id=str(row[3]),
                source_line=int(row[4]),
                source_dataset=str(row[5]),
                source_license=str(row[6]),
                original_split=str(row[7]),
            )


def render_verified_pointed_row(
    entry: PointedTextEntry,
    *,
    variant: int,
    renderer,
    source_revision: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    if variant < 0:
        raise ValueError("variant must be non-negative")
    seed = _seed("verified-pointed-v11", entry.text_sha256, variant)
    profiles = _CLEAN_PROFILES if variant == 0 else _HARD_PROFILES
    profile = profiles[seed % len(profiles)]
    rendered = renderer.render_line(
        entry.text,
        profile=profile,
        seed=seed,
        split=entry.split,
        rashi=False,
    )
    if rendered.visibility_fraction < 0.42:
        rendered = renderer.render_line(
            entry.text,
            profile="clean_digital",
            seed=seed,
            split=entry.split,
            rashi=False,
        )
    if rendered.visibility_fraction < 0.42:
        raise RuntimeError(
            f"verified pointed label cannot be rendered visibly: {entry.text_sha256}"
        )
    sample_id = f"pointed-{stable_token(entry.text_sha256, entry.split, variant, length=28)}"
    return make_sample_row(
        image_bytes=rendered.to_bytes(),
        image_path=f"{sample_id}.webp",
        text=entry.text,
        sample_id=sample_id,
        split=entry.split,
        task="line_recognition",
        granularity="line",
        modality="pointed_print",
        data_tier="gold",
        is_synthetic=True,
        sample_origin="synthetic",
        label_source="verified_text_rerender",
        label_trust="gold",
        provenance_reason="verified_logical_pointed_text_rerendered_by_pinned_pipeline",
        quality_tier="A-rerendered",
        source_repo=_OCR_REPO,
        source_revision=source_revision,
        source_path=f"{_MANIFEST_PATH}#{_CONFIG}",
        source_split=entry.original_split,
        source_id=entry.source_id,
        font_family=rendered.font.family,
        font_style=rendered.font.style,
        font_sha256=rendered.font.sha256,
        augmentation=rendered.metadata,
        provenance={
            "generator": "verified-pointed-v11",
            "manifest_path": _MANIFEST_PATH,
            "manifest_sha256": manifest_sha256,
            "source_line": entry.source_line,
            "source_dataset": entry.source_dataset,
            "source_license": entry.source_license,
            "variant": int(variant),
            "text_sha256": entry.text_sha256,
        },
        recommended_sampling_weight=1.0,
    )


POINTED_MANIFEST_PATH = _MANIFEST_PATH
POINTED_OUTPUT_CONFIG = "verified_pointed_rerender"


def iter_verified_pointed_rows(
    entries: Iterator[PointedTextEntry] | list[PointedTextEntry],
    *,
    variants: int,
    renderer,
    source_revision: str,
    manifest_sha256: str,
) -> Iterator[dict[str, Any]]:
    if int(variants) < 1:
        raise ValueError("variants must be positive")
    for entry in entries:
        for variant in range(int(variants)):
            yield render_verified_pointed_row(
                entry,
                variant=variant,
                renderer=renderer,
                source_revision=source_revision,
                manifest_sha256=manifest_sha256,
            )
