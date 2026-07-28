from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterator

from .unicode_utils import LabelRejected, classify_usable_label, split_grapheme_safe


class ArchitectureState(StrEnum):
    GOLD = "gold"
    QUARANTINE = "quarantine"
    DUPLICATE = "duplicate"
    EMPTY = "empty"
    INVALID_UTF8 = "invalid_utf8"
    UNSAFE_UNICODE = "unsafe_unicode"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ArchitectureSegment:
    document_id: str
    source_line: int
    segment_index: int
    text: str
    text_sha256: str
    state: ArchitectureState
    reason: str
    split: str
    origin: str
    metadata: dict[str, str]

    @property
    def segment_key(self) -> str:
        return f"{self.document_id}:{self.source_line}:{self.segment_index}"


_SPLIT_PRIORITY = {
    "test_synthetic": 0,
    "validation_synthetic": 1,
    "train": 2,
}


class ArchitectureTextResolver:
    """Global exact-text owner resolver backed by SQLite.

    Architecture text is present in many source documents. Deduplicating within a
    split is insufficient because the same label can leak between train,
    validation and test. The resolver indexes every trustworthy born-digital
    occurrence and assigns a single deterministic owner before rendering begins.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_revision: str,
        policy: str = "test_synthetic>validation_synthetic>train",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source_revision = str(source_revision)
        self.policy = str(policy)
        self.db = sqlite3.connect(self.path, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS owners(
              text_sha256 TEXT PRIMARY KEY,
              segment_key TEXT NOT NULL,
              split TEXT NOT NULL,
              priority INTEGER NOT NULL,
              document_id TEXT NOT NULL,
              source_line INTEGER NOT NULL,
              segment_index INTEGER NOT NULL
            );
            """
        )
        self.db.commit()
        identity = json.dumps(
            {"source_revision": self.source_revision, "policy": self.policy, "version": 2},
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if existing is not None and existing[0] != identity:
            self.db.close()
            raise RuntimeError("architecture resolver identity mismatch; use a new state directory")
        if existing is None:
            self.db.execute("INSERT INTO metadata(key,value) VALUES('identity',?)", (identity,))
            self.db.commit()

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def _owner_order(segment: ArchitectureSegment) -> tuple[int, str, int, int]:
        return (
            _SPLIT_PRIORITY[segment.split],
            segment.document_id,
            int(segment.source_line),
            int(segment.segment_index),
        )

    def build(self, corpus: "ArchitectureCorpus") -> dict[str, int | str]:
        completed = self.db.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()
        if completed is not None:
            return json.loads(completed[0])

        self.db.execute("DELETE FROM owners")
        gold_occurrences = 0
        for segment in corpus.iter_raw_segments():
            if segment.state != ArchitectureState.GOLD:
                continue
            gold_occurrences += 1
            order = self._owner_order(segment)
            current = self.db.execute(
                "SELECT priority,document_id,source_line,segment_index FROM owners WHERE text_sha256=?",
                (segment.text_sha256,),
            ).fetchone()
            if current is None or order < (int(current[0]), str(current[1]), int(current[2]), int(current[3])):
                self.db.execute(
                    """
                    INSERT INTO owners(text_sha256,segment_key,split,priority,document_id,source_line,segment_index)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(text_sha256) DO UPDATE SET
                      segment_key=excluded.segment_key,
                      split=excluded.split,
                      priority=excluded.priority,
                      document_id=excluded.document_id,
                      source_line=excluded.source_line,
                      segment_index=excluded.segment_index
                    """,
                    (
                        segment.text_sha256,
                        segment.segment_key,
                        segment.split,
                        order[0],
                        segment.document_id,
                        int(segment.source_line),
                        int(segment.segment_index),
                    ),
                )
        self.db.commit()
        canonical = int(self.db.execute("SELECT COUNT(*) FROM owners").fetchone()[0])
        digest = hashlib.sha256()
        for row in self.db.execute(
            "SELECT text_sha256,segment_key,split FROM owners ORDER BY text_sha256"
        ):
            digest.update("\x1f".join(map(str, row)).encode("utf-8"))
            digest.update(b"\n")
        summary: dict[str, int | str] = {
            "source_revision": self.source_revision,
            "policy": self.policy,
            "gold_occurrences": gold_occurrences,
            "canonical_gold_texts": canonical,
            "duplicate_gold_occurrences": gold_occurrences - canonical,
            "fingerprint": digest.hexdigest(),
        }
        self.db.execute(
            "INSERT INTO metadata(key,value) VALUES('summary',?)",
            (json.dumps(summary, sort_keys=True, separators=(",", ":")),),
        )
        self.db.commit()
        return summary

    def owner(self, text_sha256: str) -> tuple[str, str] | None:
        row = self.db.execute(
            "SELECT segment_key,split FROM owners WHERE text_sha256=?", (text_sha256,)
        ).fetchone()
        return None if row is None else (str(row[0]), str(row[1]))

    def is_canonical(self, segment: ArchitectureSegment) -> bool:
        owner = self.owner(segment.text_sha256)
        return owner is not None and owner[0] == segment.segment_key


class ArchitectureCorpus:
    def __init__(self, root: str | Path, *, max_graphemes: int = 112):
        self.root = Path(root)
        self.max_graphemes = int(max_graphemes)
        csv_path = self.root / "full_IIA_corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            self.metadata = {
                self._doc_id(row.get("corpus_index")): {k: str(v or "") for k, v in row.items()}
                for row in rows
            }

    @staticmethod
    def _doc_id(value: object) -> str:
        text = str(value or "").strip()
        return text[:-2] if text.endswith(".0") else text

    def document_split(self, document_id: str) -> str:
        bucket = int(
            hashlib.sha256(f"architecture-v9|{document_id}".encode()).hexdigest()[:8], 16
        ) % 1000
        if bucket < 960:
            return "train"
        if bucket < 980:
            return "validation_synthetic"
        return "test_synthetic"

    def iter_raw_segments(self, *, splits: set[str] | None = None) -> Iterator[ArchitectureSegment]:
        for path in sorted((self.root / "txt").glob("*.txt")):
            document_id = path.stem
            document_split = self.document_split(document_id)
            if splits is not None and document_split not in splits:
                continue
            metadata = self.metadata.get(document_id, {})
            origin = metadata.get("origin", "Unknown").strip() or "Unknown"
            try:
                raw = path.read_text(encoding="utf-8", errors="strict")
            except UnicodeDecodeError:
                yield ArchitectureSegment(
                    document_id, 0, 0, "", "", ArchitectureState.INVALID_UTF8,
                    "invalid_utf8", document_split, origin, metadata,
                )
                continue
            for line_number, raw_line in enumerate(raw.splitlines(), 1):
                if not raw_line.strip():
                    continue
                try:
                    parts = split_grapheme_safe(
                        raw_line, max_graphemes=self.max_graphemes, min_graphemes=2
                    )
                except LabelRejected as exc:
                    yield ArchitectureSegment(
                        document_id, line_number, 0, "", "", ArchitectureState.UNSAFE_UNICODE,
                        str(exc), document_split, origin, metadata,
                    )
                    continue
                for segment_index, part in enumerate(parts):
                    usable, reason, label = classify_usable_label(
                        part, maximum_graphemes=self.max_graphemes
                    )
                    if not usable or label is None:
                        yield ArchitectureSegment(
                            document_id,
                            line_number,
                            segment_index,
                            label.text if label else "",
                            label.text_sha256 if label else "",
                            ArchitectureState.REJECTED,
                            reason,
                            document_split,
                            origin,
                            metadata,
                        )
                        continue
                    if origin.casefold() == "born digital":
                        state = ArchitectureState.GOLD
                        reason = "born_digital"
                    else:
                        state = ArchitectureState.QUARANTINE
                        reason = f"origin_{origin or 'unknown'}"
                    yield ArchitectureSegment(
                        document_id,
                        line_number,
                        segment_index,
                        label.text,
                        label.text_sha256,
                        state,
                        reason,
                        document_split,
                        origin,
                        metadata,
                    )

    def iter_accounted_segments(
        self,
        *,
        splits: set[str] | None = None,
        resolver: ArchitectureTextResolver | None = None,
    ) -> Iterator[ArchitectureSegment]:
        seen: set[str] = set()
        for segment in self.iter_raw_segments(splits=splits):
            if segment.state != ArchitectureState.GOLD:
                yield segment
                continue
            if resolver is not None:
                owner = resolver.owner(segment.text_sha256)
                if owner is None:
                    raise RuntimeError(f"gold architecture text missing from resolver: {segment.text_sha256}")
                if owner[0] == segment.segment_key:
                    yield segment
                else:
                    yield ArchitectureSegment(
                        **{
                            **segment.__dict__,
                            "state": ArchitectureState.DUPLICATE,
                            "reason": f"global_exact_duplicate_owner={owner[0]}@{owner[1]}",
                        }
                    )
                continue
            if segment.text_sha256 in seen:
                yield ArchitectureSegment(
                    **{
                        **segment.__dict__,
                        "state": ArchitectureState.DUPLICATE,
                        "reason": "exact_duplicate",
                    }
                )
            else:
                seen.add(segment.text_sha256)
                yield segment
