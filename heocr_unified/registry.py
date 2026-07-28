from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class RegistryConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class AcceptDecision:
    accepted: bool
    reason: str


_LINE_TASKS = {"line_recognition", "page_transcription"}
_SPLIT_PRIORITY = {
    "test": 0,
    "validation": 1,
    "test_synthetic": 2,
    "validation_synthetic": 3,
    "train": 4,
}
_TIER_PRIORITY = {"gold": 0, "extended": 1, "quarantine": 2}


def split_priority(split: str) -> int:
    try:
        return _SPLIT_PRIORITY[str(split)]
    except KeyError as exc:
        raise ValueError(f"unknown split: {split}") from exc


def sample_priority(split: str, data_tier: str) -> int:
    try:
        tier = _TIER_PRIORITY[str(data_tier)]
    except KeyError as exc:
        raise ValueError(f"unknown data tier: {data_tier}") from exc
    return tier * 10 + split_priority(split)


def owner_scopes(data_tier: str) -> tuple[str, ...]:
    tier = str(data_tier)
    if tier == "gold":
        return ("gold",)
    if tier == "extended":
        return ("gold", "extended")
    if tier == "quarantine":
        return ()
    raise ValueError(f"unknown data tier: {data_tier}")


def train_tiers_blocked_by_evaluation(data_tier: str) -> tuple[str, ...]:
    tier = str(data_tier)
    if tier == "gold":
        return ("gold", "extended")
    if tier == "extended":
        return ("extended",)
    if tier == "quarantine":
        return ()
    raise ValueError(f"unknown data tier: {data_tier}")


class DedupRegistry:
    def __init__(self, path: str | Path, *, build_fingerprint: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._in_transaction = False
        self._create_schema()
        row = self.db.execute("SELECT value FROM metadata WHERE key='build_fingerprint'").fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO metadata(key,value) VALUES('build_fingerprint',?)",
                (build_fingerprint,),
            )
            self.db.commit()
        elif row[0] != build_fingerprint:
            self.db.close()
            raise RegistryConflict("build fingerprint mismatch; use a new work directory")

    def _create_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS samples(
              sample_id TEXT PRIMARY KEY,
              split TEXT NOT NULL,
              priority INTEGER NOT NULL,
              task TEXT NOT NULL,
              data_tier TEXT NOT NULL,
              sample_origin TEXT NOT NULL,
              byte_sha256 TEXT NOT NULL,
              visual_sha256 TEXT NOT NULL,
              text_sha256 TEXT NOT NULL,
              writer_key TEXT NOT NULL,
              document_key TEXT NOT NULL,
              page_key TEXT NOT NULL,
              source_key TEXT NOT NULL
            );
            DROP INDEX IF EXISTS samples_visual_unique;
            CREATE INDEX IF NOT EXISTS samples_visual_idx ON samples(visual_sha256,data_tier,split);
            CREATE INDEX IF NOT EXISTS samples_text_idx ON samples(task,text_sha256,split);
            CREATE INDEX IF NOT EXISTS samples_writer_idx ON samples(writer_key,split);
            CREATE INDEX IF NOT EXISTS samples_document_idx ON samples(document_key,split);
            CREATE INDEX IF NOT EXISTS samples_page_idx ON samples(page_key,split);
            CREATE TABLE IF NOT EXISTS evaluation_text_owners(
              task TEXT NOT NULL,
              text_sha256 TEXT NOT NULL,
              data_tier TEXT NOT NULL,
              split TEXT NOT NULL,
              priority INTEGER NOT NULL,
              sample_id TEXT NOT NULL,
              PRIMARY KEY(task,text_sha256,data_tier)
            );
            CREATE TABLE IF NOT EXISTS evaluation_entity_owners(
              kind TEXT NOT NULL,
              entity_key TEXT NOT NULL,
              data_tier TEXT NOT NULL,
              split TEXT NOT NULL,
              priority INTEGER NOT NULL,
              sample_id TEXT NOT NULL,
              PRIMARY KEY(kind,entity_key,data_tier)
            );
            CREATE TABLE IF NOT EXISTS text_counts(
              task TEXT NOT NULL,
              split TEXT NOT NULL,
              data_tier TEXT NOT NULL,
              text_sha256 TEXT NOT NULL,
              count INTEGER NOT NULL,
              PRIMARY KEY(task,split,data_tier,text_sha256)
            );
            CREATE TABLE IF NOT EXISTS rejects(reason TEXT PRIMARY KEY, count INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS source_tasks(
              source_key TEXT PRIMARY KEY, status TEXT NOT NULL, report_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts(
              path TEXT PRIMARY KEY, source_key TEXT NOT NULL, sha256 TEXT NOT NULL,
              rows INTEGER NOT NULL, bytes INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS architecture_ledger(
              segment_key TEXT PRIMARY KEY, document_id TEXT NOT NULL, source_line INTEGER NOT NULL,
              segment_index INTEGER NOT NULL, text_sha256 TEXT NOT NULL, source_state TEXT NOT NULL,
              outcome TEXT NOT NULL, reason TEXT NOT NULL, split TEXT NOT NULL, sample_id TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS architecture_ledger_outcome_idx
              ON architecture_ledger(source_state,outcome);
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def _maybe_commit(self) -> None:
        if not self._in_transaction:
            self.db.commit()

    def _reject(self, reason: str) -> AcceptDecision:
        self.db.execute(
            "INSERT INTO rejects(reason,count) VALUES(?,1) "
            "ON CONFLICT(reason) DO UPDATE SET count=count+1",
            (reason,),
        )
        self._maybe_commit()
        return AcceptDecision(False, reason)

    @staticmethod
    def is_evaluation_split(split: str) -> bool:
        return str(split) != "train"

    @staticmethod
    def _owner_reason(prefix: str, kind: str, split: str) -> str:
        if kind == "text":
            return f"{prefix}_text_owned_by_{split}"
        return f"{prefix}_{kind}_owned_by_{split}"

    def _check_owner(
        self,
        *,
        table: str,
        where: tuple[str, ...],
        values: tuple[str, ...],
        incoming_split: str,
        incoming_priority: int,
        kind: str,
    ) -> str | None:
        clause = " AND ".join(f"{column}=?" for column in where)
        row = self.db.execute(
            f"SELECT split,priority FROM {table} WHERE {clause} LIMIT 1", values
        ).fetchone()
        if row is None or str(row[0]) == incoming_split:
            return None
        owner_split, owner_priority = str(row[0]), int(row[1])
        if incoming_priority < owner_priority:
            raise RegistryConflict(
                f"higher-priority evaluation {kind} arrived late: "
                f"{incoming_split} should precede {owner_split}"
            )
        return self._owner_reason("evaluation", kind, owner_split)

    def _reserve_evaluation_owners(
        self,
        *,
        split: str,
        priority: int,
        task: str,
        text_sha256: str,
        writer_key: str,
        document_key: str,
        page_key: str,
        sample_id: str,
        data_tier: str,
    ) -> AcceptDecision | None:
        if data_tier == "quarantine":
            return None
        conflicts: list[str] = []
        for owner_tier in owner_scopes(data_tier):
            if task in _LINE_TASKS:
                reason = self._check_owner(
                    table="evaluation_text_owners",
                    where=("task", "text_sha256", "data_tier"),
                    values=(task, text_sha256, owner_tier),
                    incoming_split=split,
                    incoming_priority=priority,
                    kind="text",
                )
                if reason:
                    conflicts.append(reason)
            for kind, key in (("writer", writer_key), ("document", document_key), ("page", page_key)):
                if not key:
                    continue
                reason = self._check_owner(
                    table="evaluation_entity_owners",
                    where=("kind", "entity_key", "data_tier"),
                    values=(kind, key, owner_tier),
                    incoming_split=split,
                    incoming_priority=priority,
                    kind=kind,
                )
                if reason:
                    conflicts.append(reason)
        if conflicts:
            return self._reject(sorted(conflicts)[0])

        if task in _LINE_TASKS:
            self.db.execute(
                "INSERT OR IGNORE INTO evaluation_text_owners"
                "(task,text_sha256,data_tier,split,priority,sample_id) VALUES(?,?,?,?,?,?)",
                (task, text_sha256, data_tier, split, priority, sample_id),
            )
        for kind, key in (("writer", writer_key), ("document", document_key), ("page", page_key)):
            if key:
                self.db.execute(
                    "INSERT OR IGNORE INTO evaluation_entity_owners"
                    "(kind,entity_key,data_tier,split,priority,sample_id) VALUES(?,?,?,?,?,?)",
                    (kind, key, data_tier, split, priority, sample_id),
                )
        return None

    def reserve_evaluation_entity(
        self,
        *,
        split: str,
        task: str,
        text_sha256: str,
        visual_sha256: str,
        writer_key: str,
        document_key: str,
        sample_id: str,
        page_key: str = "",
        data_tier: str = "gold",
    ) -> AcceptDecision:
        """Reserve evaluation ownership before train ingestion.

        This compatibility entry point intentionally does not insert a sample row: it
        only establishes the evaluation ownership barriers used by ``accept_sample``.
        ``visual_sha256`` is accepted for API compatibility; visual ownership is
        established only once a complete sample is accepted, because a reservation
        without image bytes cannot safely prove a visual identity.
        """
        del visual_sha256
        if not self.is_evaluation_split(split):
            raise ValueError("evaluation reservation requires a non-train split")
        priority = sample_priority(split, data_tier)
        decision = self._reserve_evaluation_owners(
            split=split,
            priority=priority,
            task=task,
            text_sha256=text_sha256,
            writer_key=writer_key,
            document_key=document_key,
            page_key=page_key,
            sample_id=sample_id,
            data_tier=data_tier,
        )
        self._maybe_commit()
        return decision or AcceptDecision(True, "reserved")

    def accept_sample(
        self,
        *,
        sample_id: str,
        split: str,
        task: str,
        byte_sha256: str,
        visual_sha256: str,
        text_sha256: str,
        writer_key: str,
        document_key: str,
        page_key: str = "",
        source_key: str,
        text_cap: int,
        data_tier: str = "gold",
        sample_origin: str = "unknown",
    ) -> AcceptDecision:
        priority = sample_priority(split, data_tier)
        if self.db.execute("SELECT 1 FROM samples WHERE sample_id=?", (sample_id,)).fetchone():
            return self._reject("duplicate_sample_id")

        visuals = self.db.execute(
            "SELECT text_sha256,split,priority,sample_id,data_tier "
            "FROM samples WHERE visual_sha256=?",
            (visual_sha256,),
        ).fetchall()
        for visual in visuals:
            existing_text, existing_split, existing_priority, _, existing_tier = (
                str(visual[0]), str(visual[1]), int(visual[2]), str(visual[3]), str(visual[4])
            )
            if existing_tier == data_tier:
                if existing_text != text_sha256:
                    raise RegistryConflict("visual-label conflict")
                if priority < existing_priority:
                    raise RegistryConflict("higher-priority visual sample arrived late")
                return self._reject("duplicate_visual")
            if existing_text != text_sha256:
                if "quarantine" in {existing_tier, data_tier}:
                    if data_tier == "quarantine":
                        return self._reject("quarantine_visual_label_conflict")
                    continue
                raise RegistryConflict("cross-tier visual-label conflict")
            if split == "train" and existing_split != "train" and existing_tier in owner_scopes(data_tier):
                return self._reject("train_visual_reserved_by_evaluation")
            if split != "train" and existing_split == "train" and existing_tier in train_tiers_blocked_by_evaluation(data_tier):
                raise RegistryConflict("higher-priority evaluation visual arrived after train")

        is_eval = self.is_evaluation_split(split)
        scopes = owner_scopes(data_tier)
        if not is_eval:
            if task in _LINE_TASKS and scopes:
                placeholders = ",".join("?" for _ in scopes)
                if self.db.execute(
                    f"SELECT 1 FROM evaluation_text_owners WHERE task=? AND text_sha256=? "
                    f"AND data_tier IN ({placeholders}) LIMIT 1",
                    (task, text_sha256, *scopes),
                ).fetchone():
                    return self._reject("train_text_reserved_by_evaluation")
            for kind, key in (("writer", writer_key), ("document", document_key), ("page", page_key)):
                if not key or not scopes:
                    continue
                placeholders = ",".join("?" for _ in scopes)
                if self.db.execute(
                    f"SELECT 1 FROM evaluation_entity_owners WHERE kind=? AND entity_key=? "
                    f"AND data_tier IN ({placeholders}) LIMIT 1",
                    (kind, key, *scopes),
                ).fetchone():
                    return self._reject(f"train_{kind}_reserved_by_evaluation")
        elif data_tier != "quarantine":
            blocked_train_tiers = train_tiers_blocked_by_evaluation(data_tier)
            if task in _LINE_TASKS and blocked_train_tiers:
                placeholders = ",".join("?" for _ in blocked_train_tiers)
                late = self.db.execute(
                    f"SELECT 1 FROM samples WHERE task=? AND text_sha256=? AND split='train' "
                    f"AND data_tier IN ({placeholders}) LIMIT 1",
                    (task, text_sha256, *blocked_train_tiers),
                ).fetchone()
                if late:
                    raise RegistryConflict("higher-priority evaluation text arrived after train")
            for kind, key, column in (("writer", writer_key, "writer_key"), ("document", document_key, "document_key"), ("page", page_key, "page_key")):
                if not key or not blocked_train_tiers:
                    continue
                placeholders = ",".join("?" for _ in blocked_train_tiers)
                if self.db.execute(
                    f"SELECT 1 FROM samples WHERE {column}=? AND split='train' "
                    f"AND data_tier IN ({placeholders}) LIMIT 1",
                    (key, *blocked_train_tiers),
                ).fetchone():
                    raise RegistryConflict(f"higher-priority evaluation {kind} arrived after train")
            owner_decision = self._reserve_evaluation_owners(
                split=split, priority=priority, task=task, text_sha256=text_sha256,
                writer_key=writer_key, document_key=document_key, page_key=page_key,
                sample_id=sample_id, data_tier=data_tier,
            )
            if owner_decision is not None:
                return owner_decision

        count = self.db.execute(
            "SELECT count FROM text_counts WHERE task=? AND split=? AND data_tier=? AND text_sha256=?",
            (task, split, data_tier, text_sha256),
        ).fetchone()
        effective_cap = 1 if is_eval and task in _LINE_TASKS and data_tier != "quarantine" else int(text_cap)
        if count and int(count[0]) >= effective_cap:
            return self._reject("text_variant_cap")

        self.db.execute(
            "INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sample_id, split, priority, task, data_tier, sample_origin, byte_sha256,
             visual_sha256, text_sha256, writer_key, document_key, page_key, source_key),
        )
        self.db.execute(
            "INSERT INTO text_counts(task,split,data_tier,text_sha256,count) VALUES(?,?,?,?,1) "
            "ON CONFLICT(task,split,data_tier,text_sha256) DO UPDATE SET count=count+1",
            (task, split, data_tier, text_sha256),
        )
        self._maybe_commit()
        return AcceptDecision(True, "accepted")

    @contextlib.contextmanager
    def source_transaction(self, source_key: str) -> Iterator[None]:
        if self.source_is_complete(source_key):
            raise RegistryConflict(f"source already complete: {source_key}")
        if self._in_transaction:
            raise RegistryConflict("nested source transaction")
        self.db.execute("BEGIN IMMEDIATE")
        self._in_transaction = True
        try:
            yield
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        finally:
            self._in_transaction = False

    def mark_source_complete(self, source_key: str, report: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO source_tasks(source_key,status,report_json) VALUES(?,?,?)",
            (source_key, "complete", json.dumps(report, ensure_ascii=False, sort_keys=True)),
        )
        self._maybe_commit()

    def register_artifact(
        self, *, source_key: str, path: str, sha256: str, rows: int, bytes_count: int
    ) -> None:
        self.db.execute(
            "INSERT INTO artifacts(path,source_key,sha256,rows,bytes) VALUES(?,?,?,?,?)",
            (path, source_key, sha256, int(rows), int(bytes_count)),
        )
        self._maybe_commit()

    def source_is_complete(self, source_key: str) -> bool:
        row = self.db.execute(
            "SELECT status FROM source_tasks WHERE source_key=?", (source_key,)
        ).fetchone()
        return bool(row and row[0] == "complete")

    def source_report(self, source_key: str) -> dict | None:
        row = self.db.execute(
            "SELECT report_json FROM source_tasks WHERE source_key=?", (source_key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def artifacts_for_source(self, source_key: str) -> list[dict]:
        return [
            {"path": row[0], "sha256": row[1], "rows": int(row[2]), "bytes": int(row[3])}
            for row in self.db.execute(
                "SELECT path,sha256,rows,bytes FROM artifacts WHERE source_key=? ORDER BY path",
                (source_key,),
            )
        ]

    def record_architecture_segment(
        self,
        *,
        segment_key: str,
        document_id: str,
        source_line: int,
        segment_index: int,
        text_sha256: str,
        source_state: str,
        outcome: str,
        reason: str,
        split: str,
        sample_id: str,
    ) -> None:
        payload = (
            segment_key,
            document_id,
            int(source_line),
            int(segment_index),
            text_sha256,
            source_state,
            outcome,
            reason,
            split,
            sample_id,
        )
        existing = self.db.execute(
            "SELECT segment_key,document_id,source_line,segment_index,text_sha256,source_state,"
            "outcome,reason,split,sample_id FROM architecture_ledger WHERE segment_key=?",
            (segment_key,),
        ).fetchone()
        if existing is not None:
            if tuple(existing) != payload:
                raise RegistryConflict(f"architecture ledger conflict: {segment_key}")
            return
        self.db.execute("INSERT INTO architecture_ledger VALUES(?,?,?,?,?,?,?,?,?,?)", payload)
        self._maybe_commit()

    def architecture_ledger_summary(self) -> dict:
        outcomes = {
            str(row[0]): int(row[1])
            for row in self.db.execute(
                "SELECT outcome,COUNT(*) FROM architecture_ledger GROUP BY outcome ORDER BY outcome"
            )
        }
        total = int(self.db.execute("SELECT COUNT(*) FROM architecture_ledger").fetchone()[0])
        gold_total = int(
            self.db.execute(
                "SELECT COUNT(*) FROM architecture_ledger WHERE source_state='gold'"
            ).fetchone()[0]
        )
        gold_accounted = int(
            self.db.execute(
                "SELECT COUNT(*) FROM architecture_ledger WHERE source_state='gold' "
                "AND outcome IN "
                "('accepted','duplicate','evaluation_reserved')"
            ).fetchone()[0]
        )
        return {
            "total": total,
            "outcomes": outcomes,
            "gold_total": gold_total,
            "gold_accounted": gold_accounted,
        }

    def sample_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0])

    def text_exists(self, task: str, text_sha256: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM samples WHERE task=? AND text_sha256=? LIMIT 1",
            (task, text_sha256),
        ).fetchone() is not None

    def summary(self) -> dict:
        splits = {
            row[0]: int(row[1])
            for row in self.db.execute("SELECT split,COUNT(*) FROM samples GROUP BY split")
        }
        tasks = {
            row[0]: int(row[1])
            for row in self.db.execute("SELECT task,COUNT(*) FROM samples GROUP BY task")
        }
        tiers = {
            row[0]: int(row[1])
            for row in self.db.execute("SELECT data_tier,COUNT(*) FROM samples GROUP BY data_tier")
        }
        origins = {
            row[0]: int(row[1])
            for row in self.db.execute("SELECT sample_origin,COUNT(*) FROM samples GROUP BY sample_origin")
        }
        rejects = {
            row[0]: int(row[1])
            for row in self.db.execute("SELECT reason,count FROM rejects")
        }
        return {
            "samples": self.sample_count(),
            "splits": splits,
            "tasks": tasks,
            "tiers": tiers,
            "origins": origins,
            "rejects": rejects,
            "completed_sources": int(
                self.db.execute(
                    "SELECT COUNT(*) FROM source_tasks WHERE status='complete'"
                ).fetchone()[0]
            ),
            "artifacts": int(self.db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]),
        }
