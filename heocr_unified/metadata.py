from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from .registry import DedupRegistry


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def export_registry_metadata(registry: DedupRegistry, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    source_reports = [
        {"source_key": row[0], "status": row[1], "report": json.loads(row[2])}
        for row in registry.db.execute(
            "SELECT source_key,status,report_json FROM source_tasks ORDER BY source_key"
        )
    ]
    artifacts = [
        {"path": row[0], "source_key": row[1], "sha256": row[2], "rows": int(row[3]), "bytes": int(row[4])}
        for row in registry.db.execute(
            "SELECT path,source_key,sha256,rows,bytes FROM artifacts ORDER BY path"
        )
    ]
    write_json_atomic(metadata / "source-reports.json", source_reports)
    write_json_atomic(metadata / "artifacts.json", artifacts)
    ledger_path = metadata / "architecture-ledger.jsonl.gz"
    temp = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=9) as handle:
        for row in registry.db.execute(
            "SELECT segment_key,document_id,source_line,segment_index,text_sha256,source_state," 
            "outcome,reason,split,sample_id FROM architecture_ledger ORDER BY segment_key"
        ):
            handle.write(json.dumps({
                "segment_key": row[0], "document_id": row[1], "source_line": int(row[2]),
                "segment_index": int(row[3]), "text_sha256": row[4], "source_state": row[5],
                "outcome": row[6], "reason": row[7], "split": row[8], "sample_id": row[9],
            }, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp, ledger_path)
    summary = {
        "registry": registry.summary(),
        "architecture": registry.architecture_ledger_summary(),
        "source_reports": len(source_reports),
        "artifacts": len(artifacts),
    }
    write_json_atomic(metadata / "registry-summary.json", summary)
    return summary


def _discover_parquet_patterns(root: Path) -> dict[str, dict[str, list[str]]]:
    discovered: dict[str, dict[str, list[str]]] = {}
    data_root = root / "data"
    if not data_root.is_dir():
        return discovered
    for config_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        split_map: dict[str, list[str]] = {}
        for split_dir in sorted(path for path in config_dir.iterdir() if path.is_dir()):
            if any(split_dir.glob("*.parquet")):
                split_map.setdefault(split_dir.name, []).append(
                    f"data/{config_dir.name}/{split_dir.name}/*.parquet"
                )
        if split_map:
            discovered[config_dir.name] = split_map
    return discovered


def _merge_config_patterns(
    discovered: dict[str, dict[str, list[str]]], names: list[str]
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for name in sorted(names):
        for split, paths in discovered.get(name, {}).items():
            merged.setdefault(split, []).extend(paths)
    return {split: sorted(set(paths)) for split, paths in sorted(merged.items())}


def _yaml_data_config(name: str, data_files: dict[str, list[str]], *, default: bool = False) -> list[str]:
    if not data_files:
        return []
    lines = [f"- config_name: {name}"]
    if default:
        lines.append("  default: true")
    lines.append("  data_files:")
    split_order = {"train": 0, "validation": 1, "test": 2, "validation_synthetic": 3, "test_synthetic": 4}
    for split in sorted(data_files, key=lambda item: (split_order.get(item, 99), item)):
        lines.append(f"  - split: {split}")
        lines.append("    path:")
        for path in data_files[split]:
            lines.append(f"    - {path}")
    return lines


def write_dataset_card(output_root: str | Path, summary: dict[str, Any]) -> None:
    root = Path(output_root)
    discovered = _discover_parquet_patterns(root)
    base = sorted(
        name for name in discovered
        if not name.endswith("_extended") and not name.endswith("_quarantine")
    )
    words = [name for name in base if name == "modern_print_words"]
    characters = [name for name in base if name == "handwriting_real_characters"]
    pages = [name for name in base if name == "architecture_synthetic_pages"]
    special = set(words + characters + pages)
    line_configs = [name for name in base if name not in special]
    extended = sorted(name for name in discovered if name.endswith("_extended"))
    quarantine = sorted(name for name in discovered if name.endswith("_quarantine"))

    config_lines: list[str] = []
    config_lines += _yaml_data_config(
        "unified_recognition_lines", _merge_config_patterns(discovered, line_configs), default=True
    )
    config_lines += _yaml_data_config(
        "modern_print_words", _merge_config_patterns(discovered, words)
    )
    config_lines += _yaml_data_config(
        "handwriting_real_characters", _merge_config_patterns(discovered, characters)
    )
    config_lines += _yaml_data_config(
        "document_pages", _merge_config_patterns(discovered, pages)
    )
    config_lines += _yaml_data_config(
        "extended_recognition_lines", _merge_config_patterns(discovered, extended)
    )
    config_lines += _yaml_data_config(
        "quarantine_audit", _merge_config_patterns(discovered, quarantine)
    )
    if not config_lines:
        raise RuntimeError("cannot write dataset card before Parquet data exists")

    front_matter = "\n".join([
        "---",
        "language:",
        "- he",
        "license: other",
        "task_categories:",
        "- image-to-text",
        "pretty_name: Hebrew OCR Unified SOTA-Capable v11",
        "tags:",
        "- ocr",
        "- htr",
        "- hebrew",
        "- rtl",
        "- bidi",
        "- synthetic-data",
        "configs:",
        *config_lines,
        "---",
        "",
    ])
    card = front_matter + """# Hebrew OCR Unified Dataset

מאגר פרטי מאוחד לאימון OCR ו־HTR בעברית. תוויות הטקסט נשמרות ב־Unicode logical order וב־NFC.

## שכבות אמון

- `unified_recognition_lines` הוא config ברירת המחדל ומכיל **gold בלבד**.
- `extended_recognition_lines` הוא opt-in לחומר שימושי אך פחות ודאי, כגון diffusion או מקורות Tier B.
- `quarantine_audit` נשמר לביקורת ולמחקר בלבד; לכל שורה בו משקל אימון אפס והוא לעולם אינו נכלל בברירת המחדל.
- דפי מסמך, מילים ותווים בודדים מופרדים ל־configs ייעודיים כדי למנוע ערבוב יחידות אימון.

## מקורות

- `ssdataanalysis/hebrew-ocr-foundation-v1`
- `ssdataanalysis/hebrew-htr-curated-v1`
- `ssdataanalysis/hebrew-ocr-corpus`
- `ssdataanalysis/hebrew-architecture-corpus`

כל מקור נעול ל־commit ומתועד ב־`BUILD_CONFIG.json` וב־`SOURCE_INVENTORY.json`.

## אזהרה מדעית

המאגר נבנה להיות SOTA-capable, אך SOTA הוא תוצאה של מודל שנמדד מול benchmark אנושי נעול — לא תכונה אוטומטית של קובץ נתונים.

## QA

הבנייה מסומנת כמוכנה רק בנוכחות `LOCAL_READY.json`. סיכום הרישום:

```json
""" + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n"
    (root / "README.md").write_text(card, encoding="utf-8")
