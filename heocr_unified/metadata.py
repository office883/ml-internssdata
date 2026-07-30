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
    extended_all = sorted(name for name in discovered if name.endswith("_extended"))
    extended_words = [name for name in extended_all if name.removesuffix("_extended") == "modern_print_words"]
    extended_characters = [
        name for name in extended_all
        if name.removesuffix("_extended") == "handwriting_real_characters"
    ]
    extended_pages = [
        name for name in extended_all
        if name.removesuffix("_extended") == "architecture_synthetic_pages"
    ]
    extended_special = set(extended_words + extended_characters + extended_pages)
    extended_lines = [name for name in extended_all if name not in extended_special]
    quarantine = sorted(name for name in discovered if name.endswith("_quarantine"))

    config_lines: list[str] = []
    declared_names: set[str] = set()

    def declare(name: str, data_files: dict[str, list[str]], *, default: bool = False) -> None:
        if name in declared_names or not data_files:
            return
        config_lines.extend(_yaml_data_config(name, data_files, default=default))
        declared_names.add(name)

    # Curated convenience views come first.  They make the safe/default training
    # path obvious while preserving task boundaries between lines, words,
    # characters, and pages.
    declare(
        "unified_recognition_lines", _merge_config_patterns(discovered, line_configs), default=True
    )
    declare("modern_print_words", _merge_config_patterns(discovered, words))
    declare("handwriting_real_characters", _merge_config_patterns(discovered, characters))
    declare("document_pages", _merge_config_patterns(discovered, pages))
    declare("extended_recognition_lines", _merge_config_patterns(discovered, extended_lines))
    declare("extended_words", _merge_config_patterns(discovered, extended_words))
    declare("extended_characters", _merge_config_patterns(discovered, extended_characters))
    declare("extended_document_pages", _merge_config_patterns(discovered, extended_pages))
    declare("quarantine_audit", _merge_config_patterns(discovered, quarantine))

    # Also expose every physical config exactly once.  This is important for
    # curriculum learning, per-domain benchmarking, and reproducible sampling:
    # users can select a precise source family instead of being forced through a
    # merged convenience view.  Names already used by a convenience config are
    # intentionally not duplicated.
    for name in sorted(discovered):
        declare(name, discovered[name])
    if not config_lines:
        raise RuntimeError("cannot write dataset card before Parquet data exists")

    front_matter = "\n".join([
        "---",
        "language:",
        "- he",
        "license: other",
        "task_categories:",
        "- image-to-text",
        "pretty_name: Hebrew OCR Unified SOTA-Capable v15",
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
- `extended_recognition_lines` הוא opt-in לשורות שימושיות אך פחות ודאיות, כגון diffusion או מקורות Tier B.
- `extended_words`, `extended_characters` ו־`extended_document_pages` מבודדים פיזית; הם אינם יכולים להיכנס בטעות למסלול השורות.
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


def write_usage_documents(output_root: str | Path, *, config: dict[str, Any]) -> None:
    """Write source/licensing policy and an operationally safe training recipe."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for family, source in sorted(config["sources"].items()):
        source_rows.append(
            f"- `{family}` — `{source['repo_id']}` at commit `{source['revision']}`"
        )
    source_policy = """# Source and Trust Policy

This private dataset card intentionally declares `license: other`. The unified
repository does **not** relicense all upstream material under one blanket license.
Every row retains its source repository, immutable revision, source path, trust
tier, provenance classification, and any available upstream license metadata.

## Pinned sources

""" + "\n".join(source_rows) + """

## Trust tiers

- **gold**: human transcriptions or deterministic synthetic ground truth with
  explicit provenance and successful integrity checks.
- **extended**: opt-in material that may be useful but is not safe as the default,
  including diffusion-generated handwriting, Tier-B material, and Samaritan-script
  imagery whose labels use Hebrew Unicode but whose glyph shapes are not ordinary
  square Hebrew.
- **quarantine**: audit-only material with `recommended_sampling_weight = 0`.
  It is never included in the default training configuration.

## Architecture corpus

Only text explicitly marked **Born digital** is accepted as gold ground truth for
new rendering. Text extracted from scanned documents by an older OCR system is
preserved in quarantine rather than being redrawn as if its OCR output were truth.
Each Tier-A occurrence receives an explicit ledger outcome; nothing disappears
silently.

## Samaritan material

Samaritan handwriting is isolated in `extended` configurations. It may be selected
for specialised historical research, but it is excluded from the core Hebrew OCR
training mixture to avoid teaching the model that Samaritan glyph forms are standard
modern Hebrew glyphs.

## Distribution and compliance

Before sharing or commercial redistribution, review the per-row provenance and
upstream terms. Keeping the repository private does not erase upstream obligations.
"""
    training_recipe = """# Training Recipe

The repository is a single source of truth, but it deliberately exposes separate
configs so incompatible learning units cannot be mixed by accident.

## Recommended curriculum

1. **Core line recognition pretraining** — train on `unified_recognition_lines`.
   This is gold-only and is the default config.
2. **Optional robustness expansion** — selectively add
   `extended_recognition_lines`, respecting each row's
   `recommended_sampling_weight`. Do not blindly use every extended sample at 1×.
3. **Human HTR fine-tuning** — up-weight human handwriting train rows near the end
   of training so they are not drowned by synthetic volume.
4. **Keep human validation and human test locked**. Never use either for prompt
   construction, tokenizer fitting, hard-negative generation, checkpoint selection
   beyond the intended validation protocol, or language-model post-correction fitting.
5. **Page training is a separate objective** — use `document_pages` for detection,
   reading order, boxes, baselines, and full-page transcription. Do not mix page
   images into a line recognizer without an explicit multi-task architecture.
6. **Words and characters are separate tasks** — use `modern_print_words` and
   `handwriting_real_characters` only with task-aware sampling/heads.
7. `quarantine_audit` is not training data. Its rows carry zero weight by contract.

## Suggested sampling

- Start with task-balanced batches rather than proportional-to-row-count batches.
- Preserve `recommended_sampling_weight` within each task/tier.
- Cap repeated exact labels and avoid letting large synthetic families erase the
  contribution of human data.
- For modern Hebrew document OCR, begin without specialised Samaritan material;
  add it only for a separately reported historical model or ablation.

## Evaluation

Report at least:

- Grapheme-CER and codepoint CER;
- WER where word segmentation is meaningful;
- exact accuracy for numbers, dates, money, units, addresses, and identifiers;
- mixed Hebrew/Latin/number BiDi accuracy;
- niqqud/combining-mark accuracy;
- separate modern print, historical print, handwriting, Rashi, page-layout, and
  degradation buckets;
- performance on the locked human validation/test sets.

A state-of-the-art claim requires comparison against current baselines on frozen,
source-disjoint test sets. Training loss or synthetic-test accuracy is not enough.
"""
    (root / "SOURCE_POLICY.md").write_text(source_policy, encoding="utf-8")
    (root / "TRAINING_RECIPE.md").write_text(training_recipe, encoding="utf-8")
