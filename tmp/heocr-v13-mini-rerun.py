# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "huggingface-hub==1.24.0",
#   "pyarrow==21.0.0",
#   "Pillow==12.2.0",
#   "numpy==2.3.5",
#   "opencv-python-headless==4.13.0.92",
#   "fonttools==4.63.0",
#   "regex==2026.5.9",
#   "pandas==2.2.3",
#   "pytest==9.0.2",
# ]
# ///
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/tmp/heocr-v13-final-cleanroom-rerun")
SRC = ROOT / "source"
WORK = ROOT / "work"
EXPECTED_TREE = "1ee1b63cfa8b1c8d68c7f216d513eb9aa1e0e7cb"


def run(command: list[str], **kwargs) -> None:
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, **kwargs)


def output(command: list[str], **kwargs) -> str:
    return subprocess.check_output(command, text=True, **kwargs).strip()


shutil.rmtree(ROOT, ignore_errors=True)
ROOT.mkdir(parents=True)
run([
    "git", "clone", "--depth", "1", "--branch", "heocr-v13-source-verified",
    "https://github.com/office883/ml-internssdata.git", str(SRC),
])
tree = output(["git", "rev-parse", "HEAD^{tree}"], cwd=SRC)
commit = output(["git", "rev-parse", "HEAD"], cwd=SRC)
assert tree == EXPECTED_TREE, tree
print("SOURCE_COMMIT", commit, flush=True)
print("SOURCE_TREE", tree, flush=True)

run([sys.executable, "-m", "compileall", "-q", "heocr_unified", "tests"], cwd=SRC)
run(["bash", "-n", "RUN_ME.command"], cwd=SRC)
run(["git", "diff", "--check"], cwd=SRC)
run([sys.executable, "-m", "pytest", "-q", "--disable-warnings"], cwd=SRC)
print("V13_ALL_TESTS_PASS", flush=True)

environment = {
    **os.environ,
    "PYTHONUNBUFFERED": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}
run([
    sys.executable, "-m", "heocr_unified", "mini",
    "--config", "config.json",
    "--work-dir", str(WORK),
    "--minimum-free-gib", "0",
    "--no-upload",
], cwd=SRC, env=environment)
run([
    sys.executable, "-m", "heocr_unified", "verify", "--mini",
    "--config", "config.json",
    "--work-dir", str(WORK),
    "--minimum-free-gib", "0",
    "--no-upload",
], cwd=SRC, env=environment)

out_root = WORK / "output-mini"
required = [
    "LOCAL_READY.json",
    "qa/QA_REPORT.json",
    "qa/CORRUPTION_REPORT.json",
    "RELEASE_MANIFEST.json",
    "previews/PREVIEW_INVENTORY.json",
    "BUILD_SUMMARY.json",
    "SOURCE_INVENTORY.json",
    "FONT_MANIFEST.json",
    "ARCHITECTURE_TEXT_RESOLVER.json",
    "VERIFIED_POINTED_AUDIT.json",
    "EVALUATION_RESERVATIONS.json",
    "README.md",
    "SOURCE_POLICY.md",
    "TRAINING_RECIPE.md",
]
for relative in required:
    path = out_root / relative
    assert path.is_file() and path.stat().st_size > 0, relative

ready = json.loads((out_root / "LOCAL_READY.json").read_text())
qa = json.loads((out_root / "qa/QA_REPORT.json").read_text())
corruption = json.loads((out_root / "qa/CORRUPTION_REPORT.json").read_text())
pointed = json.loads((out_root / "VERIFIED_POINTED_AUDIT.json").read_text())
reservations = json.loads((out_root / "EVALUATION_RESERVATIONS.json").read_text())
resolver = json.loads((out_root / "ARCHITECTURE_TEXT_RESOLVER.json").read_text())

assert ready["status"] == "PASS" and ready["mode"] == "mini"
assert ready["integrity_errors"] == 0 and ready["leakage_errors"] == 0
assert ready["total_rows"] > 0 and ready["preview_sheets"] > 0
assert qa["integrity_errors"] == 0 and qa["leakage_errors"] == 0
assert qa["required_configs_present"] is True
assert qa["required_source_families_present"] is True
assert corruption["status"] == "PASS"
assert corruption["test_count"] >= 7
assert all(item["status"] == "PASS" for item in corruption["tests"])
assert pointed["status"] == "PASS" and pointed["canonical_texts"] >= 50000
assert reservations["status"] == "PASS"
assert reservations["reserved"] + reservations["rejected"] == reservations["candidates"]
assert resolver["canonical_gold_texts"] > 0
assert resolver["gold_occurrences"] >= resolver["canonical_gold_texts"]

qa_fields = [
    "all_rows", "gold_rows", "extended_rows", "quarantine_rows",
    "gold_train_rows", "gold_recognition_lines", "human_train",
    "human_validation", "human_test", "architecture_natural_lines",
    "architecture_primary_lines", "architecture_extra_variants",
    "architecture_structured_lines", "pages", "mixed_bidi", "with_digits",
    "with_combining_marks", "verified_pointed_rerender",
    "integrity_errors", "leakage_errors",
]
print("READY=" + json.dumps(ready, sort_keys=True), flush=True)
print("QA_CORE=" + json.dumps({key: qa.get(key) for key in qa_fields}, sort_keys=True), flush=True)
print("CORRUPTION=" + json.dumps(corruption, sort_keys=True), flush=True)
print("POINTED=" + json.dumps(pointed, sort_keys=True), flush=True)
print("RESERVATIONS=" + json.dumps(reservations, sort_keys=True), flush=True)
print("RESOLVER=" + json.dumps(resolver, sort_keys=True), flush=True)
print("HEOCR_V13_EXACT_REAL_MINI_RELEASE_PASS", flush=True)
