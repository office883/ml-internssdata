from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

ROOT = pathlib.Path('/tmp/heocr-v15-mounted-mini')
SRC = ROOT / 'source'
WORK = ROOT / 'work'
shutil.rmtree(ROOT, ignore_errors=True)
ROOT.mkdir(parents=True)


def run(cmd: list[str], **kwargs: Any) -> None:
    print('RUN', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def out(cmd: list[str], **kwargs: Any) -> str:
    return subprocess.check_output(cmd, text=True, **kwargs).strip()

run(['git', 'clone', '--depth', '1', '--branch', 'heocr-v15-source-verified',
     'https://github.com/office883/ml-internssdata.git', str(SRC)])
commit = out(['git', 'rev-parse', 'HEAD'], cwd=SRC)
tree = out(['git', 'rev-parse', 'HEAD^{tree}'], cwd=SRC)
print('SOURCE_COMMIT', commit, flush=True)
print('SOURCE_TREE', tree, flush=True)
assert commit == '215635a87f58bb35e2ad7ed73af2b2837f72abd9', commit
assert tree == '71e7378e3b9a8a33c5e92736071801eb3c6abed4', tree
sys.path.insert(0, str(SRC))

from heocr_unified import orchestrator
from heocr_unified.config import load_config
from heocr_unified.finalize import finalize_local_release
from heocr_unified.fonts import acquire_google_fonts, discover_fonts, pointed_coverage_families
from heocr_unified.metadata import write_json_atomic
from heocr_unified.render import TextRenderer
from heocr_unified.sources import SourceTask, sort_evaluation_first

MOUNTS = {
    'foundation': pathlib.Path('/mnt/foundation'),
    'htr': pathlib.Path('/mnt/htr'),
    'ocr': pathlib.Path('/mnt/ocr'),
    'architecture': pathlib.Path('/mnt/architecture'),
}
EXPECTED_REVISIONS = {
    'foundation': '1e277f98b17ad2efb9e6b13abbb7a06afe569a03',
    'htr': 'ec4c7074ce2b3edc79889b00319e200d129eecf7',
    'ocr': 'ce4d1c347bd4e8b98a23f23256b0ecf01fa663c5',
    'architecture': '58e7dd53a6caa42191252601f97b1dee96c3d765',
}

for name, root in MOUNTS.items():
    assert root.is_dir(), f'missing mounted source: {name}:{root}'


def discover_mounted(config: dict[str, Any]) -> tuple[list[SourceTask], dict[str, Any]]:
    tasks: list[SourceTask] = []
    inventory: dict[str, Any] = {}
    for family in ('foundation', 'htr', 'ocr'):
        source = config['sources'][family]
        revision = str(source['revision'])
        assert revision == EXPECTED_REVISIONS[family]
        root = MOUNTS[family]
        files: list[dict[str, Any]] = []
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            size = int(path.stat().st_size)
            files.append({'path': rel, 'size': size, 'blob_id': ''})
            if family == 'htr' and rel.endswith('.parquet'):
                split = pathlib.Path(rel).name.split('-', 1)[0]
                tasks.append(SourceTask(family, source['repo_id'], rel, split, revision, size))
            elif family == 'ocr' and rel.startswith('webdataset/') and rel.endswith('.tar'):
                parts = pathlib.Path(rel).parts
                split = parts[2]
                tasks.append(SourceTask(family, source['repo_id'], rel, split, revision, size))
            elif family == 'foundation' and rel.startswith('shards/') and rel.endswith('.tar'):
                name = pathlib.Path(rel).name
                split = 'validation' if name.startswith('validation') else ('test' if name.startswith('test') else 'train')
                tasks.append(SourceTask(family, source['repo_id'], rel, split, revision, size))
        inventory[family] = {
            'repo_id': source['repo_id'],
            'revision': revision,
            'file_count': len(files),
            'files': files,
            'transport': 'hf_jobs_private_readonly_volume',
            'mount': str(root),
        }
    required_ocr = {
        'modern_print_lines', 'modern_print_words', 'synthetic_handwriting_lines',
        'real_handwriting_characters', 'historical_handwriting_lines', 'biblical_pointed_lines',
        'historical_print_lines', 'lexicographic_print_lines', 'rabbinic_print_lines',
    }
    found_ocr = {pathlib.Path(task.path).parts[1] for task in tasks if task.family == 'ocr'}
    assert found_ocr == required_ocr, (found_ocr, required_ocr)
    human_splits = {
        task.split for task in tasks
        if task.family == 'htr' and task.path.startswith('stage3_human_finetune/')
    }
    assert human_splits == {'train', 'validation', 'test'}, human_splits
    return sort_evaluation_first(tasks), inventory


def mounted_prepare_environment(config: dict[str, Any], paths: orchestrator.BuildPaths, *, token: str):
    tasks, inventory = discover_mounted(config)
    architecture_source = config['sources']['architecture']
    assert architecture_source['revision'] == EXPECTED_REVISIONS['architecture']
    architecture_files = [
        p.relative_to(MOUNTS['architecture']).as_posix()
        for p in sorted(MOUNTS['architecture'].rglob('*')) if p.is_file()
    ]
    assert 'full_IIA_corpus.csv' in architecture_files
    assert any(p.startswith('txt/') and p.endswith('.txt') for p in architecture_files)
    inventory['architecture'] = {
        'repo_id': architecture_source['repo_id'],
        'revision': architecture_source['revision'],
        'file_count': len(architecture_files),
        'transport': 'hf_jobs_private_readonly_volume',
        'mount': str(MOUNTS['architecture']),
    }
    font_cfg = config['font_repo']
    font_repo = acquire_google_fonts(
        paths.fonts,
        repo_url=font_cfg['url'],
        revision=font_cfg['revision'],
        sparse_paths=list(font_cfg['paths']),
    )
    fonts = discover_fonts([font_repo], include_system=False)
    required = {
        'Alef', 'Assistant', 'Heebo', 'Rubik', 'David Libre',
        'Frank Ruhl Libre', 'Noto Sans Hebrew', 'Noto Serif Hebrew',
        'Noto Rashi Hebrew', 'Miriam Libre', 'Varela Round',
        'Secular One', 'Suez One', 'Bellefair', 'Amatic SC',
    }
    families = {font.family for font in fonts}
    missing = required - families
    assert not missing, sorted(missing)
    full_pointed = pointed_coverage_families(fonts)
    assert len(full_pointed) >= 3, sorted(full_pointed)
    renderer = TextRenderer(fonts)
    write_json_atomic(paths.output / 'SOURCE_INVENTORY.json', inventory)
    write_json_atomic(paths.output / 'FONT_MANIFEST.json', [
        {
            'family': f.family,
            'style': f.style,
            'sha256': f.sha256,
            'path': f.path.relative_to(font_repo).as_posix(),
            'has_gpos': f.has_gpos,
            'is_rashi': f.is_rashi,
        }
        for f in fonts
    ])
    return SimpleNamespace(), renderer, tasks, inventory


def mounted_download_task(task: SourceTask, *, token: str, cache: pathlib.Path) -> pathlib.Path:
    family_by_repo = {
        'ssdataanalysis/hebrew-ocr-foundation-v1': 'foundation',
        'ssdataanalysis/hebrew-htr-curated-v1': 'htr',
        'ssdataanalysis/hebrew-ocr-corpus': 'ocr',
    }
    root = MOUNTS[family_by_repo[task.repo_id]]
    path = root / task.path
    assert path.is_file(), path
    if task.size:
        assert path.stat().st_size == task.size, (path, path.stat().st_size, task.size)
    return path


def mounted_pointed_manifest(config: dict[str, Any], *, token: str, cache: pathlib.Path, inventory: dict[str, Any]):
    source = config['sources']['ocr']
    rel = str(config['pointed_manifest_path'])
    path = MOUNTS['ocr'] / rel
    assert path.is_file(), path
    expected = orchestrator._inventory_file(inventory, 'ocr', rel)
    assert int(expected.get('size') or 0) == path.stat().st_size
    return path, {
        'repo_id': source['repo_id'],
        'revision': source['revision'],
        'path': rel,
        'bytes': path.stat().st_size,
        'sha256': orchestrator._file_sha256(path),
        'blob_id': '',
        'transport': 'hf_jobs_private_readonly_volume',
    }


orchestrator._token = lambda: 'mounted-private-volume'
orchestrator.prepare_environment = mounted_prepare_environment
orchestrator._download_task = mounted_download_task
orchestrator._download_pointed_manifest = mounted_pointed_manifest
orchestrator._architecture_snapshot = lambda config, *, token, cache: MOUNTS['architecture']

config = load_config(
    SRC / 'config.json',
    overrides={
        'work_dir': str(WORK),
        'upload': False,
        'minimum_free_gib': 0,
    },
)
paths, build_summary = orchestrator.run_local_build(config, mini=True)
ready = finalize_local_release(
    paths.output,
    registry_path=paths.state / 'registry.sqlite',
    config=config,
    mini=True,
)

qa = json.loads((paths.output / 'qa' / 'QA_REPORT.json').read_text())
corr = json.loads((paths.output / 'qa' / 'CORRUPTION_REPORT.json').read_text())
pointed = json.loads((paths.output / 'VERIFIED_POINTED_AUDIT.json').read_text())
reservations = json.loads((paths.output / 'EVALUATION_RESERVATIONS.json').read_text())
resolver = json.loads((paths.output / 'ARCHITECTURE_TEXT_RESOLVER.json').read_text())

assert ready['status'] == 'PASS'
assert ready['integrity_errors'] == 0
assert ready['leakage_errors'] == 0
assert qa['integrity_errors'] == 0
assert qa['leakage_errors'] == 0
assert qa['required_configs_present'] is True
assert qa['required_source_families_present'] is True
assert corr['status'] == 'PASS'
assert corr['test_count'] >= 7
assert all(item['status'] == 'PASS' for item in corr['tests'])
assert pointed['status'] == 'PASS'
assert pointed['canonical_texts'] >= 50000
assert reservations['status'] == 'PASS'
assert reservations['reserved'] + reservations['rejected'] == reservations['candidates']
assert resolver['canonical_gold_texts'] > 0
assert resolver['gold_occurrences'] >= resolver['canonical_gold_texts']

print('READY=' + json.dumps(ready, sort_keys=True), flush=True)
print('QA_CORE=' + json.dumps({k: qa.get(k) for k in [
    'all_rows', 'gold_rows', 'extended_rows', 'quarantine_rows',
    'gold_train_rows', 'gold_recognition_lines', 'human_train',
    'human_validation', 'human_test', 'architecture_natural_lines',
    'architecture_primary_lines', 'architecture_extra_variants',
    'architecture_structured_lines', 'pages', 'mixed_bidi',
    'with_digits', 'with_combining_marks', 'verified_pointed_rerender',
    'integrity_errors', 'leakage_errors',
]}, sort_keys=True), flush=True)
print('CORRUPTION=' + json.dumps(corr, sort_keys=True), flush=True)
print('POINTED=' + json.dumps(pointed, sort_keys=True), flush=True)
print('RESERVATIONS=' + json.dumps(reservations, sort_keys=True), flush=True)
print('RESOLVER=' + json.dumps(resolver, sort_keys=True), flush=True)
print('HEOCR_V15_MOUNTED_EXACT_REAL_FOUR_SOURCE_MINI_RELEASE_PASS', flush=True)
