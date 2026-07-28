from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download

from .architecture import (
    ArchitectureCorpus, ArchitectureSegment, ArchitectureState, ArchitectureTextResolver,
)
from .architecture_build import process_architecture_chunk
from .builder import process_source_rows
from .config import build_fingerprint
from .fonts import acquire_google_fonts, discover_fonts
from .generated_build import generate_page_specs, render_page_row, render_structured_row
from .identity import stable_token
from .metadata import export_registry_metadata, write_dataset_card, write_json_atomic
from .pipeline import process_downloaded_task
from .registry import DedupRegistry
from .render import TextRenderer
from .sources import SourceTask, discover_source_tasks, select_mini_tasks
from .structured import generate_structured_examples
from .writer import AtomicParquetWriter


@dataclass(frozen=True)
class BuildPaths:
    work: Path
    output: Path
    cache: Path
    fonts: Path
    state: Path
    logs: Path

    @classmethod
    def create(cls, work_dir: str | Path, *, mini: bool) -> "BuildPaths":
        base = Path(work_dir).expanduser().resolve()
        suffix = "mini" if mini else "full"
        obj = cls(
            work=base,
            output=base / f"output-{suffix}",
            cache=base / "cache",
            fonts=base / "fonts",
            state=base / f"state-{suffix}",
            logs=base / f"logs-{suffix}",
        )
        for path in (obj.work, obj.output, obj.cache, obj.fonts, obj.state, obj.logs):
            path.mkdir(parents=True, exist_ok=True)
        return obj


def _chunks(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    chunk: list[Any] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _token() -> str:
    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Hugging Face authentication is required; run `hf auth login`")
    return token


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def prepare_environment(config: dict[str, Any], paths: BuildPaths, *, token: str) -> tuple[HfApi, TextRenderer, list[SourceTask], dict]:
    if _free_gib(paths.work) < float(config.get("minimum_free_gib", 0)):
        raise RuntimeError(
            f"insufficient free space: {_free_gib(paths.work):.1f} GiB < {config['minimum_free_gib']} GiB"
        )
    api = HfApi(token=token)
    identity = api.whoami()
    expected_owner = str(config["output_repo"]).split("/", 1)[0]
    if identity.get("name") != expected_owner:
        raise RuntimeError(f"HF account mismatch: {identity.get('name')} != {expected_owner}")
    tasks, inventory = discover_source_tasks(api, config)
    architecture_source = config["sources"]["architecture"]
    info = api.dataset_info(architecture_source["repo_id"], revision=architecture_source["revision"])
    if info.sha != architecture_source["revision"]:
        raise RuntimeError("architecture revision mismatch")
    inventory["architecture"] = {
        "repo_id": architecture_source["repo_id"],
        "revision": architecture_source["revision"],
    }
    font_cfg = config["font_repo"]
    font_repo = acquire_google_fonts(
        paths.fonts,
        repo_url=font_cfg["url"],
        revision=font_cfg["revision"],
        sparse_paths=list(font_cfg["paths"]),
    )
    fonts = discover_fonts([font_repo], include_system=False)
    required = {"Alef", "Assistant", "Heebo", "Rubik", "David Libre", "Frank Ruhl Libre", "Noto Sans Hebrew", "Noto Serif Hebrew", "Noto Rashi Hebrew"}
    families = {font.family for font in fonts}
    missing = required - families
    if missing:
        raise RuntimeError(f"required pinned font families missing: {sorted(missing)}")
    renderer = TextRenderer(fonts)
    write_json_atomic(paths.output / "SOURCE_INVENTORY.json", inventory)
    write_json_atomic(paths.output / "FONT_MANIFEST.json", [
        {"family": f.family, "style": f.style, "sha256": f.sha256, "path": f.path.relative_to(font_repo).as_posix(), "has_gpos": f.has_gpos, "is_rashi": f.is_rashi}
        for f in fonts
    ])
    return api, renderer, tasks, inventory


def _download_task(task: SourceTask, *, token: str, cache: Path) -> Path:
    path = Path(hf_hub_download(
        repo_id=task.repo_id, repo_type="dataset", revision=task.revision,
        filename=task.path, token=token, cache_dir=cache,
    ))
    if task.size and path.stat().st_size != task.size:
        raise RuntimeError(f"source size mismatch after download: {task.path}")
    return path


def build_visual_sources(
    tasks: Sequence[SourceTask], *, paths: BuildPaths, registry: DedupRegistry,
    config: dict[str, Any], token: str, mini: bool,
) -> list[dict[str, Any]]:
    reports = []
    for task in tasks:
        local = _download_task(task, token=token, cache=paths.cache)
        if mini:
            if task.family == "htr" and "stage3_human_finetune" in task.path:
                limit = 80 if task.split == "train" else 40
            else:
                limit = 24
        else:
            limit = None
        reports.append(process_downloaded_task(
            task=task, local_path=local, registry=registry, output_root=paths.output,
            config=config, limit=limit,
        ))
    return reports


def _architecture_snapshot(config: dict[str, Any], *, token: str, cache: Path) -> Path:
    source = config["sources"]["architecture"]
    return Path(snapshot_download(
        repo_id=source["repo_id"], repo_type="dataset", revision=source["revision"],
        allow_patterns=["txt/*.txt", "full_IIA_corpus.csv", "README.md"],
        token=token, cache_dir=cache,
    ))


def build_architecture_natural(
    corpus: ArchitectureCorpus, *, resolver: ArchitectureTextResolver,
    renderer: TextRenderer, registry: DedupRegistry,
    paths: BuildPaths, config: dict[str, Any], mini: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    revision = config["sources"]["architecture"]["revision"]
    reports = []
    pools: dict[str, list[str]] = {"train": [], "validation_synthetic": [], "test_synthetic": []}
    split_order = ("test_synthetic", "validation_synthetic", "train")
    chunk_size = 40 if mini else int(config["architecture_chunk_size"])
    pool_limit = 200 if mini else int(config["page_pool_limit"])
    for split in split_order:
        iterator = corpus.iter_accounted_segments(splits={split}, resolver=resolver)
        if mini:
            def limited() -> Iterator[ArchitectureSegment]:
                for index, item in enumerate(iterator):
                    if index >= 120:
                        break
                    yield item
            selected: Iterable[ArchitectureSegment] = limited()
        else:
            selected = iterator
        for chunk_index, chunk in enumerate(_chunks(selected, chunk_size)):
            for segment in chunk:
                if segment.state == ArchitectureState.GOLD and len(pools[split]) < pool_limit:
                    pools[split].append(segment.text)
            source_key = f"architecture@{revision}:natural:{split}:{chunk_index:06d}"
            if registry.source_is_complete(source_key):
                report = registry.source_report(source_key)
                if report is None: raise RuntimeError("completed architecture chunk lacks report")
                reports.append(report); continue
            writer = AtomicParquetWriter(
                paths.output, config_name="architecture_synthetic_lines", split=split,
                source_token=stable_token(source_key, length=20), rows_per_shard=int(config["rows_per_shard"]),
            )
            reports.append(process_architecture_chunk(
                source_key=source_key, segments=chunk, renderer=renderer, registry=registry,
                writer=writer, output_root=paths.output, config=config,
                architecture_revision=revision,
            ))
    return reports, pools


def build_structured(
    *, renderer: TextRenderer, registry: DedupRegistry, paths: BuildPaths,
    config: dict[str, Any], mini: bool, pools: dict[str, list[str]],
) -> list[dict[str, Any]]:
    revision = config["sources"]["architecture"]["revision"]
    count = 240 if mini else int(config["architecture_structured_lines"])
    examples = list(generate_structured_examples(count, seed=20260726))
    reports = []
    chunk_size = 40 if mini else int(config["structured_chunk_size"])
    for split in ("test_synthetic", "validation_synthetic", "train"):
        subset = [row for row in examples if row.split == split]
        for example in subset:
            if len(pools[split]) < (200 if mini else int(config["page_pool_limit"])):
                pools[split].append(example.text)
        for chunk_index, chunk in enumerate(_chunks(subset, chunk_size)):
            source_key = f"architecture@{revision}:structured:{split}:{chunk_index:06d}"
            if registry.source_is_complete(source_key):
                report = registry.source_report(source_key)
                if report is None: raise RuntimeError("completed structured chunk lacks report")
                reports.append(report); continue
            writer = AtomicParquetWriter(
                paths.output, config_name="architecture_structured_lines", split=split,
                source_token=stable_token(source_key, length=20), rows_per_shard=int(config["rows_per_shard"]),
            )
            rows = (render_structured_row(item, renderer=renderer, architecture_revision=revision) for item in chunk)
            reports.append(process_source_rows(
                source_key=source_key, rows=rows, registry=registry, writer=writer,
                output_root=paths.output, config=config,
                report_metadata={"family":"architecture", "kind":"structured", "split":split},
            ))
    return reports


def build_pages(
    *, renderer: TextRenderer, registry: DedupRegistry, paths: BuildPaths,
    config: dict[str, Any], mini: bool, pools: dict[str, list[str]],
) -> list[dict[str, Any]]:
    revision = config["sources"]["architecture"]["revision"]
    for split, values in pools.items():
        if len(values) < 12:
            raise RuntimeError(f"page pool too small for {split}: {len(values)}")
    count = 20 if mini else int(config["architecture_pages"])
    specs = list(generate_page_specs(pools, count, seed=20260726))
    reports = []
    chunk_size = 5 if mini else int(config["page_chunk_size"])
    for split in ("test_synthetic", "validation_synthetic", "train"):
        subset = [row for row in specs if row.split == split]
        for chunk_index, chunk in enumerate(_chunks(subset, chunk_size)):
            source_key = f"architecture@{revision}:pages:{split}:{chunk_index:06d}"
            if registry.source_is_complete(source_key):
                report=registry.source_report(source_key)
                if report is None: raise RuntimeError("completed page chunk lacks report")
                reports.append(report); continue
            writer = AtomicParquetWriter(
                paths.output, config_name="architecture_synthetic_pages", split=split,
                source_token=stable_token(source_key, length=20), rows_per_shard=int(config["page_rows_per_shard"]),
            )
            rows = (render_page_row(item, renderer=renderer, architecture_revision=revision) for item in chunk)
            reports.append(process_source_rows(
                source_key=source_key, rows=rows, registry=registry, writer=writer,
                output_root=paths.output, config=config,
                report_metadata={"family":"architecture", "kind":"pages", "split":split},
            ))
    return reports


def run_local_build(config: dict[str, Any], *, mini: bool = False) -> tuple[BuildPaths, dict[str, Any]]:
    token = _token()
    paths = BuildPaths.create(config["work_dir"], mini=mini)
    fingerprint = build_fingerprint(config)
    write_json_atomic(paths.output / "BUILD_CONFIG.json", config)
    (paths.output / "BUILD_FINGERPRINT").write_text(fingerprint + "\n", encoding="ascii")
    api, renderer, tasks, inventory = prepare_environment(config, paths, token=token)
    selected = select_mini_tasks(tasks) if mini else list(tasks)
    evaluation_tasks = [task for task in selected if task.split != "train"]
    training_tasks = [task for task in selected if task.split == "train"]
    registry = DedupRegistry(paths.state / "registry.sqlite", build_fingerprint=fingerprint)
    resolver: ArchitectureTextResolver | None = None
    try:
        # Reserve all human/real/synthetic evaluation identities before any train row.
        evaluation_reports = build_visual_sources(
            evaluation_tasks,
            paths=paths,
            registry=registry,
            config=config,
            token=token,
            mini=mini,
        )

        architecture_root = _architecture_snapshot(config, token=token, cache=paths.cache)
        corpus = ArchitectureCorpus(
            architecture_root, max_graphemes=int(config["architecture_max_graphemes"])
        )
        architecture_revision = config["sources"]["architecture"]["revision"]
        resolver = ArchitectureTextResolver(
            paths.state / "architecture-text-resolver.sqlite",
            source_revision=architecture_revision,
        )
        resolver_summary = resolver.build(corpus)
        write_json_atomic(paths.output / "ARCHITECTURE_TEXT_RESOLVER.json", resolver_summary)

        # Architecture resolves test -> validation -> train globally, so its own
        # training rows can never pre-empt a synthetic evaluation owner.
        natural_reports, pools = build_architecture_natural(
            corpus,
            resolver=resolver,
            renderer=renderer,
            registry=registry,
            paths=paths,
            config=config,
            mini=mini,
        )
        structured_reports = build_structured(
            renderer=renderer,
            registry=registry,
            paths=paths,
            config=config,
            mini=mini,
            pools=pools,
        )
        page_reports = build_pages(
            renderer=renderer,
            registry=registry,
            paths=paths,
            config=config,
            mini=mini,
            pools=pools,
        )

        training_reports = build_visual_sources(
            training_tasks,
            paths=paths,
            registry=registry,
            config=config,
            token=token,
            mini=mini,
        )
        visual_reports = evaluation_reports + training_reports
        summary = export_registry_metadata(registry, paths.output)
        summary["mode"] = "mini" if mini else "full"
        summary["visual_source_reports"] = len(visual_reports)
        summary["visual_evaluation_reports"] = len(evaluation_reports)
        summary["visual_training_reports"] = len(training_reports)
        summary["architecture_natural_reports"] = len(natural_reports)
        summary["architecture_structured_reports"] = len(structured_reports)
        summary["architecture_page_reports"] = len(page_reports)
        summary["architecture_resolver"] = resolver_summary
        write_dataset_card(paths.output, summary)
        write_json_atomic(paths.output / "BUILD_SUMMARY.json", summary)
        return paths, summary
    finally:
        if resolver is not None:
            resolver.close()
        registry.close()
