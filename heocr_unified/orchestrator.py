from __future__ import annotations

import collections
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
from .pipeline import iter_downloaded_task_rows, process_downloaded_task
from .pointed import (
    POINTED_MANIFEST_PATH, POINTED_OUTPUT_CONFIG, PointedTextResolver,
    iter_verified_pointed_rows,
)
from .registry import DedupRegistry, sample_priority
from .render import TextRenderer
from .sources import SourceTask, discover_source_tasks, select_mini_tasks
from .structured import generate_structured_examples
from .unicode_utils import namespace_key
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


def reserve_evaluation_candidates(
    registry: DedupRegistry, candidates: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    required = {
        "sample_id", "split", "task", "text_sha256", "visual_sha256",
        "writer_id", "source_document", "source_page", "data_tier", "sample_origin",
    }
    rows = []
    for candidate in candidates:
        missing = required - set(candidate)
        if missing:
            raise ValueError(f"evaluation candidate lacks fields: {sorted(missing)}")
        if str(candidate["split"]) == "train":
            raise ValueError("evaluation reservation candidate cannot be train")
        rows.append(dict(candidate))
    rows.sort(key=lambda row: (
        sample_priority(str(row["split"]), str(row["data_tier"])),
        str(row["sample_id"]),
    ))
    rejects: collections.Counter[str] = collections.Counter()
    reserved = 0
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
        decision = registry.reserve_evaluation_entity(
            split=str(row["split"]),
            task=str(row["task"]),
            text_sha256=str(row["text_sha256"]),
            visual_sha256=str(row["visual_sha256"]),
            writer_key=str(row.get("writer_id") or ""),
            document_key=str(row.get("source_document") or ""),
            page_key=str(row.get("source_page") or ""),
            sample_id=str(row["sample_id"]),
            data_tier=str(row["data_tier"]),
            sample_origin=str(row["sample_origin"]),
            record_reject=False,
        )
        if decision.accepted:
            reserved += 1
        else:
            rejects[decision.reason] += 1
    return {
        "status": "PASS",
        "candidates": len(rows),
        "reserved": reserved,
        "rejected": len(rows) - reserved,
        "rejects": dict(sorted(rejects.items())),
        "fingerprint": digest.hexdigest(),
    }


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_file(inventory: dict[str, Any], family: str, path: str) -> dict[str, Any]:
    for item in inventory.get(family, {}).get("files", []):
        if str(item.get("path")) == str(path):
            return dict(item)
    raise RuntimeError(f"required source file is missing: {family}:{path}")


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
    pointed_path = str(config.get("pointed_manifest_path") or POINTED_MANIFEST_PATH)
    pointed_file = _inventory_file(inventory, "ocr", pointed_path)
    inventory["verified_pointed"] = {
        "repo_id": config["sources"]["ocr"]["repo_id"],
        "revision": config["sources"]["ocr"]["revision"],
        "path": pointed_path,
        "size": int(pointed_file.get("size") or 0),
        "blob_id": str(pointed_file.get("blob_id") or ""),
    }
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


def _task_limit(task: SourceTask, *, mini: bool) -> int | None:
    if not mini:
        return None
    if task.family == "htr" and "stage3_human_finetune" in task.path:
        return 80 if task.split == "train" else 40
    return 24


def _visual_evaluation_candidates(
    tasks: Sequence[SourceTask], *, paths: BuildPaths, token: str, mini: bool
) -> Iterator[dict[str, Any]]:
    for task in tasks:
        if task.split == "train":
            raise ValueError("visual evaluation candidate source cannot be train")
        local = _download_task(task, token=token, cache=paths.cache)
        for row in iter_downloaded_task_rows(
            task=task, local_path=local, limit=_task_limit(task, mini=mini)
        ):
            if str(row["split"]) == "train":
                raise RuntimeError(f"evaluation source emitted train row: {task.source_key}")
            if str(row.get("data_tier") or "") == "quarantine":
                continue
            yield {
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "task": str(row["task"]),
                "text_sha256": str(row["text_sha256"]),
                "visual_sha256": str(row["visual_sha256"]),
                "writer_id": str(row.get("writer_id") or ""),
                "source_document": str(row.get("source_document") or ""),
                "source_page": str(row.get("source_page") or ""),
                "data_tier": str(row["data_tier"]),
                "sample_origin": str(row["sample_origin"]),
                "source_family": task.family,
                "source_path": task.path,
            }


def _selected_architecture_segments(
    corpus: ArchitectureCorpus, *, resolver: ArchitectureTextResolver, split: str, mini: bool
) -> Iterator[ArchitectureSegment]:
    iterator = corpus.iter_accounted_segments(splits={split}, resolver=resolver)
    if not mini:
        yield from iterator
        return
    for index, item in enumerate(iterator):
        if index >= 120:
            break
        yield item


def _generated_evaluation_candidates(
    *,
    pointed_resolver: PointedTextResolver,
    corpus: ArchitectureCorpus,
    architecture_resolver: ArchitectureTextResolver,
    config: dict[str, Any],
    mini: bool,
) -> Iterator[dict[str, Any]]:
    pointed_limit = int(config["pointed_mini_per_split"]) if mini else None
    for split in ("test_synthetic", "validation_synthetic"):
        for entry in pointed_resolver.iter_entries(split, limit=pointed_limit):
            yield {
                "sample_id": f"pointed-reservation-{entry.text_sha256}",
                "split": split,
                "task": "line_recognition",
                "text_sha256": entry.text_sha256,
                "visual_sha256": "",
                "writer_id": "",
                "source_document": "",
                "source_page": "",
                "data_tier": "gold",
                "sample_origin": "synthetic",
                "source_family": "verified_pointed",
            }
        for segment in _selected_architecture_segments(
            corpus, resolver=architecture_resolver, split=split, mini=mini
        ):
            if segment.state != ArchitectureState.GOLD:
                continue
            yield {
                "sample_id": f"arch-{stable_token(segment.segment_key, segment.text_sha256)}",
                "split": split,
                "task": "line_recognition",
                "text_sha256": segment.text_sha256,
                "visual_sha256": "",
                "writer_id": "",
                "source_document": namespace_key(
                    "ssdataanalysis/hebrew-architecture-corpus", "document", segment.document_id
                ),
                "source_page": "",
                "data_tier": "gold",
                "sample_origin": "synthetic",
                "source_family": "architecture_natural",
            }

    structured_count = 240 if mini else int(config["architecture_structured_lines"])
    for example in generate_structured_examples(structured_count, seed=20260726):
        if example.split == "train":
            continue
        yield {
            "sample_id": f"arch-structured-reservation-{example.group_id}-{example.index}",
            "split": example.split,
            "task": "line_recognition",
            "text_sha256": example.text_sha256,
            "visual_sha256": "",
            "writer_id": "",
            "source_document": namespace_key(
                "ssdataanalysis/hebrew-architecture-corpus", "structured-group", example.group_id
            ),
            "source_page": "",
            "data_tier": "gold",
            "sample_origin": "synthetic",
            "source_family": "architecture_structured",
        }


def build_evaluation_reservations(
    *,
    evaluation_tasks: Sequence[SourceTask],
    pointed_resolver: PointedTextResolver,
    corpus: ArchitectureCorpus,
    architecture_resolver: ArchitectureTextResolver,
    paths: BuildPaths,
    registry: DedupRegistry,
    config: dict[str, Any],
    token: str,
    mini: bool,
) -> dict[str, Any]:
    candidates = list(_visual_evaluation_candidates(
        evaluation_tasks, paths=paths, token=token, mini=mini
    ))
    candidates.extend(_generated_evaluation_candidates(
        pointed_resolver=pointed_resolver,
        corpus=corpus,
        architecture_resolver=architecture_resolver,
        config=config,
        mini=mini,
    ))
    report = reserve_evaluation_candidates(registry, candidates)
    report["visual_candidates"] = sum(1 for row in candidates if row.get("visual_sha256"))
    report["generated_candidates"] = len(candidates) - int(report["visual_candidates"])
    return report


def build_visual_sources(
    tasks: Sequence[SourceTask], *, paths: BuildPaths, registry: DedupRegistry,
    config: dict[str, Any], token: str, mini: bool,
) -> list[dict[str, Any]]:
    reports = []
    for task in tasks:
        local = _download_task(task, token=token, cache=paths.cache)
        limit = _task_limit(task, mini=mini)
        reports.append(process_downloaded_task(
            task=task, local_path=local, registry=registry, output_root=paths.output,
            config=config, limit=limit,
        ))
    return reports


def _download_pointed_manifest(
    config: dict[str, Any], *, token: str, cache: Path, inventory: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    source = config["sources"]["ocr"]
    path = str(config.get("pointed_manifest_path") or POINTED_MANIFEST_PATH)
    expected = _inventory_file(inventory, "ocr", path)
    local = Path(hf_hub_download(
        repo_id=source["repo_id"], repo_type="dataset", revision=source["revision"],
        filename=path, token=token, cache_dir=cache,
    ))
    size = local.stat().st_size
    expected_size = int(expected.get("size") or 0)
    if expected_size and size != expected_size:
        raise RuntimeError(f"pointed manifest size mismatch: {size} != {expected_size}")
    return local, {
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "path": path,
        "bytes": size,
        "sha256": _file_sha256(local),
        "blob_id": str(expected.get("blob_id") or ""),
    }


def build_verified_pointed(
    *,
    resolver: PointedTextResolver,
    renderer: TextRenderer,
    registry: DedupRegistry,
    paths: BuildPaths,
    config: dict[str, Any],
    manifest_info: dict[str, Any],
    splits: Sequence[str],
    mini: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    variants = int(config["pointed_variants_per_text"])
    chunk_size = 6 if mini else int(config["pointed_chunk_size"])
    per_split_limit = int(config["pointed_mini_per_split"]) if mini else None
    for split in splits:
        entries = resolver.iter_entries(split, limit=per_split_limit)
        produced_any = False
        for chunk_index, chunk in enumerate(_chunks(entries, chunk_size)):
            produced_any = True
            source_key = (
                f"{manifest_info['repo_id']}@{manifest_info['revision']}:"
                f"verified-pointed:{split}:{chunk_index:06d}:{manifest_info['sha256']}"
            )
            if registry.source_is_complete(source_key):
                report = registry.source_report(source_key)
                if report is None:
                    raise RuntimeError("completed verified-pointed chunk lacks report")
                reports.append(report)
                continue
            writer = AtomicParquetWriter(
                paths.output,
                config_name=POINTED_OUTPUT_CONFIG,
                split=split,
                source_token=stable_token(source_key, length=20),
                rows_per_shard=int(config["rows_per_shard"]),
            )
            rows = iter_verified_pointed_rows(
                chunk,
                variants=variants,
                renderer=renderer,
                source_revision=str(manifest_info["revision"]),
                manifest_sha256=str(manifest_info["sha256"]),
            )
            reports.append(process_source_rows(
                source_key=source_key,
                rows=rows,
                registry=registry,
                writer=writer,
                output_root=paths.output,
                config=config,
                report_metadata={
                    "family": "ocr",
                    "kind": "verified_pointed_rerender",
                    "source_path": manifest_info["path"],
                    "source_bytes": int(manifest_info["bytes"]),
                    "source_sha256": manifest_info["sha256"],
                    "split": split,
                    "variants_per_text": variants,
                },
            ))
        if not produced_any:
            raise RuntimeError(f"verified pointed resolver produced no entries for {split}")
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
        selected: Iterable[ArchitectureSegment] = _selected_architecture_segments(
            corpus, resolver=resolver, split=split, mini=mini
        )
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
    pointed_resolver: PointedTextResolver | None = None
    try:
        # Build immutable text-owner indices before ingesting any evaluation row.
        # This makes ownership independent of source/shard/thread order.
        pointed_manifest, pointed_manifest_info = _download_pointed_manifest(
            config, token=token, cache=paths.cache, inventory=inventory
        )
        inventory["verified_pointed"] = dict(pointed_manifest_info)
        write_json_atomic(paths.output / "SOURCE_INVENTORY.json", inventory)
        pointed_resolver = PointedTextResolver(
            paths.state / "verified-pointed-resolver.sqlite",
            source_revision=str(pointed_manifest_info["revision"]),
            manifest_sha256=str(pointed_manifest_info["sha256"]),
            max_graphemes=int(config["pointed_max_graphemes"]),
        )
        pointed_summary = pointed_resolver.build(pointed_manifest)
        write_json_atomic(paths.output / "VERIFIED_POINTED_AUDIT.json", pointed_summary)

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

        reservation_report = build_evaluation_reservations(
            evaluation_tasks=evaluation_tasks,
            pointed_resolver=pointed_resolver,
            corpus=corpus,
            architecture_resolver=resolver,
            paths=paths,
            registry=registry,
            config=config,
            token=token,
            mini=mini,
        )
        write_json_atomic(paths.output / "EVALUATION_RESERVATIONS.json", reservation_report)

        evaluation_reports = build_visual_sources(
            evaluation_tasks,
            paths=paths,
            registry=registry,
            config=config,
            token=token,
            mini=mini,
        )
        pointed_evaluation_reports = build_verified_pointed(
            resolver=pointed_resolver, renderer=renderer, registry=registry, paths=paths,
            config=config, manifest_info=pointed_manifest_info,
            splits=("test_synthetic", "validation_synthetic"), mini=mini,
        )

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

        pointed_training_reports = build_verified_pointed(
            resolver=pointed_resolver, renderer=renderer, registry=registry, paths=paths,
            config=config, manifest_info=pointed_manifest_info, splits=("train",), mini=mini,
        )
        training_reports = build_visual_sources(
            training_tasks,
            paths=paths,
            registry=registry,
            config=config,
            token=token,
            mini=mini,
        )

        visual_reports = (
            evaluation_reports + pointed_evaluation_reports +
            pointed_training_reports + training_reports
        )
        summary = export_registry_metadata(registry, paths.output)
        summary["mode"] = "mini" if mini else "full"
        summary["evaluation_reservations"] = reservation_report
        summary["visual_source_reports"] = len(visual_reports)
        summary["visual_evaluation_reports"] = len(evaluation_reports)
        summary["visual_training_reports"] = len(training_reports)
        summary["verified_pointed_evaluation_reports"] = len(pointed_evaluation_reports)
        summary["verified_pointed_training_reports"] = len(pointed_training_reports)
        summary["verified_pointed_audit"] = pointed_summary
        summary["verified_pointed_manifest"] = pointed_manifest_info
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
        if pointed_resolver is not None:
            pointed_resolver.close()
        registry.close()
