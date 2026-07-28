from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceTask:
    family: str
    repo_id: str
    path: str
    split: str
    revision: str = ""
    size: int = 0

    @property
    def source_key(self) -> str:
        return f"{self.repo_id}@{self.revision}:{self.path}"


@dataclass(frozen=True)
class MappedSource:
    output_config: str
    task: str
    granularity: str
    modality: str
    data_tier: str
    synthetic: bool
    output_split: str


def normal_output_split(split: str, *, synthetic: bool, human: bool = False) -> str:
    value = split.lower()
    if value == "train":
        return "train"
    base = "validation" if value.startswith("val") else "test"
    if human:
        return base
    return f"{base}_synthetic" if synthetic else base


def classify_source_task(task: SourceTask) -> MappedSource:
    path = task.path
    split = task.split
    if task.family == "htr":
        stage = Path(path).parts[0]
        if stage == "stage3_human_finetune":
            return MappedSource("handwriting_real_lines", "line_recognition", "line", "handwriting", "gold", False, normal_output_split(split, synthetic=False, human=True))
        if stage == "stage2_diffusion_augmentation":
            return MappedSource("handwriting_diffusion_lines", "line_recognition", "line", "handwriting", "extended", True, normal_output_split(split, synthetic=True))
        return MappedSource("handwriting_synthetic_lines", "line_recognition", "line", "handwriting", "gold", True, normal_output_split(split, synthetic=True))
    if task.family == "ocr":
        parts = Path(path).parts
        config = parts[1]
        mapping = {
            "modern_print_lines": ("modern_print_lines", "line_recognition", "line", "print"),
            "modern_print_words": ("modern_print_words", "word_recognition", "word", "print"),
            "synthetic_handwriting_lines": ("handwriting_synthetic_lines", "line_recognition", "line", "handwriting"),
            "real_handwriting_characters": ("handwriting_real_characters", "character_recognition", "character", "handwriting"),
            "historical_handwriting_lines": ("handwriting_historical_lines", "line_recognition", "line", "handwriting"),
            "biblical_pointed_lines": ("biblical_pointed_lines", "line_recognition", "line", "print"),
            "historical_print_lines": ("historical_print_lines", "line_recognition", "line", "print"),
            "lexicographic_print_lines": ("lexicographic_print_lines", "line_recognition", "line", "print"),
            "rabbinic_print_lines": ("rabbinic_rashi_lines", "line_recognition", "line", "print"),
        }
        if config not in mapping:
            raise ValueError(f"unknown OCR config: {config}")
        output, output_task, granularity, modality = mapping[config]
        synthetic = config in {"modern_print_lines", "modern_print_words", "synthetic_handwriting_lines", "biblical_pointed_lines", "historical_print_lines", "lexicographic_print_lines", "rabbinic_print_lines"}
        tier = "gold" if config != "synthetic_handwriting_lines" else "extended"
        return MappedSource(output, output_task, granularity, modality, tier, synthetic, normal_output_split(split, synthetic=synthetic))
    if task.family == "foundation":
        name = Path(path).name
        output_split = normal_output_split(split, synthetic=True)
        if "rashi" in name:
            return MappedSource("rabbinic_rashi_lines", "line_recognition", "line", "print", "gold", True, output_split)
        if "niqqud" in name:
            return MappedSource("biblical_pointed_lines", "line_recognition", "line", "print", "gold", True, output_split)
        if "structured" in name or "mixed-bidi" in name:
            return MappedSource("structured_bidi_lines", "line_recognition", "line", "print", "gold", True, output_split)
        return MappedSource("modern_print_lines", "line_recognition", "line", "print", "gold", True, output_split)
    raise ValueError(f"unknown source family: {task.family}")


def sort_evaluation_first(tasks: Iterable[SourceTask]) -> list[SourceTask]:
    def key(task: SourceTask) -> tuple[int, int, str, str]:
        mapped = classify_source_task(task)
        is_human = task.family == "htr" and "stage3_human_finetune" in task.path
        is_eval = mapped.output_split != "train"
        # Highest-trust human evaluation is reserved first. Test owns an exact
        # duplicate over validation; all training sources are processed last.
        family_rank = 0 if (is_human and is_eval) else (1 if is_eval else 2)
        split_rank = 0 if mapped.output_split.startswith("test") else (
            1 if mapped.output_split.startswith("validation") else 2
        )
        return family_rank, split_rank, task.repo_id, task.path
    return sorted(tasks, key=key)


def discover_source_tasks(api, config: dict) -> tuple[list[SourceTask], dict]:
    tasks: list[SourceTask] = []
    inventory: dict[str, dict] = {}
    for family in ("foundation", "htr", "ocr"):
        source = config["sources"][family]
        repo_id = source["repo_id"]
        revision = source["revision"]
        info = api.dataset_info(repo_id, revision=revision)
        if info.sha != revision:
            raise RuntimeError(f"resolved revision mismatch for {repo_id}: {info.sha} != {revision}")
        files: list[dict] = []
        for item in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision=revision, recursive=True, expand=True):
            if item.__class__.__name__ != "RepoFile":
                continue
            path = str(item.path)
            size = int(getattr(item, "size", 0) or 0)
            files.append({"path": path, "size": size, "blob_id": str(getattr(item, "blob_id", "") or "")})
            if family == "htr" and path.endswith(".parquet"):
                split = Path(path).name.split("-", 1)[0]
                tasks.append(SourceTask(family, repo_id, path, split, revision, size))
            elif family == "ocr" and path.startswith("webdataset/") and path.endswith(".tar"):
                parts = Path(path).parts
                split = parts[2]
                tasks.append(SourceTask(family, repo_id, path, split, revision, size))
            elif family == "foundation" and path.startswith("shards/") and path.endswith(".tar"):
                name = Path(path).name
                split = "validation" if name.startswith("validation") else ("test" if name.startswith("test") else "train")
                tasks.append(SourceTask(family, repo_id, path, split, revision, size))
        inventory[family] = {"repo_id": repo_id, "revision": revision, "file_count": len(files), "files": files}
    required_ocr = {
        "modern_print_lines", "modern_print_words", "synthetic_handwriting_lines",
        "real_handwriting_characters", "historical_handwriting_lines", "biblical_pointed_lines",
        "historical_print_lines", "lexicographic_print_lines", "rabbinic_print_lines",
    }
    found_ocr = {Path(task.path).parts[1] for task in tasks if task.family == "ocr"}
    missing = required_ocr - found_ocr
    if missing:
        raise RuntimeError(f"missing OCR configs: {sorted(missing)}")
    human_splits = {task.split for task in tasks if task.family == "htr" and task.path.startswith("stage3_human_finetune/")}
    if human_splits != {"train", "validation", "test"}:
        raise RuntimeError(f"incomplete human HTR splits: {sorted(human_splits)}")
    return sort_evaluation_first(tasks), inventory


def select_mini_tasks(tasks: Iterable[SourceTask]) -> list[SourceTask]:
    tasks = list(tasks)
    selected: dict[tuple, SourceTask] = {}
    for task in tasks:
        if task.family == "ocr":
            config = Path(task.path).parts[1]
            key = ("ocr", config, task.split)
            prior = selected.get(key)
            if prior is None or task.size < prior.size:
                selected[key] = task
        elif task.family == "htr":
            stage = Path(task.path).parts[0]
            key = ("htr", stage, task.split)
            prior = selected.get(key)
            if prior is None or task.size < prior.size:
                selected[key] = task
        elif task.family == "foundation":
            name = Path(task.path).name
            if task.split != "train":
                key = ("foundation", task.split)
            elif "natural" in name:
                key = ("foundation", "natural")
            elif "structured" in name:
                key = ("foundation", "structured")
            elif "mixed-bidi" in name:
                key = ("foundation", "mixed")
            elif "niqqud" in name:
                key = ("foundation", "niqqud")
            elif "rashi" in name:
                key = ("foundation", "rashi")
            else:
                continue
            prior = selected.get(key)
            if prior is None or task.size < prior.size:
                selected[key] = task
    return sort_evaluation_first(selected.values())
