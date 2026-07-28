from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

from .augment import LINE_PROFILES, PAGE_PROFILES
from .identity import stable_token
from .ingest import make_sample_row
from .render import PAGE_LAYOUTS
from .structured import StructuredExample
from .unicode_utils import namespace_key

_ARCH_REPO = "ssdataanalysis/hebrew-architecture-corpus"


def _seed(*parts: object) -> int:
    return int(hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def render_structured_row(example: StructuredExample, *, renderer, architecture_revision: str) -> dict:
    seed = _seed("structured-v9", example.group_id, example.index, example.text_sha256)
    profile = LINE_PROFILES[seed % len(LINE_PROFILES)]
    rendered = renderer.render_line(example.text, profile=profile, seed=seed, split=example.split)
    if rendered.visibility_fraction < 0.42:
        rendered = renderer.render_line(example.text, profile="clean_digital", seed=seed, split=example.split)
    if rendered.visibility_fraction < 0.42:
        raise RuntimeError("structured line visibility below threshold")
    sample_id = f"arch-structured-{stable_token(example.group_id, example.index, example.text_sha256)}"
    return make_sample_row(
        image_bytes=rendered.to_bytes(),
        image_path=f"{sample_id}.webp",
        text=example.text,
        sample_id=sample_id,
        split=example.split,
        task="line_recognition",
        granularity="line",
        modality="print",
        data_tier="gold",
        is_synthetic=True,
        sample_origin="synthetic",
        label_source="synthetic_ground_truth",
        label_trust="gold",
        provenance_reason="generated_from_curated_architecture_text",
        quality_tier="A",
        source_repo=_ARCH_REPO,
        source_revision=architecture_revision,
        source_path="generated/structured-lines",
        source_split=example.split,
        source_id=f"{example.group_id}:{example.index}",
        source_document=namespace_key(_ARCH_REPO, "structured-group", example.group_id),
        font_family=rendered.font.family,
        font_style=rendered.font.style,
        font_sha256=rendered.font.sha256,
        augmentation=rendered.metadata,
        provenance={"generator": "architecture-structured-v9", "template": example.template, "group_id": example.group_id},
    )


@dataclass(frozen=True)
class PageSpec:
    index: int
    group_id: str
    split: str
    lines: tuple[str, ...]
    profile: str
    layout: str
    seed: int


def _page_split(index: int) -> str:
    # Exact 96/2/2 distribution in every complete group of 50 pages.
    position = index % 50
    if position == 0:
        return "validation_synthetic"
    if position == 1:
        return "test_synthetic"
    return "train"


def generate_page_specs(
    pools: Mapping[str, Sequence[str]],
    count: int,
    *,
    seed: int = 20260726,
) -> Iterator[PageSpec]:
    required = {"train", "validation_synthetic", "test_synthetic"}
    missing = required - set(pools)
    if missing:
        raise ValueError(f"missing page text pools: {sorted(missing)}")
    for split in required:
        if len(pools[split]) < 12:
            raise ValueError(f"page pool {split} is too small")
    for index in range(int(count)):
        split = _page_split(index)
        page_seed = _seed("architecture-page-v9", seed, index)
        rng = random.Random(page_seed)
        available = pools[split]
        line_count = rng.randint(12, 24)
        start = page_seed % len(available)
        stride = 1 + (page_seed // max(len(available), 1)) % max(1, len(available) - 1)
        lines = tuple(available[(start + offset * stride) % len(available)] for offset in range(line_count))
        yield PageSpec(
            index=index,
            group_id=f"architecture-page-{seed}-{index:07d}",
            split=split,
            lines=lines,
            profile=PAGE_PROFILES[page_seed % len(PAGE_PROFILES)],
            layout=PAGE_LAYOUTS[(page_seed // len(PAGE_PROFILES)) % len(PAGE_LAYOUTS)],
            seed=page_seed,
        )


def render_page_row(spec: PageSpec, *, renderer, architecture_revision: str) -> dict:
    rendered = renderer.render_page(
        list(spec.lines), profile=spec.profile, layout=spec.layout, seed=spec.seed, split=spec.split
    )
    if float(rendered.metadata["visibility_fraction"]) < 0.35:
        rendered = renderer.render_page(
            list(spec.lines), profile="clean_digital", layout=spec.layout, seed=spec.seed, split=spec.split
        )
    annotations = sorted(rendered.annotations, key=lambda row: int(row["reading_order"]))
    if not annotations:
        raise RuntimeError("page renderer produced no annotations")
    text = "\n".join(str(row["text"]) for row in annotations)
    sample_id = f"arch-page-{stable_token(spec.group_id, text)}"
    return make_sample_row(
        image_bytes=rendered.to_bytes(),
        image_path=f"{sample_id}.webp",
        text=text,
        sample_id=sample_id,
        split=spec.split,
        task="page_transcription",
        granularity="page",
        modality="document",
        data_tier="gold",
        is_synthetic=True,
        sample_origin="synthetic",
        label_source="synthetic_ground_truth",
        label_trust="gold",
        provenance_reason="generated_from_curated_architecture_text",
        quality_tier="A",
        source_repo=_ARCH_REPO,
        source_revision=architecture_revision,
        source_path="generated/pages",
        source_split=spec.split,
        source_id=spec.group_id,
        source_document=namespace_key(_ARCH_REPO, "page-group", spec.group_id),
        source_page=namespace_key(_ARCH_REPO, "page", spec.group_id),
        augmentation=rendered.metadata,
        annotations=annotations,
        provenance={
            "generator": "architecture-pages-v9",
            "layout": spec.layout,
            "profile": spec.profile,
            "page_index": spec.index,
        },
    )
