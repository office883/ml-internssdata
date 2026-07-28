from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import features
import pytest

from heocr_unified.architecture import ArchitectureCorpus
from heocr_unified.architecture_build import process_architecture_chunk
from heocr_unified.config import load_config
from heocr_unified.registry import DedupRegistry
from heocr_unified.render import TextRenderer


@dataclass(frozen=True)
class _Artifact:
    path: Path
    rows: int
    bytes: int
    sha256: str


class _Writer:
    def __init__(self, root: Path): self.root=root; self.rows=[]; self.artifacts=[]
    def add(self,row): self.rows.append(row)
    def finish(self):
        path=self.root/"arch.bin"; path.parent.mkdir(parents=True,exist_ok=True)
        payload="\n".join(row["sample_id"] for row in self.rows).encode(); path.write_bytes(payload)
        a=_Artifact(path,len(self.rows),len(payload),hashlib.sha256(payload).hexdigest()); self.artifacts=[a]; return [a]
    def cleanup(self):
        for a in self.artifacts: a.path.unlink(missing_ok=True)


def _corpus(tmp_path: Path) -> ArchitectureCorpus:
    root=tmp_path/"corpus"; (root/"txt").mkdir(parents=True)
    pd.DataFrame([
        {"corpus_index":"1","origin":"Born digital","year":2020,"title":"א"},
        {"corpus_index":"2","origin":"Scanned","year":1980,"title":"ב"},
    ]).to_csv(root/"full_IIA_corpus.csv",index=False)
    (root/"txt"/"1.txt").write_text("שורת אדריכלות תקינה 17\nשורת אדריכלות תקינה 17\n",encoding="utf-8")
    (root/"txt"/"2.txt").write_text("טקסט OCR ישן שאינו תווית זהב\n",encoding="utf-8")
    return ArchitectureCorpus(root)


@pytest.mark.skipif(not features.check_feature("raqm"), reason="Pillow lacks RAQM")
def test_architecture_chunk_renders_gold_and_accounts_every_segment(tmp_path: Path) -> None:
    config=load_config(None,overrides={"work_dir":str(tmp_path),"upload":False})
    registry=DedupRegistry(tmp_path/"registry.sqlite",build_fingerprint="fp")
    renderer=TextRenderer.from_system_fonts()
    report=process_architecture_chunk(
        source_key="architecture:chunk:0", segments=list(_corpus(tmp_path).iter_accounted_segments()),
        renderer=renderer, registry=registry, writer=_Writer(tmp_path/"out"),
        output_root=tmp_path, config=config, architecture_revision="b"*40,
    )
    ledger=registry.architecture_ledger_summary()
    assert report["segments"] == 3
    assert report["accepted"] == 1
    assert ledger["total"] == 3
    assert ledger["gold_total"] == 1
    assert ledger["gold_accounted"] == 1
    assert ledger["outcomes"]["accepted"] == 1
    assert ledger["outcomes"]["duplicate"] == 1
    assert ledger["outcomes"]["quarantined"] == 1
    stored = registry.db.execute(
        "SELECT data_tier,sample_origin FROM samples WHERE source_key=?",
        ("architecture:chunk:0",),
    ).fetchall()
    assert stored == [("gold", "synthetic")]


def test_gold_architecture_render_failure_is_fatal_not_quarantined(tmp_path: Path) -> None:
    class BrokenRenderer:
        def render_line(self, *args, **kwargs):
            raise RuntimeError("no single font covers all visible code points")

    corpus = _corpus(tmp_path)
    config = load_config(None, overrides={"work_dir": str(tmp_path), "upload": False})
    registry = DedupRegistry(tmp_path / "registry-fail.sqlite", build_fingerprint="fp")
    with pytest.raises(RuntimeError, match="gold architecture segment cannot be rendered"):
        process_architecture_chunk(
            source_key="architecture:chunk:fail",
            segments=list(corpus.iter_accounted_segments()),
            renderer=BrokenRenderer(), registry=registry, writer=_Writer(tmp_path / "out-fail"),
            output_root=tmp_path, config=config, architecture_revision="b" * 40,
        )
    assert registry.architecture_ledger_summary()["gold_total"] == 0


def _single_gold_segment(split: str = "train"):
    from heocr_unified.architecture import ArchitectureSegment, ArchitectureState
    from heocr_unified.unicode_utils import normalize_label_strict
    text = "תכנית קומה 17 שטח 248.5 מ״ר"
    return ArchitectureSegment(
        document_id="doc-1",
        source_line=1,
        segment_index=0,
        text=text,
        text_sha256=normalize_label_strict(text).text_sha256,
        origin="Born digital",
        metadata={},
        split=split,
        state=ArchitectureState.GOLD,
        reason="born_digital",
    )


class _FakeRendered:
    def __init__(self, seed: int, profile: str):
        from PIL import Image
        from heocr_unified.fonts import FontInfo
        # Seed-dependent raster prevents the registry from treating variants as the same image.
        value = 32 + (seed % 180)
        self.image = Image.new("RGB", (64, 24), (255, 255, 255))
        for x in range(4, 60):
            self.image.putpixel((x, 10 + (seed % 3)), (value, value, value))
        self.blank = Image.new("RGB", (64, 24), (255, 255, 255))
        self.visibility_fraction = 1.0
        self.metadata = {"profile": profile, "seed": seed}
        self.font = FontInfo(
            path=Path(__file__), family="Fixture Hebrew", style="Regular",
            sha256="f" * 64, cmap=frozenset(range(0x110000)),
            has_gpos=True, is_rashi=False,
        )

    def to_bytes(self):
        import io
        buffer = io.BytesIO()
        self.image.save(buffer, "PNG")
        return buffer.getvalue()


class _FakeRenderer:
    def __init__(self): self.calls=[]
    def render_line(self, text, *, profile, seed, split, rashi=False):
        self.calls.append((text, profile, seed, split, rashi))
        return _FakeRendered(seed, profile)


def _variant_report(tmp_path: Path, *, split: str, rate: float):
    config = load_config(None, overrides={
        "work_dir": str(tmp_path), "upload": False,
        "architecture_extra_variant_rate": rate,
    })
    registry = DedupRegistry(tmp_path / f"registry-{split}-{rate}.sqlite", build_fingerprint="fp")
    writer = _Writer(tmp_path / f"out-{split}-{rate}")
    renderer = _FakeRenderer()
    report = process_architecture_chunk(
        source_key=f"architecture:variant:{split}:{rate}",
        segments=[_single_gold_segment(split)], renderer=renderer,
        registry=registry, writer=writer, output_root=tmp_path,
        config=config, architecture_revision="b" * 40,
    )
    return report, writer.rows, renderer.calls, registry


def test_architecture_extra_variant_rate_one_adds_a_second_train_variant(tmp_path: Path) -> None:
    report, rows, calls, registry = _variant_report(tmp_path, split="train", rate=1.0)
    assert report["segments"] == 1
    assert report["accepted"] == 2
    assert report["primary_variants"] == 1
    assert report["extra_variants"] == 1
    assert len(rows) == 2
    assert len({row["sample_id"] for row in rows}) == 2
    assert len({row["image_sha256"] for row in rows}) == 2
    assert len(calls) == 2
    assert registry.architecture_ledger_summary()["gold_accounted"] == 1


def test_architecture_extra_variants_are_not_added_to_evaluation_splits(tmp_path: Path) -> None:
    for split in ("validation_synthetic", "test_synthetic"):
        report, rows, calls, _registry = _variant_report(tmp_path / split, split=split, rate=1.0)
        assert report["accepted"] == 1
        assert report["extra_variants"] == 0
        assert len(rows) == 1
        assert len(calls) == 1


def test_architecture_extra_variant_rate_zero_keeps_one_train_render(tmp_path: Path) -> None:
    report, rows, calls, _registry = _variant_report(tmp_path, split="train", rate=0.0)
    assert report["accepted"] == 1
    assert report["extra_variants"] == 0
    assert len(rows) == 1
    assert len(calls) == 1
