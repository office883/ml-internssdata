from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from heocr_unified.writer import TieredParquetRouter


@dataclass(frozen=True)
class _Artifact:
    path: Path
    rows: int
    bytes: int
    sha256: str


class _FakeWriter:
    instances: list["_FakeWriter"] = []

    def __init__(self, output_root, *, config_name, split, source_token, rows_per_shard):
        self.config_name = config_name
        self.split = split
        self.rows = []
        self.artifacts = []
        _FakeWriter.instances.append(self)

    def add(self, row): self.rows.append(row)
    def finish(self):
        self.artifacts = [_Artifact(Path(f"/{self.config_name}.parquet"), len(self.rows), len(self.rows), self.config_name)] if self.rows else []
        return self.artifacts
    def cleanup(self): self.artifacts.clear()


def test_routes_gold_extended_and_quarantine_to_separate_configs(tmp_path) -> None:
    _FakeWriter.instances.clear()
    router = TieredParquetRouter(
        tmp_path,
        base_config_name="modern_print_lines",
        split="train",
        source_token="s",
        rows_per_shard=10,
        writer_cls=_FakeWriter,
    )
    router.add({"data_tier": "gold", "sample_id": "g"})
    router.add({"data_tier": "extended", "sample_id": "e"})
    router.add({"data_tier": "quarantine", "sample_id": "q"})
    artifacts = router.finish()
    assert {writer.config_name for writer in _FakeWriter.instances} == {
        "modern_print_lines",
        "modern_print_lines_extended",
        "modern_print_lines_quarantine",
    }
    assert sum(item.rows for item in artifacts) == 3
