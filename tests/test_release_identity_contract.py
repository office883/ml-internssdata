from __future__ import annotations

import json
import tomllib
from pathlib import Path

from heocr_unified import __version__
from heocr_unified.config import DEFAULT_CONFIG


def test_release_version_and_work_directory_are_consistent() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    user_config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    readme = Path("README_HE.md").read_text(encoding="utf-8")
    assert __version__ == "15.0.0"
    assert project["version"] == __version__
    assert user_config["builder_version"] == __version__
    assert DEFAULT_CONFIG["builder_version"] == __version__
    assert "Builder v15" in readme
    assert "hebrew-ocr-unified-work-v15" in readme
    assert DEFAULT_CONFIG["work_dir"].endswith("hebrew-ocr-unified-work-v15")


def test_locked_runtime_dependencies_match_project_dependencies() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    locked = {
        line.strip() for line in Path("requirements-lock.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert set(project["dependencies"]) <= locked
    assert "pytest==9.0.2" in locked
    assert project["requires-python"] == ">=3.12,<3.13"


def test_default_font_inventory_is_broad_and_unique() -> None:
    paths = DEFAULT_CONFIG["font_repo"]["paths"]
    assert len(paths) == len(set(paths)) >= 15
    for required in (
        "ofl/notosanshebrew", "ofl/notoserifhebrew", "ofl/frankruhllibre",
        "ofl/miriamlibre", "ofl/varelaround", "ofl/secularone",
        "ofl/suezone", "ofl/bellefair", "ofl/amaticsc",
    ):
        assert required in paths
