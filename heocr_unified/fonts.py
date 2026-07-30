from __future__ import annotations

import hashlib
import os
import re
import subprocess
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fontTools.ttLib import TTFont, TTCollection




# Minimum character inventory that each disjoint synthetic split must be able
# to render with at least one pinned non-Rashi family.  It deliberately covers
# modern Hebrew, digits, Hebrew punctuation, niqqud, meteg, and qamats qatan.
POINTED_COVERAGE_CODEPOINTS = frozenset(
    {ord(ch) for ch in "אבגדהוזחטיךכלםמןנסעףפץצקרשת0123456789-.,()׳״־׃"}
    | set(range(0x05B0, 0x05BE))
    | {0x05BF, 0x05C1, 0x05C2, 0x05C7}
)

@dataclass(frozen=True)
class FontInfo:
    path: Path
    family: str
    style: str
    sha256: str
    cmap: frozenset[int]
    has_gpos: bool
    is_rashi: bool

    def supports(self, text: str, *, require_marks: bool = False) -> bool:
        required = {ord(char) for char in text if not char.isspace()}
        if not required.issubset(self.cmap):
            return False
        return not require_marks or self.has_gpos




_FALLBACK_FONT_IDENTIFIERS = frozenset({
    "lastresort",
    "lastresortregular",
    "applesymbols",
    "applecoloremoji",
})


def _font_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _is_fallback_font(info: FontInfo) -> bool:
    """Reject cmap-wide fallback/symbol fonts that do not contain real Hebrew ink."""
    identifiers = {
        _font_identifier(info.family),
        _font_identifier(info.path.stem),
        _font_identifier(info.path.name),
    }
    return bool(identifiers & _FALLBACK_FONT_IDENTIFIERS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _name(font: TTFont, name_id: int, fallback: str) -> str:
    table = font.get("name")
    if table is None:
        return fallback
    candidates: list[str] = []
    for record in table.names:
        if record.nameID != name_id:
            continue
        try:
            candidates.append(record.toUnicode().strip())
        except Exception:
            continue
    return next((value for value in candidates if value), fallback)


def _font_family_name(font: TTFont, fallback: str) -> str:
    # OpenType Name ID 16 is the typographic family and remains stable for
    # variable-font instances. ID 1 may contain values such as "Rubik Light".
    typographic = _name(font, 16, "")
    return typographic or _name(font, 1, fallback)


def _font_style_name(font: TTFont, fallback: str) -> str:
    return _name(font, 17, "") or _name(font, 2, fallback)


def _load_one(path: Path, font_number: int = 0) -> FontInfo | None:
    try:
        font = TTFont(path, fontNumber=font_number, lazy=True)
        cmap: set[int] = set()
        for table in font["cmap"].tables:
            cmap.update(table.cmap.keys())
        family = _font_family_name(font, path.stem)
        style = _font_style_name(font, "Regular")
        has_gpos = "GPOS" in font
        font.close()
        return FontInfo(
            path=path,
            family=family,
            style=style,
            sha256=_sha256(path),
            cmap=frozenset(cmap),
            has_gpos=has_gpos,
            is_rashi="rashi" in family.lower() or "rashi" in path.name.lower(),
        )
    except Exception:
        return None


@lru_cache(maxsize=16)
def _discover_fonts_cached(extra_dirs: tuple[str, ...], include_system: bool) -> tuple[FontInfo, ...]:
    system_directories = [
        Path.home() / "Library/Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path.home() / ".fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
    ]
    directories = (system_directories if include_system else []) + [Path(value) for value in extra_dirs]
    seen: set[Path] = set()
    output: list[FontInfo] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            info = _load_one(resolved)
            if info is None or _is_fallback_font(info):
                continue
            probe = "אבגדהוזחטיךכלםמןנסעףפץצקרשת0123456789-.,()"
            if info.supports(probe):
                output.append(info)
    return tuple(sorted(output, key=lambda row: (row.family.casefold(), row.style.casefold(), str(row.path))))


def discover_fonts(
    extra_dirs: Iterable[str | Path] = (), *, include_system: bool = True
) -> list[FontInfo]:
    normalized = tuple(str(Path(value).expanduser().resolve()) for value in extra_dirs)
    return list(_discover_fonts_cached(normalized, bool(include_system)))


def acquire_google_fonts(destination: str | Path, *, repo_url: str, revision: str, sparse_paths: list[str]) -> Path:
    destination = Path(destination).expanduser().resolve()
    repo = destination / "google-fonts"
    destination.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init"], check=True)
        subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", repo_url], check=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "init", "--cone"], check=True)
    subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", *sparse_paths], check=True)
    subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"font revision mismatch: {actual} != {revision}")
    return repo


def pointed_coverage_families(fonts: list[FontInfo]) -> set[str]:
    """Return non-Rashi families that cover the full pointed Hebrew probe."""
    return {
        font.family for font in fonts
        if not font.is_rashi
        and font.has_gpos
        and POINTED_COVERAGE_CODEPOINTS.issubset(font.cmap)
    }


def split_font_families(fonts: list[FontInfo]) -> dict[str, list[FontInfo]]:
    family_rows: dict[str, list[FontInfo]] = {}
    for font in fonts:
        if not font.is_rashi:
            family_rows.setdefault(font.family, []).append(font)
    families = sorted(
        family_rows,
        key=lambda family: hashlib.sha256(f"heocr-font-split-v12|{family}".encode()).hexdigest(),
    )
    if len(families) < 3:
        return {"train": fonts, "validation_synthetic": fonts, "test_synthetic": fonts}

    # Hold out about 20% per synthetic evaluation split, capped at three
    # families so the training pool remains broad even in small font sets.
    holdout = max(1, min(3, len(families) // 5))
    if len(families) - 2 * holdout < 1:
        holdout = max(1, (len(families) - 1) // 2)

    coverage_families = pointed_coverage_families(fonts)
    full_coverage = [family for family in families if family in coverage_families]

    validation: set[str] = set()
    test: set[str] = set()
    reserved_train: str | None = None
    if len(full_coverage) >= 3:
        # Reserve a complete Hebrew/niqqud family for every disjoint split.
        # Without this, a hash-only split can assign a family lacking meteg or
        # sof pasuq to an evaluation pool and make valid pointed labels
        # impossible to render.
        validation.add(full_coverage[0])
        test.add(full_coverage[1])
        reserved_train = full_coverage[2]

    available = [
        family for family in families
        if family not in validation and family not in test and family != reserved_train
    ]
    while len(validation) < holdout and available:
        validation.add(available.pop(0))
    while len(test) < holdout and available:
        test.add(available.pop(0))
    train = set(families) - validation - test

    pools = {
        "train": [font for font in fonts if font.family in train or font.is_rashi],
        "validation_synthetic": [font for font in fonts if font.family in validation or font.is_rashi],
        "test_synthetic": [font for font in fonts if font.family in test or font.is_rashi],
    }
    return pools
