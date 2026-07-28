from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import regex

_BIDI_CONTROLS = {
    "\u061c", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c",
    "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069",
}
_PRESENTATION_START = 0xFB1D
_PRESENTATION_END = 0xFB4F
_GRAPHEME_RE = regex.compile(r"\X")
_SPACE_RE = regex.compile(r"[\p{Zs}\t\v\f]+")
_MARKUP_RE = re.compile(r"<[^>]{1,200}>|\[\[|\]\]|&(?:[A-Za-z]+|#\d+);", re.I)
_REPEATED_RE = regex.compile(r"(.)\1{7,}")


class LabelRejected(ValueError):
    """Raised when a source string cannot safely become an OCR label."""


@dataclass(frozen=True)
class NormalizedLabel:
    text: str
    original_sha256: str
    text_sha256: str
    base_direction: str
    codepoints: int
    graphemes: int
    words: int
    hebrew_chars: int
    latin_chars: int
    digits: int
    combining_marks: int
    mixed_bidi: bool


def grapheme_clusters(text: str) -> list[str]:
    return _GRAPHEME_RE.findall(text)


def _decompose_presentation_forms(text: str) -> str:
    output: list[str] = []
    for char in text:
        cp = ord(char)
        output.append(unicodedata.normalize("NFKC", char) if _PRESENTATION_START <= cp <= _PRESENTATION_END else char)
    return "".join(output)


def _strong_direction(text: str) -> str:
    if any("HEBREW" in unicodedata.name(ch, "") for ch in text):
        return "rtl"
    for char in text:
        bidi = unicodedata.bidirectional(char)
        if bidi in {"R", "AL"}:
            return "rtl"
        if bidi == "L":
            return "ltr"
    return "rtl"


def normalize_label_strict(value: str, *, collapse_spaces: bool = True) -> NormalizedLabel:
    if not isinstance(value, str):
        raise LabelRejected("label is not a string")
    original_sha = hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()
    if value.startswith("\ufeff"):
        value = value.removeprefix("\ufeff")
    for char in value:
        if char == "\ufffd":
            raise LabelRejected("replacement character is forbidden")
        if char in _BIDI_CONTROLS:
            raise LabelRejected("bidi control is forbidden")
        if unicodedata.category(char) == "Cf":
            raise LabelRejected("format/invisible character is forbidden")
        if ord(char) < 32 and char not in "\n\t\r":
            raise LabelRejected("control character is forbidden")
    text = _decompose_presentation_forms(value)
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    if collapse_spaces:
        text = "\n".join(_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
        text = "\n".join(line for line in text.split("\n") if line)
    text = text.strip()
    if not text:
        raise LabelRejected("empty label")
    if unicodedata.normalize("NFC", text) != text:
        raise LabelRejected("label is not NFC")
    if any(_PRESENTATION_START <= ord(ch) <= _PRESENTATION_END for ch in text):
        raise LabelRejected("presentation form survived normalization")
    clusters = grapheme_clusters(text)
    for cluster in clusters:
        if not cluster:
            continue
        if unicodedata.combining(cluster[0]):
            raise LabelRejected("orphan combining mark")
    hebrew = sum(1 for ch in text if "HEBREW" in unicodedata.name(ch, ""))
    latin = sum(1 for ch in text if "LATIN" in unicodedata.name(ch, ""))
    digits = sum(1 for ch in text if ch.isdigit())
    marks = sum(1 for ch in text if unicodedata.combining(ch))
    return NormalizedLabel(
        text=text,
        original_sha256=original_sha,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        base_direction=_strong_direction(text),
        codepoints=len(text),
        graphemes=len(clusters),
        words=len(text.split()),
        hebrew_chars=hebrew,
        latin_chars=latin,
        digits=digits,
        combining_marks=marks,
        mixed_bidi=bool(hebrew and (latin or digits)),
    )


def classify_usable_label(text: str, *, maximum_graphemes: int = 112) -> tuple[bool, str, NormalizedLabel | None]:
    try:
        label = normalize_label_strict(text)
    except LabelRejected as exc:
        return False, str(exc).replace(" ", "_"), None
    if label.graphemes < 2:
        return False, "too_short", label
    if label.graphemes > maximum_graphemes:
        return False, "too_long", label
    if label.hebrew_chars == 0:
        return False, "no_hebrew", label
    if _MARKUP_RE.search(label.text):
        return False, "markup_like", label
    if _REPEATED_RE.search(label.text):
        return False, "repeated_character_run", label
    return True, "ok", label


def split_grapheme_safe(text: str, *, max_graphemes: int = 112, min_graphemes: int = 8) -> list[str]:
    label = normalize_label_strict(text)
    if label.graphemes <= max_graphemes:
        return [label.text]
    output: list[str] = []
    remaining = label.text
    while len(grapheme_clusters(remaining)) > max_graphemes:
        prefix = "".join(grapheme_clusters(remaining)[:max_graphemes])
        cut = -1
        for pattern in (r"[.!?׃:]\s+", r"[,;—–-]\s+", r"\s+"):
            matches = [m for m in re.finditer(pattern, prefix) if len(grapheme_clusters(prefix[:m.end()])) >= min_graphemes]
            if matches:
                cut = matches[-1].end()
                break
        if cut < 0:
            cut = len(prefix)
        part = prefix[:cut].strip()
        if part:
            output.append(normalize_label_strict(part).text)
        remaining = remaining[cut:].strip()
        if not remaining:
            break
    if remaining:
        output.append(normalize_label_strict(remaining).text)
    return output


def namespace_key(repo_id: str, kind: str, value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return f"{repo_id}:{kind}:{text}" if text else ""
