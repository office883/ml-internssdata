from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

from .unicode_utils import normalize_label_strict


@dataclass(frozen=True)
class StructuredExample:
    index: int
    group_id: str
    text: str
    text_sha256: str
    split: str
    mixed_bidi: bool
    with_digits: bool
    template: str


_TERMS = (
    "תכנית קומה", "חתך א–א", "חזית דרומית", "פרט איטום", "תקרת בטון",
    "קיר תמך", "פתח מילוט", "חדר מדרגות", "מערכת ניקוז", "בידוד תרמי",
    "שלד המבנה", "מפלס כניסה", "שטח עיקרי", "שטח שירות", "קו בניין",
    "רצפה צפה", "פיר מעלית", "חניון תת־קרקעי", "מרפסת מקורה", "גג טכני",
)
_CITIES = (
    "תל אביב", "ירושלים", "חיפה", "באר שבע", "חדרה", "גבעתיים",
    "רמת גן", "נתניה", "הרצליה", "אשדוד", "ראשון לציון", "פתח תקווה",
)
_STREETS = (
    "הרצל", "בן־יהודה", "ז׳בוטינסקי", "הנשיא", "העצמאות", "ויצמן",
    "ארלוזורוב", "ביאליק", "הנביאים", "הגליל", "הכרמל", "השלום",
)


def _split_for_group(group_id: str) -> str:
    bucket = int(hashlib.sha256(f"heocr-structured-v9|{group_id}".encode()).hexdigest()[:8], 16) % 1000
    if bucket < 960:
        return "train"
    if bucket < 980:
        return "validation_synthetic"
    return "test_synthetic"


def _text(index: int) -> tuple[str, str]:
    term = _TERMS[index % len(_TERMS)]
    city = _CITIES[(index * 7 + 3) % len(_CITIES)]
    street = _STREETS[(index * 11 + 5) % len(_STREETS)]
    n = index + 1
    templates = (
        ("scale", f"{term} — קנ״מ 1:{25 * (1 + n % 8)} — גיליון A-{n:06d}"),
        ("area", f"{term}: שטח {35 + (n * 17) % 965}.{n % 100:02d} מ״ר — מס׳ מדידה {n:07d}"),
        ("level", f"{term} — מפלס {'+' if n % 3 else '-'}{(n % 87):02d}.{(n * 13) % 100:02d} — REV {chr(65 + n % 26)}-{n:05d}"),
        ("parcel", f"גוש {5000 + n % 4999} חלקה {1 + n % 999} — תכנית {n % 999:03d}/{n % 77:02d}/{2000 + n % 40}"),
        ("permit", f"היתר מס׳ {2000 + n % 40}-{n:06d} — {term} — סטטוס: {'מאושר' if n % 2 else 'לבדיקה'}"),
        ("address", f"{street} {1 + n % 240}, {city} — {term} — דירה {1 + n % 99}, קומה {n % 32}"),
        ("drawing", f"Detail {1 + n % 99} / Sheet A-{100 + n % 900} — {term} — Job IL-{n:07d}"),
        ("coords", f"{term} — X={100000 + (n * 7919) % 899999}.{n % 100:02d} Y={500000 + (n * 3571) % 499999}.{(n * 3) % 100:02d}"),
        ("dimension", f"{term} — מידה {50 + n % 8950}×{40 + (n * 3) % 5960} מ״מ — ID D-{n:07d}"),
        ("revision", f"{term} — Revision {chr(65 + n % 26)}.{n % 10} — נבדק בתאריך {(n % 28) + 1:02d}/{(n % 12) + 1:02d}/{2020 + n % 20}"),
        ("material", f"{term} — בטון C{20 + 5 * (n % 9)}/37 — פלדה B{400 + 100 * (n % 3)} — סימון M-{n:06d}"),
        ("legal", f"בקשה מס׳ {n:08d} לפי תיק בניין {100000 + n % 900000} — {street} {1 + n % 240}, {city}"),
        ("phone", f"איש קשר לפרויקט {term}: 0{2 + n % 7}-{1000000 + n % 8999999} — ref-{n:07d}@example.co.il"),
        ("percentage", f"{term} — התקדמות {n % 101}% — סטייה {(n % 17) - 8:+d}.{n % 10}% — WBS-{n:06d}"),
        ("loads", f"{term} — עומס שימושי {1 + n % 12}.{n % 10} kN/m² — Load Case LC-{n:05d}"),
        ("thermal", f"{term} — U={0.10 + (n % 190) / 100:.2f} W/m²K — R={0.50 + (n % 450) / 100:.2f} m²K/W"),
    )
    return templates[index % len(templates)]


def generate_structured_examples(count: int, *, seed: int = 20260726) -> Iterator[StructuredExample]:
    if count < 0:
        raise ValueError("count must be non-negative")
    offset = int(hashlib.sha256(str(seed).encode()).hexdigest()[:12], 16) % 10_000_000
    for index in range(int(count)):
        absolute = offset + index
        template, raw = _text(absolute)
        label = normalize_label_strict(raw)
        group_id = f"structured-{absolute // 8:09d}"
        yield StructuredExample(
            index=index,
            group_id=group_id,
            text=label.text,
            text_sha256=label.text_sha256,
            split=_split_for_group(group_id),
            mixed_bidi=label.mixed_bidi,
            with_digits=bool(label.digits),
            template=template,
        )
