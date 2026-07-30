from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from .augment import LINE_PROFILES, PAGE_PROFILES, apply_profile_pair
from .fonts import FontInfo, discover_fonts, split_font_families
from .unicode_utils import NormalizedLabel, grapheme_clusters, normalize_label_strict

PAGE_LAYOUTS = ["single_column", "two_columns", "table", "form"]


class _PageLayoutOverflow(RuntimeError):
    """The current font size cannot fit all page content without loss."""


@dataclass(frozen=True)
class RenderedLine:
    image: Image.Image
    blank: Image.Image
    text: str
    visibility_fraction: float
    metadata: dict[str, Any]
    font: FontInfo

    def to_bytes(self, *, fmt: str = "WEBP", quality: int = 95) -> bytes:
        buffer = io.BytesIO()
        save_kwargs = {"lossless": True, "quality": quality} if fmt.upper() == "WEBP" else {}
        self.image.save(buffer, fmt, **save_kwargs)
        return buffer.getvalue()


@dataclass(frozen=True)
class RenderedPage:
    image: Image.Image
    blank: Image.Image
    annotations: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_bytes(self, *, fmt: str = "WEBP") -> bytes:
        buffer = io.BytesIO()
        self.image.save(buffer, fmt, lossless=True, quality=95)
        return buffer.getvalue()


def require_raqm() -> None:
    if not features.check_feature("raqm"):
        raise RuntimeError("Pillow lacks RAQM; Hebrew shaping cannot be trusted")


def _direction(label: NormalizedLabel) -> str:
    return "rtl" if label.base_direction == "rtl" else "ltr"


def _mask_difference(image: Image.Image, blank: Image.Image) -> np.ndarray:
    a = np.asarray(image.convert("RGB"), dtype=np.int16)
    b = np.asarray(blank.convert("RGB"), dtype=np.int16)
    return np.max(np.abs(a - b), axis=2) >= 12


def _transform_points(points: list[tuple[float, float]], matrix: np.ndarray) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(array, matrix).reshape(-1, 2)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in transformed]


class TextRenderer:
    def __init__(self, fonts: list[FontInfo]):
        require_raqm()
        if not fonts:
            raise RuntimeError("no Hebrew-capable fonts found")
        self.fonts = fonts
        self.pools = split_font_families(fonts)

    @classmethod
    def from_system_fonts(cls, extra_dirs: Iterable[str | Path] = ()) -> "TextRenderer":
        return cls(discover_fonts(extra_dirs))

    def _choose_font(self, label: NormalizedLabel, *, seed: int, split: str, rashi: bool = False) -> FontInfo:
        pool = self.pools.get(split, self.fonts)
        require_marks = label.combining_marks > 0
        eligible = [font for font in pool if font.is_rashi == rashi and font.supports(label.text, require_marks=require_marks)]
        if not eligible and not rashi:
            eligible = [font for font in pool if not font.is_rashi and font.supports(label.text, require_marks=require_marks)]
        if not eligible:
            raise RuntimeError("no single font covers all label code points")
        return eligible[seed % len(eligible)]

    def _base_line(self, label: NormalizedLabel, *, seed: int, split: str, rashi: bool) -> tuple[Image.Image, Image.Image, FontInfo, dict[str, Any]]:
        rng = random.Random(seed)
        font_info = self._choose_font(label, seed=seed, split=split, rashi=rashi)
        supersample = 3
        font_px = rng.randint(38, 68)
        font = ImageFont.truetype(str(font_info.path), font_px * supersample, layout_engine=ImageFont.Layout.RAQM)
        probe = Image.new("RGB", (8, 8), "white")
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), label.text, font=font, direction=_direction(label), language="he")
        margin_x = rng.randint(14, 28) * supersample
        margin_y = rng.randint(10, 20) * supersample
        width = max(16, bbox[2] - bbox[0] + margin_x * 2)
        height = max(16, bbox[3] - bbox[1] + margin_y * 2)
        background = (rng.randint(245, 255), rng.randint(245, 255), rng.randint(240, 255))
        image = Image.new("RGB", (width, height), background)
        blank = Image.new("RGB", (width, height), background)
        ink = rng.randint(0, 45)
        ImageDraw.Draw(image).text(
            (margin_x - bbox[0], margin_y - bbox[1]),
            label.text,
            font=font,
            fill=(ink, ink, ink),
            direction=_direction(label),
            language="he",
        )
        target_size = (max(8, width // supersample), max(8, height // supersample))
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        blank = blank.resize(target_size, Image.Resampling.LANCZOS)
        return image, blank, font_info, {"font_px": font_px, "supersampling": supersample, "direction": _direction(label)}

    def render_line(self, text: str, *, profile: str, seed: int, split: str, rashi: bool = False) -> RenderedLine:
        label = normalize_label_strict(text)
        image, blank, font, base_meta = self._base_line(label, seed=seed, split=split, rashi=rashi)
        before = _mask_difference(image, blank)
        augmented = apply_profile_pair(image, blank, profile=profile, seed=seed ^ 0x617567)
        after = _mask_difference(augmented.image, augmented.blank)
        before_count = int(before.sum())
        visibility = min(1.0, int(after.sum()) / max(before_count, 1))
        metadata = {
            **base_meta,
            "profile": profile,
            "augmentation": augmented.metadata,
            "font_family": font.family,
            "font_style": font.style,
            "font_sha256": font.sha256,
            "visibility_fraction": round(visibility, 6),
        }
        return RenderedLine(augmented.image, augmented.blank, label.text, visibility, metadata, font)

    @staticmethod
    def _text_bbox(draw: ImageDraw.ImageDraw, label: NormalizedLabel, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
        return draw.textbbox(
            (0, 0), label.text, font=font, direction=_direction(label), language="he"
        )

    def _wrap_label(
        self,
        label: NormalizedLabel,
        *,
        draw: ImageDraw.ImageDraw,
        font_info: FontInfo,
        font_px: int,
        max_width: int,
    ) -> list[tuple[NormalizedLabel, ImageFont.FreeTypeFont, tuple[int, int, int, int]]]:
        """Wrap at spaces and, when needed, at grapheme boundaries without loss."""
        font = ImageFont.truetype(
            str(font_info.path), font_px, layout_engine=ImageFont.Layout.RAQM
        )

        def width(value: str) -> int:
            normalized = normalize_label_strict(value)
            bbox = self._text_bbox(draw, normalized, font)
            return bbox[2] - bbox[0]

        words = label.text.split(" ")
        logical_lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if width(candidate) <= max_width:
                current = candidate
                continue
            if current:
                logical_lines.append(current)
                current = ""
            if width(word) <= max_width:
                current = word
                continue
            clusters = grapheme_clusters(word)
            fragment = ""
            for cluster in clusters:
                next_fragment = fragment + cluster
                if fragment and width(next_fragment) > max_width:
                    logical_lines.append(fragment)
                    fragment = cluster
                else:
                    fragment = next_fragment
            if fragment:
                current = fragment
        if current:
            logical_lines.append(current)
        if not logical_lines:
            logical_lines = [label.text]

        output = []
        for value in logical_lines:
            normalized = normalize_label_strict(value)
            bbox = self._text_bbox(draw, normalized, font)
            if bbox[2] - bbox[0] > max_width:
                raise _PageLayoutOverflow("wrapped page fragment still exceeds cell width")
            output.append((normalized, font, bbox))
        compact_in = "".join(label.text.split())
        compact_out = "".join(item[0].text.replace(" ", "") for item in output)
        if compact_in != compact_out:
            raise RuntimeError("page wrapping lost or changed grapheme content")
        return output

    def render_page(self, lines: list[str], *, profile: str, layout: str, seed: int, split: str) -> RenderedPage:
        if layout not in PAGE_LAYOUTS:
            raise ValueError(f"unknown page layout: {layout}")
        labels = [normalize_label_strict(text) for text in lines]
        if not labels:
            raise ValueError("page requires at least one line")

        width, height = 1200, 1900
        background = (252, 250, 245)
        bottom = height - 80
        last_overflow: RuntimeError | None = None

        def render_layout(page_font_px: int):
            image = Image.new("RGB", (width, height), background)
            blank = Image.new("RGB", (width, height), background)
            draw = ImageDraw.Draw(image)
            blank_draw = ImageDraw.Draw(blank)
            annotations: list[dict[str, Any]] = []
            order = 0

            def fragment_step(bbox: tuple[int, int, int, int], *, role: str) -> int:
                text_height = bbox[3] - bbox[1]
                role_padding = 9 if role == "field" else 7
                return max(18, text_height + role_padding, page_font_px + 5)

            def draw_fragment(
                label: NormalizedLabel,
                font: ImageFont.FreeTypeFont,
                bbox: tuple[int, int, int, int],
                font_info: FontInfo,
                *,
                x_right: int,
                y: int,
                role: str,
                row: int,
                column: int,
                source_line_index: int,
                fragment_index: int,
                fragment_count: int,
            ) -> int:
                nonlocal order
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = x_right - text_width - bbox[0]
                x0, y0, x1, y1 = x + bbox[0], y, x + bbox[2], y + text_height
                if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                    raise _PageLayoutOverflow("page text geometry is out of bounds before augmentation")
                draw.text(
                    (x, y - bbox[1]),
                    label.text,
                    font=font,
                    fill=(20, 20, 20),
                    direction=_direction(label),
                    language="he",
                )
                annotations.append({
                    "text": label.text,
                    "reading_order": order,
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "polygon": [
                        [float(x0), float(y0)], [float(x1), float(y0)],
                        [float(x1), float(y1)], [float(x0), float(y1)],
                    ],
                    "baseline": [
                        [float(x0), float(max(y0, y1 - 3))],
                        [float(x1), float(max(y0, y1 - 3))],
                    ],
                    "block_role": role,
                    "row": row,
                    "column": column,
                    "source_line_index": source_line_index,
                    "fragment_index": fragment_index,
                    "fragment_count": fragment_count,
                    "font_family": font_info.family,
                    "font_sha256": font_info.sha256,
                })
                order += 1
                return fragment_step(bbox, role=role)

            def wrapped(
                label: NormalizedLabel,
                *,
                max_width: int,
                role: str,
                sequence: int,
            ):
                font_info = self._choose_font(
                    label, seed=seed + sequence * 13, split=split, rashi=False
                )
                role_font_px = page_font_px + (2 if role == "field" else 0)
                return font_info, self._wrap_label(
                    label,
                    draw=draw,
                    font_info=font_info,
                    font_px=role_font_px,
                    max_width=max_width,
                )

            if layout == "single_column":
                y = 80
                for sequence, label in enumerate(labels):
                    font_info, fragments = wrapped(
                        label, max_width=width - 180, role="body", sequence=sequence
                    )
                    for fragment_index, (fragment, font, bbox) in enumerate(fragments):
                        step = fragment_step(bbox, role="body")
                        if y + step > bottom:
                            raise _PageLayoutOverflow("page content does not fit single-column canvas")
                        y += draw_fragment(
                            fragment, font, bbox, font_info,
                            x_right=width - 90, y=y, role="body", row=-1, column=0,
                            source_line_index=sequence, fragment_index=fragment_index,
                            fragment_count=len(fragments),
                        )
                    y += max(2, page_font_px // 7)

            elif layout == "two_columns":
                gap, margin = 50, 70
                column_width = (width - margin * 2 - gap) // 2
                split_at = (len(labels) + 1) // 2
                sequence = 0
                for column, subset in enumerate((labels[:split_at], labels[split_at:])):
                    x_right = width - margin - column * (column_width + gap)
                    y = 80
                    for label in subset:
                        font_info, fragments = wrapped(
                            label, max_width=column_width, role="body", sequence=sequence
                        )
                        for fragment_index, (fragment, font, bbox) in enumerate(fragments):
                            step = fragment_step(bbox, role="body")
                            if y + step > bottom:
                                raise _PageLayoutOverflow("page content does not fit two-column canvas")
                            y += draw_fragment(
                                fragment, font, bbox, font_info,
                                x_right=x_right, y=y, role="body", row=-1, column=column,
                                source_line_index=sequence, fragment_index=fragment_index,
                                fragment_count=len(fragments),
                            )
                        y += max(2, page_font_px // 7)
                        sequence += 1

            elif layout == "table":
                margin, top = 70, 90
                col_widths = [360, 360, 340]
                x_positions = [width - margin]
                for value in col_widths:
                    x_positions.append(x_positions[-1] - value)
                sequence = 0
                logical_rows = [labels[i:i + 3] for i in range(0, len(labels), 3)]
                y = top
                for row_index, row_labels in enumerate(logical_rows):
                    prepared = []
                    content_heights: list[int] = []
                    for column, label in enumerate(row_labels):
                        info, fragments = wrapped(
                            label,
                            max_width=col_widths[column] - 24,
                            role="body",
                            sequence=sequence,
                        )
                        steps = [fragment_step(bbox, role="body") for _, _, bbox in fragments]
                        prepared.append((column, sequence, info, fragments, steps))
                        content_heights.append(sum(steps))
                        sequence += 1
                    row_height = max(page_font_px + 29, max(content_heights, default=0) + 20)
                    if y + row_height > bottom:
                        raise _PageLayoutOverflow("page content does not fit table canvas")
                    for drawer in (draw, blank_draw):
                        drawer.rectangle(
                            (x_positions[-1], y, x_positions[0], y + row_height),
                            outline=(110, 110, 110), width=1,
                        )
                        for boundary in x_positions[1:-1]:
                            drawer.line(
                                (boundary, y, boundary, y + row_height),
                                fill=(110, 110, 110), width=1,
                            )
                    for column, source_index, info, fragments, _steps in prepared:
                        local_y = y + 10
                        x_right = x_positions[column] - 12
                        for fragment_index, (fragment, font, bbox) in enumerate(fragments):
                            local_y += draw_fragment(
                                fragment, font, bbox, info,
                                x_right=x_right, y=local_y, role="cell",
                                row=row_index, column=column,
                                source_line_index=source_index,
                                fragment_index=fragment_index,
                                fragment_count=len(fragments),
                            )
                    y += row_height

            else:  # form
                y = 80
                for row_index, label in enumerate(labels):
                    info, fragments = wrapped(
                        label, max_width=width - 230, role="field", sequence=row_index
                    )
                    steps = [fragment_step(bbox, role="field") for _, _, bbox in fragments]
                    field_height = max(page_font_px + 33, sum(steps) + 18)
                    if y + field_height > bottom:
                        raise _PageLayoutOverflow("page content does not fit form canvas")
                    rectangle = (80, y, width - 80, y + field_height)
                    for drawer in (draw, blank_draw):
                        drawer.rounded_rectangle(
                            rectangle, radius=8, outline=(150, 150, 150), width=1
                        )
                    local_y = y + 9
                    for fragment_index, (fragment, font, bbox) in enumerate(fragments):
                        local_y += draw_fragment(
                            fragment, font, bbox, info,
                            x_right=width - 105, y=local_y, role="field",
                            row=row_index, column=0,
                            source_line_index=row_index, fragment_index=fragment_index,
                            fragment_count=len(fragments),
                        )
                    y += field_height + max(4, page_font_px // 4)

            if not annotations:
                raise RuntimeError("page renderer produced no text annotations")
            if [item["reading_order"] for item in annotations] != list(range(len(annotations))):
                raise RuntimeError("page reading order is not contiguous")
            return image, blank, annotations

        image = blank = None
        annotations: list[dict[str, Any]] = []
        page_font_px = 0
        for candidate in range(27, 10, -2):
            try:
                image, blank, annotations = render_layout(candidate)
                page_font_px = candidate
                break
            except _PageLayoutOverflow as exc:
                last_overflow = exc
        if image is None or blank is None or not annotations:
            detail = f": {last_overflow}" if last_overflow is not None else ""
            raise RuntimeError(f"page content cannot fit canvas without dropping text{detail}")

        before = _mask_difference(image, blank)

        def apply(current_profile: str):
            augmented = apply_profile_pair(
                image, blank, profile=current_profile, seed=seed ^ 0x70616765
            )
            transformed: list[dict[str, Any]] = []
            valid = True
            out_width, out_height = augmented.image.size
            for row in annotations:
                polygon = _transform_points(
                    [(x, y) for x, y in row["polygon"]], augmented.homography
                )
                baseline = _transform_points(
                    [(x, y) for x, y in row["baseline"]], augmented.homography
                )
                xs = [point[0] for point in polygon]
                ys = [point[1] for point in polygon]
                points = polygon + baseline
                if not all(
                    -1e-3 <= float(x) <= out_width + 1e-3
                    and -1e-3 <= float(y) <= out_height + 1e-3
                    for x, y in points
                ):
                    valid = False
                new = dict(row)
                new["polygon"] = polygon
                new["baseline"] = baseline
                new["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
                transformed.append(new)
            return augmented, transformed, valid

        augmented, transformed, valid = apply(profile)
        effective_profile = profile
        if not valid and profile != "clean_digital":
            augmented, transformed, valid = apply("clean_digital")
            effective_profile = "clean_digital"
        if not valid:
            raise RuntimeError("page annotations leave image bounds after augmentation")
        after = _mask_difference(augmented.image, augmented.blank)
        visibility = min(1.0, int(after.sum()) / max(int(before.sum()), 1))
        metadata = {
            "requested_profile": profile,
            "profile": effective_profile,
            "layout": layout,
            "seed": seed,
            "page_font_px": page_font_px,
            "visibility_fraction": round(visibility, 6),
            "augmentation": augmented.metadata,
        }
        return RenderedPage(augmented.image, augmented.blank, transformed, metadata)
