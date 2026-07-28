from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter

LINE_PROFILES = [
    "clean_digital", "office_scan", "phone_photo", "perspective_photo", "photocopy",
    "fax", "low_resolution", "motion_blur", "colored_paper", "uneven_lighting",
    "broken_ink", "bleed_through", "dark_ui", "blueprint", "yellowed_archive",
    "grayscale_scan", "fold_shadow", "screen_capture", "thermal_receipt",
    "low_contrast", "extreme",
]
PAGE_PROFILES = [
    "clean_digital", "office_scan", "photocopy", "phone_photo", "yellowed_archive",
    "grayscale_scan", "fold_shadow", "screen_capture", "low_contrast", "blueprint",
    "uneven_lighting", "bleed_through",
]


@dataclass(frozen=True)
class PairAugmentation:
    image: Image.Image
    blank: Image.Image
    metadata: dict[str, Any]
    homography: np.ndarray


def _array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _warp_pair(image: np.ndarray, blank: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    flags = cv2.INTER_CUBIC
    border = cv2.BORDER_REPLICATE
    return (
        cv2.warpPerspective(image, matrix, size, flags=flags, borderMode=border),
        cv2.warpPerspective(blank, matrix, size, flags=flags, borderMode=border),
    )


def _motion_kernel(size: int, angle: float) -> np.ndarray:
    size = max(3, size | 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = size // 2
    x0 = int(center - math.cos(angle) * radius)
    y0 = int(center - math.sin(angle) * radius)
    x1 = int(center + math.cos(angle) * radius)
    y1 = int(center + math.sin(angle) * radius)
    cv2.line(kernel, (x0, y0), (x1, y1), 1.0, 1)
    kernel /= max(kernel.sum(), 1.0)
    return kernel


def apply_profile_pair(image: Image.Image, blank: Image.Image, *, profile: str, seed: int) -> PairAugmentation:
    if profile not in LINE_PROFILES and profile not in PAGE_PROFILES:
        raise ValueError(f"unknown augmentation profile: {profile}")
    rng = random.Random(seed)
    a = _array(image).astype(np.float32)
    b = _array(blank).astype(np.float32)
    height, width = a.shape[:2]
    homography = np.eye(3, dtype=np.float32)
    metadata: dict[str, Any] = {"profile": profile, "seed": seed}

    if profile in {"phone_photo", "perspective_photo", "extreme"}:
        strength = rng.uniform(0.006, 0.026 if profile != "extreme" else 0.045)
        dx, dy = max(1, int(width * strength)), max(1, int(height * strength))
        src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        dst = np.float32([
            [rng.randint(0, dx), rng.randint(0, dy)],
            [width - 1 - rng.randint(0, dx), rng.randint(0, dy)],
            [width - 1 - rng.randint(0, dx), height - 1 - rng.randint(0, dy)],
            [rng.randint(0, dx), height - 1 - rng.randint(0, dy)],
        ])
        homography = cv2.getPerspectiveTransform(src, dst)
        a, b = _warp_pair(a, b, homography, (width, height))
        metadata["perspective_strength"] = round(strength, 6)

    angle_limit = {
        "office_scan": 0.7, "phone_photo": 2.0, "perspective_photo": 2.7,
        "photocopy": 0.8, "fax": 0.4, "low_resolution": 1.0,
        "motion_blur": 1.3, "extreme": 3.5,
    }.get(profile, 0.35)
    angle = rng.uniform(-angle_limit, angle_limit)
    if abs(angle) > 0.04:
        center = (width / 2.0, height / 2.0)
        affine = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation = np.vstack([affine, [0, 0, 1]]).astype(np.float32)
        homography = rotation @ homography
        a, b = _warp_pair(a, b, rotation, (width, height))
        metadata["rotation_deg"] = round(angle, 5)

    if profile in {"office_scan", "photocopy", "fax", "low_resolution", "thermal_receipt", "extreme"}:
        scale = rng.uniform(0.48 if profile != "extreme" else 0.35, 0.86)
        interpolation = cv2.INTER_NEAREST if profile in {"fax", "thermal_receipt"} else cv2.INTER_AREA
        small_size = (max(8, int(width * scale)), max(8, int(height * scale)))
        a = cv2.resize(a, small_size, interpolation=interpolation)
        b = cv2.resize(b, small_size, interpolation=interpolation)
        a = cv2.resize(a, (width, height), interpolation=cv2.INTER_NEAREST if profile in {"fax", "thermal_receipt"} else cv2.INTER_CUBIC)
        b = cv2.resize(b, (width, height), interpolation=cv2.INTER_NEAREST if profile in {"fax", "thermal_receipt"} else cv2.INTER_CUBIC)
        metadata["resample_scale"] = round(scale, 5)

    if profile in {"motion_blur", "phone_photo", "extreme"}:
        size = rng.choice([3, 5, 7] if profile != "extreme" else [7, 9, 11])
        kernel = _motion_kernel(size, rng.uniform(0, math.pi))
        a = cv2.filter2D(a, -1, kernel)
        b = cv2.filter2D(b, -1, kernel)
        metadata["motion_kernel"] = size
    elif profile in {"office_scan", "photocopy", "low_resolution", "uneven_lighting", "bleed_through", "extreme"}:
        sigma = rng.uniform(0.15, 1.25 if profile != "extreme" else 2.0)
        a = cv2.GaussianBlur(a, (0, 0), sigma)
        b = cv2.GaussianBlur(b, (0, 0), sigma)
        metadata["blur_sigma"] = round(sigma, 5)

    # Shared content-independent paper/illumination field.
    yy, xx = np.mgrid[0:height, 0:width]
    angle2 = rng.uniform(0, 2 * math.pi)
    gradient = (np.cos(angle2) * (xx / max(width - 1, 1) - 0.5) + np.sin(angle2) * (yy / max(height - 1, 1) - 0.5))
    field = np.zeros((height, width, 1), dtype=np.float32)
    if profile in {"uneven_lighting", "phone_photo", "perspective_photo", "fold_shadow", "extreme"}:
        amount = rng.uniform(12, 38 if profile != "extreme" else 55)
        field += gradient[..., None] * amount
        metadata["illumination_amount"] = round(amount, 4)
    if profile in {"colored_paper", "photocopy", "office_scan", "yellowed_archive", "bleed_through", "extreme"}:
        noise = np.random.default_rng(seed ^ 0xA551).normal(0, rng.uniform(1.5, 7.5), (height, width)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 2.0)[..., None]
        field += noise
    a += field
    b += field

    if profile == "fold_shadow":
        center = rng.randint(max(1, width // 4), max(2, 3 * width // 4))
        shadow = np.exp(-((xx - center) ** 2) / (2 * max(4, width * 0.025) ** 2)) * rng.uniform(-55, -25)
        a += shadow[..., None]
        b += shadow[..., None]
        metadata["fold_x"] = center

    if profile == "broken_ink":
        for target_name, target in (("image", a), ("blank", b)):
            gray = cv2.cvtColor(np.clip(target, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            inverse = 255 - gray
            kernel = np.ones((2, 2), np.uint8)
            inverse = cv2.erode(inverse, kernel, iterations=1)
            converted = cv2.cvtColor(255 - inverse, cv2.COLOR_GRAY2RGB).astype(np.float32)
            if target_name == "image":
                a = converted
            else:
                b = converted

    if profile == "bleed_through":
        ghost_seed = np.random.default_rng(seed ^ 0xB1EED)
        ghost = ghost_seed.normal(0, 1, (height, width)).astype(np.float32)
        ghost = cv2.GaussianBlur(ghost, (0, 0), 8)
        ghost = (ghost - ghost.min()) / max(float(ghost.max() - ghost.min()), 1e-6)
        ghost = (ghost * rng.uniform(-20, -8))[..., None]
        a += ghost
        b += ghost

    contrast = rng.uniform(0.82, 1.16)
    offset = rng.uniform(-8, 8)
    if profile == "low_contrast":
        contrast, offset = rng.uniform(0.48, 0.68), rng.uniform(55, 85)
    a = a * contrast + offset
    b = b * contrast + offset
    metadata.update({"contrast": round(contrast, 4), "offset": round(offset, 4)})

    if profile in {"grayscale_scan", "fax", "photocopy", "thermal_receipt"}:
        a_gray = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        b_gray = cv2.cvtColor(np.clip(b, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if profile in {"fax", "thermal_receipt"}:
            threshold = rng.randint(150, 205)
            _, a_gray = cv2.threshold(a_gray, threshold, 255, cv2.THRESH_BINARY)
            _, b_gray = cv2.threshold(b_gray, threshold, 255, cv2.THRESH_BINARY)
            metadata["threshold"] = threshold
        a = cv2.cvtColor(a_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
        b = cv2.cvtColor(b_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    elif profile == "dark_ui":
        a = np.clip(255 - a, 0, 255)
        b = np.clip(255 - b, 0, 255)
        a *= np.array([0.82, 0.92, 1.0], dtype=np.float32)
        b *= np.array([0.82, 0.92, 1.0], dtype=np.float32)
    elif profile == "blueprint":
        def blueprint(target: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(np.clip(target, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            ink = 255 - gray
            result = np.empty((*gray.shape, 3), dtype=np.float32)
            result[..., 0] = 12 + ink * 0.12
            result[..., 1] = 50 + ink * 0.36
            result[..., 2] = 105 + ink * 0.58
            return result
        a, b = blueprint(a), blueprint(b)
    elif profile == "yellowed_archive":
        tint = np.array([1.0, 0.96, 0.82], dtype=np.float32)
        a *= tint
        b *= tint
    elif profile == "screen_capture":
        stripe = ((xx % 3) == 0).astype(np.float32)[..., None] * -4.0
        a += stripe
        b += stripe

    # JPEG round-trip with identical quality; codec noise differs only where pixels differ.
    if profile in {"office_scan", "phone_photo", "photocopy", "low_resolution", "screen_capture", "extreme"}:
        quality = rng.randint(48 if profile != "extreme" else 32, 88)
        def jpeg(target: np.ndarray) -> np.ndarray:
            ok, encoded = cv2.imencode(".jpg", np.clip(target, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("JPEG encode failed")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32)
        a, b = jpeg(a), jpeg(b)
        metadata["jpeg_quality"] = quality

    return PairAugmentation(_image(a), _image(b), metadata, homography)
