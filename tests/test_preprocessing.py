import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr.preprocessing import (
    ImageValidationError,
    PreprocessConfig,
    build_variants,
    save_variants,
    validate_image,
)


def _save(path: Path, image: Image.Image, dpi=None) -> None:
    kwargs = {"dpi": dpi} if dpi else {}
    image.save(path, **kwargs)


def _document(path: Path, size=(120, 80), dpi=(72, 72)) -> None:
    image = Image.new("RGB", size, (255, 255, 255))
    pixels = image.load()
    y = min(size[1] // 2, size[1] - 2)
    x_end = min(size[0] - 2, max(12, size[0] - 10))
    for x in range(8, x_end):
        pixels[x, y] = (0, 0, 0)
        pixels[x, y + 1] = (0, 0, 0)
    _save(path, image, dpi=dpi)


def test_rejects_missing_empty_corrupt_and_unsupported(tmp_path: Path) -> None:
    missing = tmp_path / "no.png"
    with pytest.raises(ImageValidationError):
        validate_image(missing)

    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ImageValidationError):
        validate_image(empty)

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not a png")
    with pytest.raises(ImageValidationError):
        validate_image(bad)

    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")
    with pytest.raises(ImageValidationError):
        validate_image(text)


def test_blank_image_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "blank.png"
    Image.new("RGB", (40, 40), (240, 240, 240)).save(path)
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image(path)


def test_rgba_and_grayscale_are_accepted_and_source_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    image = Image.new("RGBA", (80, 60), (255, 255, 255, 0))
    draw_pixels = image.load()
    for x in range(5, 70):
        draw_pixels[x, 30] = (10, 10, 10, 255)
    image.save(path)
    before = path.read_bytes()

    loaded, info = validate_image(path)
    assert loaded.mode == "RGB"
    assert any("RGBA" in warning for warning in info.warnings)
    variants = build_variants(loaded, info, PreprocessConfig(enable_deskew=False, enable_upscale=False))
    by_name = {variant.name: variant for variant in variants}
    assert by_name["original"].image.ndim == 3
    assert by_name["grayscale"].image.ndim == 2
    assert path.read_bytes() == before


def test_low_dpi_is_resampled_with_lanczos_and_scale_is_capped(tmp_path: Path) -> None:
    path = tmp_path / "low.png"
    _document(path, size=(100, 40), dpi=(72, 72))
    before = path.read_bytes()
    loaded, info = validate_image(path)
    assert info.dpi[0] == pytest.approx(72, abs=0.1)
    assert info.dpi[1] == pytest.approx(72, abs=0.1)
    config = PreprocessConfig(enable_deskew=False, max_scale=2.0, target_dpi=300)
    variants = build_variants(loaded, info, config)
    combined = next(variant for variant in variants if variant.name == "combined")
    original = next(variant for variant in variants if variant.name == "original")
    assert original.width == 100
    assert combined.scale == pytest.approx(2.0)
    assert combined.width == 200
    assert "lanczos-upscale" in combined.steps
    assert path.read_bytes() == before


def test_sufficient_dpi_is_not_upscaled(tmp_path: Path) -> None:
    path = tmp_path / "print.png"
    _document(path, size=(100, 40), dpi=(300, 300))
    loaded, info = validate_image(path)
    variants = build_variants(loaded, info, PreprocessConfig(enable_deskew=False))
    combined = next(variant for variant in variants if variant.name == "combined")
    assert combined.scale == 1.0
    assert combined.width == 100


def test_binarization_and_denoise_variants(tmp_path: Path) -> None:
    path = tmp_path / "noise.png"
    rng = np.random.default_rng(0)
    array = np.full((80, 80, 3), 255, dtype=np.uint8)
    array[20:60, 10:70] = 0
    salt = rng.random((80, 80)) > 0.9
    array[salt] = 0
    image = Image.fromarray(array)
    image.save(path)
    loaded, info = validate_image(path)
    config = PreprocessConfig(enable_upscale=False, enable_deskew=False)
    variants = build_variants(loaded, info, config)
    saved = save_variants(variants, tmp_path / "out")
    names = {path.stem for path in saved}
    assert {"original", "grayscale", "denoised", "contrast", "otsu", "adaptive", "combined"} <= names
    by_name = {variant.name: variant for variant in variants}
    assert set(np.unique(by_name["otsu"].image)).issubset({0, 255})
    assert set(np.unique(by_name["adaptive"].image)).issubset({0, 255})
    assert by_name["grayscale"].image.ndim == 2
    assert not np.array_equal(by_name["denoised"].image, by_name["grayscale"].image)


def test_deskew_records_a_correction_for_a_rotated_page(tmp_path: Path) -> None:
    from PIL import ImageDraw

    image = Image.new("L", (240, 140), 255)
    draw = ImageDraw.Draw(image)
    for y in (40, 70, 100):
        draw.line((16, y, 220, y), fill=0, width=2)
    rotated = image.rotate(6, expand=True, fillcolor=255)
    path = tmp_path / "rotated.png"
    rotated.save(path, dpi=(300, 300))
    loaded, info = validate_image(path)
    variants = build_variants(loaded, info, PreprocessConfig(enable_upscale=False))
    combined = next(variant for variant in variants if variant.name == "combined")
    assert any(step.startswith("deskew:") for step in combined.steps)
    assert combined.width > loaded.width
    assert combined.height > loaded.height


def test_unknown_variant_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    _document(path)
    loaded, info = validate_image(path)
    variants = build_variants(loaded, info, PreprocessConfig(enable_deskew=False, enable_upscale=False))
    from ocr.preprocessing import select_variants

    with pytest.raises(KeyError):
        select_variants(variants, ["original", "missing"])
