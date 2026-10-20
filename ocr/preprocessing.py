
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
LANCZOS = Image.Resampling.LANCZOS


class ImageValidationError(ValueError):
    """The file is missing, unreadable, empty, or not a supported image."""


@dataclass
class PreprocessConfig:
    enable_upscale: bool = True
    target_dpi: float = 300.0
    max_target_dpi: float = 400.0
    max_scale: float = 2.0
    min_short_side_px: int = 1200
    assumed_dpi_when_missing: float = 96.0
    enable_denoise: bool = True
    denoise_method: str = "median"
    median_ksize: int = 3
    nlmeans_h: float = 7.0
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    enable_otsu: bool = True
    enable_adaptive: bool = True
    adaptive_block_size: int = 31
    adaptive_c: int = 10
    enable_deskew: bool = True
    deskew_min_degrees: float = 0.6
    deskew_max_degrees: float = 12.0
    deskew_step_degrees: float = 0.5
    enable_sharpen: bool = False
    enable_illumination: bool = False
    # Combined-variant upscale cap. Isolated denoised/otsu/clahe variants are
    # unchanged. Source files are never rewritten.
    max_long_side_px: int = 1280

    def __post_init__(self) -> None:
        if self.denoise_method not in {"median", "nlmeans"}:
            raise ValueError("denoise_method must be 'median' or 'nlmeans'")
        if self.median_ksize % 2 == 0 or self.median_ksize < 3:
            raise ValueError("median_ksize must be an odd integer >= 3")
        if self.max_scale < 1:
            raise ValueError("max_scale must be >= 1")
        if self.max_long_side_px < 32:
            raise ValueError("max_long_side_px must be >= 32")
        if not 150 <= self.target_dpi <= self.max_target_dpi <= 600:
            raise ValueError(
                "target_dpi must sit between 150 and max_target_dpi (<= 600)"
            )


@dataclass
class ImageInfo:
    path: str | None
    width: int
    height: int
    mode: str
    dpi: tuple[float, float] | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Variant:
    name: str
    image: np.ndarray
    steps: list[str]
    width: int
    height: int
    scale: float
    dpi: tuple[float, float] | None
    notes: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        return {
            "name": self.name,
            "steps": list(self.steps),
            "width": self.width,
            "height": self.height,
            "scale": self.scale,
            "dpi": list(self.dpi) if self.dpi else None,
            "notes": list(self.notes),
            "interpolation": "lanczos" if self.scale > 1 else None,
        }


def validate_image(path: str | Path) -> tuple[Image.Image, ImageInfo]:
    """Open an image without writing back to the source path."""
    source = Path(path)

    if not source.is_file():
        raise ImageValidationError(f"Image not found: {source}")

    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImageValidationError(
            f"Unsupported format {source.suffix!r}. "
            f"Expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )

    if source.stat().st_size == 0:
        raise ImageValidationError(f"Empty image file: {source}")

    try:
        with Image.open(source) as opened:
            opened.load()
            image = opened.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            f"Corrupted or unreadable image: {source}"
        ) from exc

    return _normalize_loaded(image, str(source))


def validate_array(
    image: Image.Image,
    label: str = "<memory>",
) -> tuple[Image.Image, ImageInfo]:
    return _normalize_loaded(image, label)


def _normalize_loaded(
    image: Image.Image,
    label: str | None,
) -> tuple[Image.Image, ImageInfo]:
    warnings: list[str] = []

    if getattr(image, "n_frames", 1) > 1:
        warnings.append("multi-frame image; only frame 0 is used")
        image.seek(0)

    if image.width < 8 or image.height < 8:
        raise ImageValidationError(
            f"Image is too small ({image.width}x{image.height}) to be a document"
        )

    dpi = _read_dpi(image)
    mode = image.mode

    if mode == "RGBA":
        image = _composite_on_white(image)
        warnings.append("RGBA flattened onto white; source file unchanged")
    elif mode == "LA":
        image = _composite_on_white(image.convert("RGBA"))
        warnings.append("LA flattened onto white; source file unchanged")
    elif mode == "P":
        image = image.convert("RGBA")
        image = _composite_on_white(image)
        warnings.append("palette image converted to RGB")
    elif mode == "CMYK":
        image = image.convert("RGB")
        warnings.append("CMYK converted to RGB")
    elif mode == "1":
        image = image.convert("L")
    elif mode in {"I", "I;16", "F"}:
        image = image.point(
            lambda value: value * (255.0 / max(image.getextrema()[1], 1))
        )
        image = image.convert("L")
        warnings.append(f"{mode} converted to 8-bit grayscale")
    elif mode in {"L", "RGB"}:
        pass
    else:
        try:
            image = image.convert("RGB")
            warnings.append(f"mode {mode} converted to RGB")
        except (ValueError, OSError) as exc:
            raise ImageValidationError(
                f"Unsupported image mode {mode}"
            ) from exc

    if image.mode == "RGBA":
        alpha = np.asarray(image.getchannel("A"))
        if int(alpha.max()) == 0:
            raise ImageValidationError("Image is fully transparent")

    array = np.asarray(image)

    if array.size == 0 or float(array.std()) < 1e-6:
        raise ImageValidationError("Image is empty (all pixels are identical)")

    info = ImageInfo(
        path=label,
        width=image.width,
        height=image.height,
        mode=image.mode,
        dpi=dpi,
        warnings=warnings,
    )

    return image, info


def _composite_on_white(image: Image.Image) -> Image.Image:
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.getchannel("A"))
    return background


def _read_dpi(image: Image.Image) -> tuple[float, float] | None:
    dpi = image.info.get("dpi")

    if (
        isinstance(dpi, tuple)
        and len(dpi) == 2
        and all(isinstance(value, (int, float)) and value > 1 for value in dpi)
    ):
        return float(dpi[0]), float(dpi[1])

    # TIFF resolution tags, in pixels per inch when the unit is inch.
    ifd = getattr(image, "tag_v2", None)

    if ifd is not None and 282 in ifd and 283 in ifd:
        unit = int(ifd.get(296, 2))
        x_res = float(ifd[282])
        y_res = float(ifd[283])

        if unit == 3:
            x_res *= 2.54
            y_res *= 2.54

        if x_res > 1 and y_res > 1:
            return x_res, y_res

    return None


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def upscale_if_needed(
    image: np.ndarray,
    dpi: tuple[float, float] | None,
    config: PreprocessConfig,
) -> tuple[np.ndarray, float, list[str]]:
    """Resample when resolution is below the target."""
    notes: list[str] = []

    if not config.enable_upscale:
        return image, 1.0, ["upscale disabled"]

    height, width = image.shape[:2]
    short_side = min(height, width)

    if dpi is not None:
        source_dpi = min(dpi)

        if source_dpi >= config.target_dpi:
            return image, 1.0, [
                f"dpi {source_dpi:.0f} already at or above "
                f"{config.target_dpi:.0f}"
            ]

        desired = min(config.target_dpi, config.max_target_dpi) / source_dpi
        notes.append(f"source dpi {source_dpi:.1f}")
    else:
        if short_side >= config.min_short_side_px:
            return image, 1.0, [
                "no dpi metadata; short side already "
                f"{short_side}px so upscaling was skipped"
            ]

        desired = config.target_dpi / config.assumed_dpi_when_missing
        notes.append(
            "no dpi metadata; assumed "
            f"{config.assumed_dpi_when_missing:.0f} DPI "
            "for the scale decision only"
        )

    scale = float(min(config.max_scale, max(desired, 1.0)))

    if scale <= 1.01:
        return image, 1.0, notes + ["scale capped at 1"]

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    longest = max(new_width, new_height)

    if longest > config.max_long_side_px:
        fit = config.max_long_side_px / float(longest)
        scale = scale * fit
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        notes.append(
            f"upscale limited to long side {config.max_long_side_px}px"
        )

    new_size = (new_width, new_height)
    pil = Image.fromarray(image)
    resized = pil.resize(new_size, resample=LANCZOS)

    notes.append(
        f"lanczos upscale x{scale:.2f} to {new_size[0]}x{new_size[1]}"
    )

    return np.asarray(resized), scale, notes


def denoise(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if config.denoise_method == "nlmeans":
        return cv2.fastNlMeansDenoising(
            gray,
            None,
            h=config.nlmeans_h,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    return cv2.medianBlur(gray, config.median_ksize)


def apply_clahe(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    tile = max(2, config.clahe_tile_size)

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(tile, tile),
    )

    return clahe.apply(gray)


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return binary


def adaptive_binarize(
    gray: np.ndarray,
    config: PreprocessConfig,
) -> np.ndarray:
    height, width = gray.shape[:2]
    block = config.adaptive_block_size

    if block % 2 == 0:
        block += 1

    max_block = min(height, width)

    if max_block % 2 == 0:
        max_block -= 1

    block = min(block, max_block)

    if block < 3:
        return otsu_binarize(gray)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        config.adaptive_c,
    )


def correct_illumination(gray: np.ndarray) -> np.ndarray:
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25,
        sigmaY=25,
    )

    background = np.maximum(background, 1)

    return cv2.divide(gray, background, scale=255)


def sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)

    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def deskew_angle(gray: np.ndarray, config: PreprocessConfig) -> float:
    """Return the correction angle in degrees. Zero means leave the image."""
    if not config.enable_deskew:
        return 0.0

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _threshold, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    if int(binary.sum()) < 255 * 40:
        return 0.0

    height, width = binary.shape[:2]
    center = (width / 2.0, height / 2.0)
    best_angle = 0.0
    best_score = -1.0

    angle = -config.deskew_max_degrees

    while angle <= config.deskew_max_degrees + 1e-6:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            binary,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        projection = rotated.sum(axis=1).astype(np.float64)
        score = float(projection.var())

        if score > best_score:
            best_score = score
            best_angle = float(angle)

        angle += config.deskew_step_degrees

    if abs(best_angle) < config.deskew_min_degrees:
        return 0.0

    return best_angle


def rotate_expand(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by angle degrees and expand the canvas."""
    if abs(angle) < 1e-3:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_width = int(np.ceil(height * sin + width * cos))
    new_height = int(np.ceil(height * cos + width * sin))

    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]

    border = (255, 255, 255) if image.ndim == 3 else 255

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def build_variants(
    image: Image.Image,
    info: ImageInfo,
    config: PreprocessConfig | None = None,
    include: list[str] | tuple[str, ...] | None = None,
) -> list[Variant]:
    config = config or PreprocessConfig()
    wanted = set(include) if include is not None else None

    def keep(name: str) -> bool:
        return wanted is None or name in wanted

    # ImageOps.exif_transpose returns a copy; the source file remains unchanged.
    oriented = ImageOps.exif_transpose(image)
    original = np.asarray(oriented).copy()
    gray = to_gray(original)

    variants: list[Variant] = []
    if keep("original"):
        variants.append(
            _variant(
                "original",
                original,
                ["decoded"],
                1.0,
                info.dpi,
                list(info.warnings),
            )
        )
    if keep("grayscale"):
        variants.append(_variant("grayscale", gray, ["grayscale"], 1.0, info.dpi))

    if keep("denoised") and config.enable_denoise:
        variants.append(
            _variant(
                "denoised",
                denoise(gray, config),
                ["grayscale", config.denoise_method],
                1.0,
                info.dpi,
            )
        )

    if keep("contrast") and config.enable_clahe:
        variants.append(
            _variant(
                "contrast",
                apply_clahe(gray, config),
                ["grayscale", "clahe"],
                1.0,
                info.dpi,
            )
        )

    if keep("otsu") and config.enable_otsu:
        variants.append(
            _variant("otsu", otsu_binarize(gray), ["grayscale", "otsu"], 1.0, info.dpi)
        )

    if keep("adaptive") and config.enable_adaptive:
        variants.append(
            _variant(
                "adaptive",
                adaptive_binarize(gray, config),
                ["grayscale", "adaptive-gaussian"],
                1.0,
                info.dpi,
            )
        )

    if keep("combined"):
        variants.append(_combined_variant(original, info, config))

    return variants


def _combined_variant(
    original: np.ndarray,
    info: ImageInfo,
    config: PreprocessConfig,
) -> Variant:
    """Handwriting-preserving stack: modest upscale and deskew only.

    Isolated variants still expose denoise, CLAHE, Otsu, and adaptive
    thresholding. Those steps are not chained here: median blur, CLAHE,
    binarization, and unsharp masking remove thin pen strokes and Arabic
    diacritics that this checkpoint is expected to read from grayscale.
    """
    steps = ["grayscale"]
    notes: list[str] = [
        "combined skips denoise, clahe, sharpen, illumination, and binarize"
    ]

    scaled, scale, scale_notes = upscale_if_needed(original, info.dpi, config)
    notes.extend(scale_notes)

    if scale > 1:
        steps.append("lanczos-upscale")

    gray = to_gray(scaled)
    angle = deskew_angle(gray, config)

    if abs(angle) >= config.deskew_min_degrees:
        gray = rotate_expand(gray, angle)
        steps.append(f"deskew:{angle:.1f}deg")
        notes.append("rotation uses an expanded white canvas")
    else:
        notes.append("deskew not applied")

    achieved_dpi = None

    if info.dpi is not None and scale > 0:
        achieved_dpi = (
            info.dpi[0] * scale,
            info.dpi[1] * scale,
        )

    return _variant(
        "combined",
        gray,
        steps,
        scale,
        achieved_dpi or info.dpi,
        notes,
    )


def _variant(
    name: str,
    image: np.ndarray,
    steps: list[str],
    scale: float,
    dpi: tuple[float, float] | None,
    notes: list[str] | None = None,
) -> Variant:
    array = np.ascontiguousarray(image)
    height, width = array.shape[:2]

    return Variant(
        name=name,
        image=array,
        steps=steps,
        width=width,
        height=height,
        scale=scale,
        dpi=dpi,
        notes=notes or [],
    )


def select_variants(
    variants: list[Variant],
    names: list[str] | tuple[str, ...] | None,
) -> list[Variant]:
    if not names:
        return list(variants)

    by_name = {variant.name: variant for variant in variants}
    missing = [name for name in names if name not in by_name]

    if missing:
        known = ", ".join(by_name)
        raise KeyError(
            f"Unknown variant(s): {', '.join(missing)}. Known: {known}"
        )

    return [by_name[name] for name in names]


def save_variant(variant: Variant, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{variant.name}.png"
    image = Image.fromarray(variant.image)

    save_kwargs = {}

    if variant.dpi is not None:
        save_kwargs["dpi"] = variant.dpi

    image.save(path, format="PNG", **save_kwargs)

    meta_path = directory / f"{variant.name}.json"
    meta_path.write_text(
        json.dumps(variant.to_metadata(), indent=2),
        encoding="utf-8",
    )

    return path


def save_variants(
    variants: list[Variant],
    directory: str | Path,
) -> list[Path]:
    return [save_variant(variant, directory) for variant in variants]


def config_to_dict(config: PreprocessConfig) -> dict:
    return asdict(config)


def crop_region(
    image: Image.Image | np.ndarray,
    box: tuple[int, int, int, int],
    scale: float = 1.0,
) -> Image.Image:
    """
    Crop a region without modifying the original image.

    Args:
        image: A PIL image or NumPy image array.
        box: (left, top, right, bottom), in pixels.
        scale: Optional enlargement factor; must be >= 1.

    Returns:
        Cropped RGB PIL image, optionally enlarged using Lanczos.
    """
    if scale < 1.0:
        raise ValueError("scale must be >= 1.0")

    if isinstance(image, np.ndarray):
        if image.ndim not in (2, 3):
            raise ImageValidationError(
                "Unsupported image array dimensions"
            )
        source = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        source = image.copy()
    else:
        raise TypeError(
            "image must be a PIL image or NumPy array"
        )

    source = ImageOps.exif_transpose(source).convert("RGB")

    left, top, right, bottom = box
    width, height = source.size

    if not (
        0 <= left < right <= width
        and 0 <= top < bottom <= height
    ):
        raise ValueError(
            f"Invalid crop box {box} for image size {width}x{height}"
        )

    cropped = source.crop((left, top, right, bottom))

    if scale > 1.0:
        new_size = (
            round(cropped.width * scale),
            round(cropped.height * scale),
        )
        cropped = cropped.resize(new_size, resample=LANCZOS)

    return cropped
