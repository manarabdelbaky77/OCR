"""Local Pixtral 12B OCR. No OpenRouter, no cloud API, no other OCR models.

Inference uses Hugging Face Transformers APIs that this environment already
declares for Pixtral:

- ``AutoModelForVision2Seq`` maps ``pixtral`` to ``LlavaForConditionalGeneration``
  in ``transformers.models.auto.modeling_auto``.
- ``AutoProcessor`` maps ``pixtral`` to ``PixtralProcessor``.
- ``PixtralProcessor.__call__`` accepts PIL images and text; it expands the
  processor ``image_token`` (default ``[IMG]``) against ``pixel_values``.
- ``LlavaForConditionalGeneration.generate`` is documented on that class.

Weights are loaded only from ``LOCAL_PIXTRAL_MODEL_PATH`` with
``local_files_only=True``. This module does not download files and does not
call HTTP APIs. Presence of weights on disk is not verified in this edit pass.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ocr.config import load_dotenv
from ocr.preprocessing import Variant

INFERENCE_ID = "local-pixtral"
MODEL_ID = "pixtral-12b"
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
_DEFAULT_CHECKPOINT = _PROJECT_ROOT / "models" / "pixtral-12b"

_WEIGHT_INDEX_CANDIDATES = (
    "model.safetensors.index.json",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "pytorch_model.bin",
)

TRANSCRIPTION_PROMPT = (
    "Transcribe all visible text in this image. Preserve Arabic and English, "
    "mixed-language lines, reading order, numbers, dates, and punctuation. "
    "Do not translate. Do not guess or complete illegible text. "
    "Do not add labels, diagnosis, or commentary. Reply with the transcription only."
)

_LONG_CHAR_LIMIT = 3500
_REPEAT_NGRAM = 6
_REPEAT_NGRAM_COUNT = 4


class LocalModelError(RuntimeError):
    """Raised when the local Pixtral checkpoint cannot be used."""


OcrClientError = LocalModelError


def resolve_model_path(raw: str | Path | None) -> Path:
    """Resolve LOCAL_PIXTRAL_MODEL_PATH against the project root, not cwd."""
    if raw is None or str(raw).strip() == "":
        return _DEFAULT_CHECKPOINT
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


@dataclass
class LocalPixtralOcrConfig:
    model: str = MODEL_ID
    model_path: Path = _DEFAULT_CHECKPOINT
    device: str = "auto"
    dtype: str = "float32"
    max_new_tokens: int = 512
    max_image_side: int = 1280
    inference: str = INFERENCE_ID
    base_url: str = ""

    def __post_init__(self) -> None:
        self.model_path = resolve_model_path(self.model_path)
        if not self.base_url:
            self.base_url = self.model_path.as_uri()
        self.inference = INFERENCE_ID

    @classmethod
    def from_env(cls) -> "LocalPixtralOcrConfig":
        load_dotenv()
        model_path = resolve_model_path(os.environ.get("LOCAL_PIXTRAL_MODEL_PATH"))
        return cls(
            model=os.environ.get("LOCAL_PIXTRAL_MODEL", MODEL_ID).strip() or MODEL_ID,
            model_path=model_path,
            device=os.environ.get("LOCAL_PIXTRAL_DEVICE", "auto").strip() or "auto",
            dtype=os.environ.get("LOCAL_PIXTRAL_DTYPE", "float32").strip() or "float32",
            max_new_tokens=int(os.environ.get("LOCAL_PIXTRAL_MAX_NEW_TOKENS", "512")),
            max_image_side=int(os.environ.get("LOCAL_PIXTRAL_MAX_IMAGE_SIDE", "1280")),
            base_url=model_path.as_uri(),
        )


PixtralOcrConfig = LocalPixtralOcrConfig
LocalOcrConfig = LocalPixtralOcrConfig
HandwrittenOcrConfig = LocalPixtralOcrConfig


@dataclass
class Transcription:
    text: str | None
    model: str
    status: str
    preprocessing_method: str
    latency_ms: float
    error: str | None = None
    served_model: str | None = None
    raw_text: str | None = None
    quality_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "model": self.model,
            "status": self.status,
            "preprocessing_method": self.preprocessing_method,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }
        if self.served_model is not None:
            payload["served_model"] = self.served_model
        if self.raw_text is not None:
            payload["raw_text"] = self.raw_text
        if self.quality_warnings:
            payload["quality_warnings"] = list(self.quality_warnings)
        return payload


class LocalPixtralOcrClient:
    """Loads a local Pixtral Transformers checkpoint lazily. No network inference."""

    def __init__(
        self,
        config: LocalPixtralOcrConfig | None = None,
        loader: Callable[[Path], tuple[Any, Any]] | None = None,
    ) -> None:
        self.config = config or LocalPixtralOcrConfig()
        self._loader = loader
        self._model = None
        self._processor = None
        self._device = "cpu"

    def transcribe(
        self,
        image: Image.Image | np.ndarray | Variant,
        preprocessing_method: str,
    ) -> Transcription:
        started = time.perf_counter()
        try:
            self.ensure_ready()
            text = self._generate(_as_pil(image))
        except Exception as exc:
            return Transcription(
                text=None,
                model=self.config.model,
                status="error",
                preprocessing_method=preprocessing_method,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=_public_error(exc),
            )
        warnings = diagnose_transcription(text)
        return Transcription(
            text=text,
            raw_text=text,
            model=self.config.model,
            status="success",
            preprocessing_method=preprocessing_method,
            latency_ms=(time.perf_counter() - started) * 1000,
            quality_warnings=warnings,
        )

    def ensure_ready(self) -> None:
        if self._model is not None:
            return
        path = Path(self.config.model_path)
        _assert_checkpoint(path)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        self._device = _resolve_device(self.config.device)
        self.config.inference = f"local-pixtral-{self._device}"
        if self._loader is not None:
            self._model, self._processor = self._loader(path)
        else:
            self._model, self._processor = _load_from_disk(
                path, self._device, self.config.dtype
            )
        model = self._model
        if hasattr(model, "eval"):
            model.eval()

    def checkpoint_dir(self) -> Path:
        return Path(self.config.model_path)

    def _generate(self, image: Image.Image) -> str:
        import torch

        model = self._model
        processor = self._processor
        prepared = _limit_image_side(image, self.config.max_image_side)
        prompt = _prompt_with_image_token(processor)
        inputs = processor(
            images=prepared,
            text=prompt,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in dict(inputs).items()
        }
        input_ids = inputs.get("input_ids")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        if input_ids is not None and hasattr(generated, "__getitem__"):
            generated = generated[:, input_ids.shape[1] :]
        decoded = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


PixtralOcrClient = LocalPixtralOcrClient
LocalOcrClient = LocalPixtralOcrClient
HandwrittenOcrClient = LocalPixtralOcrClient


def diagnose_transcription(text: str | None) -> list[str]:
    if text is None or not str(text).strip():
        return ["empty transcription"]

    warnings: list[str] = []
    body = str(text)
    if len(body) > _LONG_CHAR_LIMIT:
        warnings.append(
            f"unusually long transcription ({len(body)} characters)"
        )

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    consecutive = 0
    for current, following in zip(lines, lines[1:]):
        if current == following:
            consecutive += 1
    if consecutive >= 2:
        warnings.append("consecutive duplicate lines")

    tokens = body.split()
    n = _REPEAT_NGRAM
    if len(tokens) >= n * _REPEAT_NGRAM_COUNT:
        counts: dict[tuple[str, ...], int] = {}
        for index in range(len(tokens) - n + 1):
            gram = tuple(tokens[index : index + n])
            counts[gram] = counts.get(gram, 0) + 1
        peak = max(counts.values()) if counts else 0
        if peak >= _REPEAT_NGRAM_COUNT:
            warnings.append(
                f"repeated {_REPEAT_NGRAM}-gram appears {peak} times"
            )

    return warnings


def _assert_checkpoint(path: Path) -> None:
    if not path.is_dir():
        raise LocalModelError(
            f"Local Pixtral checkpoint directory is missing: {path}. "
            "Place a Transformers-format Pixtral 12B checkpoint in that folder. "
            "This client does not download weights and does not call OpenRouter "
            "or any other external OCR service."
        )
    config_file = path / "config.json"
    if not config_file.is_file():
        raise LocalModelError(
            f"Local Pixtral checkpoint is incomplete at {path}. Missing config.json. "
            "This client does not download weights."
        )
    if not any((path / name).is_file() for name in _WEIGHT_INDEX_CANDIDATES):
        listed = ", ".join(_WEIGHT_INDEX_CANDIDATES)
        raise LocalModelError(
            f"Local Pixtral checkpoint at {path} has no weight file. "
            f"Expected one of: {listed}. This client does not download weights."
        )


def _prompt_with_image_token(processor: Any) -> str:
    token = getattr(processor, "image_token", None) or "[IMG]"
    return f"{token}\n{TRANSCRIPTION_PROMPT}"


def _resolve_device(requested: str) -> str:
    choice = (requested or "auto").strip().lower()
    try:
        import torch
    except ImportError as exc:
        raise LocalModelError(
            "PyTorch is not installed. Install the torch build from "
            "requirements-local.txt. This client does not download packages."
        ) from exc
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise LocalModelError(
                "LOCAL_PIXTRAL_DEVICE=cuda but CUDA is not available."
            )
        return "cuda"
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cpu":
        return "cpu"
    raise LocalModelError(
        f"Unsupported LOCAL_PIXTRAL_DEVICE={requested!r}. Use auto, cpu, or cuda."
    )


def _load_from_disk(path: Path, device: str, dtype: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model = AutoModelForVision2Seq.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=dtype or "float32",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    return model, processor


def _limit_image_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side < 32:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


def _as_pil(image: Image.Image | np.ndarray | Variant) -> Image.Image:
    if isinstance(image, Variant):
        image = image.image
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    return Image.fromarray(array).convert("RGB")


def _public_error(exc: BaseException) -> str:
    message = str(exc)
    lowered = message.lower()
    if "data:image" in lowered or "bearer " in lowered:
        return f"{type(exc).__name__}: OCR request failed"
    return f"{type(exc).__name__}: {message}"
