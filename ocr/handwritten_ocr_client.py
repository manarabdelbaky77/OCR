"""Compatibility exports. The OCR backend is local Pixtral 12B only."""

from ocr.pixtral_ocr_client import (
    INFERENCE_ID,
    MODEL_ID,
    HandwrittenOcrClient,
    HandwrittenOcrConfig,
    LocalModelError,
    LocalOcrClient,
    LocalOcrConfig,
    LocalPixtralOcrClient,
    LocalPixtralOcrConfig,
    OcrClientError,
    PixtralOcrClient,
    PixtralOcrConfig,
    Transcription,
    diagnose_transcription,
)

__all__ = [
    "INFERENCE_ID",
    "MODEL_ID",
    "HandwrittenOcrClient",
    "HandwrittenOcrConfig",
    "LocalModelError",
    "LocalOcrClient",
    "LocalOcrConfig",
    "LocalPixtralOcrClient",
    "LocalPixtralOcrConfig",
    "OcrClientError",
    "PixtralOcrClient",
    "PixtralOcrConfig",
    "Transcription",
    "diagnose_transcription",
]
