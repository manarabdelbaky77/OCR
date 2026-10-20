"""Compatibility exports. The active backend is local Pixtral, not Qwen.

Import ``ocr.pixtral_ocr_client.LocalPixtralOcrClient`` as the implementation.
These names exist so callers that used LocalOcrClient during the unfinished
Qwen attempt still resolve to Pixtral. Qwen is not loaded from this module.
"""

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
    resolve_model_path,
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
    "resolve_model_path",
]
