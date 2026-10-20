import inspect
from pathlib import Path

import numpy as np
from PIL import Image

from ocr.cli import main
from ocr.pixtral_ocr_client import (
    INFERENCE_ID,
    MODEL_ID,
    HandwrittenOcrClient,
    LocalModelError,
    LocalPixtralOcrClient,
    LocalPixtralOcrConfig,
    PixtralOcrClient,
    PixtralOcrConfig,
    diagnose_transcription,
    resolve_model_path,
)
from ocr.pipeline import compare_image
from ocr.preprocessing import ImageValidationError, PreprocessConfig
from ocr import server as ocr_server


def _image() -> Image.Image:
    image = Image.new("RGB", (24, 24), "white")
    image.putpixel((2, 2), (0, 0, 0))
    return image


class _Tensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def to(self, device):
        return self

    def __getitem__(self, item):
        return self


class _FakeProcessor:
    image_token = "[IMG]"

    def __call__(self, images=None, text=None, return_tensors="pt"):
        assert images is not None
        assert isinstance(text, str) and self.image_token in text
        return {"input_ids": _Tensor([[1, 2, 3]]), "pixel_values": _Tensor([[0]])}

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["باراسيتامول 500 mg"]


class _FakeModel:
    def eval(self):
        return self

    def generate(self, **kwargs):
        return _Tensor([[1, 2, 3, 4, 5]])


def _stub_loader(_path: Path):
    return _FakeModel(), _FakeProcessor()


def test_missing_checkpoint_does_not_invent_text(tmp_path) -> None:
    missing = tmp_path / "pixtral-12b"
    client = LocalPixtralOcrClient(LocalPixtralOcrConfig(model_path=missing))
    result = client.transcribe(_image(), "original")
    assert result.status == "error"
    assert result.text is None
    assert str(missing) in (result.error or "")
    assert "does not download" in (result.error or "").lower() or "missing" in (result.error or "").lower()
    assert "openrouter" not in (result.error or "").lower()
    assert "data:image" not in (result.error or "")


def test_incomplete_checkpoint_is_rejected(tmp_path) -> None:
    folder = tmp_path / "pixtral-12b"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    client = LocalPixtralOcrClient(LocalPixtralOcrConfig(model_path=folder))
    result = client.transcribe(_image(), "original")
    assert result.status == "error"
    assert result.text is None
    assert "weight" in (result.error or "").lower()


def test_missing_transformers_does_not_invent_text(tmp_path) -> None:
    folder = tmp_path / "pixtral-12b"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"not-a-real-weight")

    def boom(_path: Path):
        raise ImportError("No module named 'transformers'")

    client = LocalPixtralOcrClient(
        LocalPixtralOcrConfig(model_path=folder),
        loader=boom,
    )
    result = client.transcribe(_image(), "original")
    assert result.status == "error"
    assert result.text is None
    assert "transformers" in (result.error or "")


def test_init_does_not_load_weights(tmp_path) -> None:
    called = {"n": 0}

    def loader(_path: Path):
        called["n"] += 1
        return _FakeModel(), _FakeProcessor()

    folder = tmp_path / "pixtral-12b"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"x")
    client = LocalPixtralOcrClient(
        LocalPixtralOcrConfig(model_path=folder),
        loader=loader,
    )
    assert called["n"] == 0
    client.ensure_ready()
    assert called["n"] == 1


def test_mocked_local_inference_returns_processor_text(tmp_path) -> None:
    folder = tmp_path / "pixtral-12b"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"x")
    client = LocalPixtralOcrClient(
        LocalPixtralOcrConfig(model_path=folder),
        loader=_stub_loader,
    )
    result = client.transcribe(np.zeros((16, 16), dtype=np.uint8), "grayscale")
    assert result.status == "success"
    assert result.text == "باراسيتامول 500 mg"
    assert result.model == MODEL_ID


def test_generate_failure_does_not_invent_text(tmp_path) -> None:
    folder = tmp_path / "pixtral-12b"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"x")

    class Broken:
        def eval(self):
            return self

        def generate(self, **kwargs):
            raise RuntimeError("cuda OOM")

    def loader(_path: Path):
        return Broken(), _FakeProcessor()

    result = LocalPixtralOcrClient(
        LocalPixtralOcrConfig(model_path=folder),
        loader=loader,
    ).transcribe(_image(), "original")
    assert result.status == "error"
    assert result.text is None
    assert "OOM" in (result.error or "")


def test_from_env_uses_project_relative_default(monkeypatch) -> None:
    monkeypatch.delenv("LOCAL_PIXTRAL_MODEL_PATH", raising=False)
    monkeypatch.delenv("PIXTRAL_API_KEY", raising=False)
    config = LocalPixtralOcrConfig.from_env()
    expected = resolve_model_path(None)
    assert config.model_path == expected
    assert config.model_path.as_posix().endswith("models/pixtral-12b")
    assert config.inference == INFERENCE_ID
    assert config.base_url.startswith("file:")
    assert "openrouter" not in config.base_url


def test_invalid_image_fails_before_local_pixtral(tmp_path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not-an-image")
    client = LocalPixtralOcrClient(LocalPixtralOcrConfig(model_path=tmp_path / "missing-ckpt"))
    try:
        compare_image(path, client, variants=["original"], output_dir=tmp_path / "out")
    except ImageValidationError:
        return
    raise AssertionError("invalid images must fail validation before Pixtral loads")


def test_pipeline_uses_local_pixtral_client_without_loading(tmp_path) -> None:
    image_path = tmp_path / "page.png"
    _image().save(image_path)

    class _Client:
        def transcribe(self, image, preprocessing_method: str):
            from ocr.pixtral_ocr_client import Transcription

            return Transcription(
                text="نص تجريبي 500 mg",
                model=MODEL_ID,
                status="success",
                preprocessing_method=preprocessing_method,
                latency_ms=3.0,
            )

    result = compare_image(
        image_path,
        _Client(),
        variants=["original"],
        config=PreprocessConfig(enable_upscale=False, enable_deskew=False),
        output_dir=tmp_path / "out",
    )
    assert result.transcriptions[0].text == "نص تجريبي 500 mg"
    saved = (tmp_path / "out" / "transcriptions.json").read_text(encoding="utf-8")
    assert "نص تجريبي 500 mg" in saved


def test_diagnose_flags_empty_and_repeats() -> None:
    assert diagnose_transcription("   ") == ["empty transcription"]
    repeated = "token " * 40
    warnings = diagnose_transcription(repeated.strip())
    assert any("gram" in item for item in warnings)


def test_cli_and_server_route_to_local_pixtral() -> None:
    cli_source = inspect.getsource(main)
    server_source = inspect.getsource(ocr_server)
    client_source = inspect.getsource(LocalPixtralOcrClient)
    for source in (cli_source, server_source, client_source):
        assert "openrouter.ai" not in source.lower()
        assert "PIXTRAL_API_KEY" not in source
        assert "mistralai.client" not in source
        assert "ocr.process" not in source
        assert "Qwen2_5" not in source
        assert "Qwen2_5_VLForConditionalGeneration" not in source
    assert "LocalPixtralOcrClient" in cli_source
    assert "LocalPixtralOcrConfig" in cli_source
    assert "LocalPixtralOcrClient" in server_source
    assert HandwrittenOcrClient is LocalPixtralOcrClient
    assert PixtralOcrClient is LocalPixtralOcrClient
    assert PixtralOcrConfig is LocalPixtralOcrConfig
    assert LocalModelError is not None
    load_source = inspect.getsource(LocalPixtralOcrClient.ensure_ready)
    assert "local_files_only" in inspect.getsource(
        __import__("ocr.pixtral_ocr_client", fromlist=["_load_from_disk"])._load_from_disk
    )
    assert "from_pretrained" in inspect.getsource(
        __import__("ocr.pixtral_ocr_client", fromlist=["_load_from_disk"])._load_from_disk
    )
    assert "ensure_ready" in load_source
