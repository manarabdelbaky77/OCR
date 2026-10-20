import csv
import json
from pathlib import Path

from PIL import Image

from ocr.batch import discover_images, run_real_images
from ocr.pixtral_ocr_client import (
    INFERENCE_ID,
    MODEL_ID,
    LocalPixtralOcrClient,
    LocalPixtralOcrConfig,
    Transcription,
)
from ocr.preprocessing import PreprocessConfig


class _ScriptedClient:
    def __init__(self, outcomes: list[Transcription]) -> None:
        self.config = LocalPixtralOcrConfig(model=MODEL_ID)
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def ensure_ready(self) -> None:
        return None

    def transcribe(self, image, preprocessing_method: str) -> Transcription:
        self.calls.append(preprocessing_method)
        if not self._outcomes:
            raise AssertionError("unexpected extra image")
        return self._outcomes.pop(0)


def _page(path: Path) -> None:
    image = Image.new("RGB", (40, 30), (255, 255, 255))
    image.putpixel((4, 4), (0, 0, 0))
    image.save(path)


def test_discovers_existing_data_images() -> None:
    found = discover_images(Path("data/images"))
    names = [path.name for path in found]
    assert len(found) == 9
    assert names == sorted(names)
    assert "azza_mohamed_prescription2.png" in names
    assert "medical-image.jpg" in names


def test_missing_checkpoint_records_not_run_without_fake_text(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _page(image_root / "page.png")
    before = (image_root / "page.png").read_bytes()
    missing = tmp_path / "models" / "pixtral-12b"
    client = LocalPixtralOcrClient(LocalPixtralOcrConfig(model_path=missing))
    report = run_real_images(image_root, tmp_path / "out", client, variant="original")
    assert report.model_ran_locally is False
    assert report.inference == INFERENCE_ID
    assert report.succeeded == 0
    assert report.not_run == 1
    assert str(missing) in (report.blocker or "")
    assert "openrouter" not in (report.blocker or "").lower()
    assert "PIXTRAL_API_KEY" not in (report.blocker or "")
    assert not list((tmp_path / "out").rglob("transcription.txt"))
    summary = list(csv.DictReader((tmp_path / "out" / "summary.csv").open(encoding="utf-8")))
    assert summary[0]["image_path"] == "page.png"
    assert summary[0]["status"] == "not_run"
    assert summary[0]["preprocessing_method"] == "original"
    assert summary[0]["text_file"] == ""
    assert (image_root / "page.png").read_bytes() == before


def test_batch_metadata_is_local(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _page(image_root / "page.png")
    folder = tmp_path / "ckpt"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"x")
    config = LocalPixtralOcrConfig(model=MODEL_ID, model_path=folder)

    class _Boom:
        def __init__(self) -> None:
            self.config = config

        def ensure_ready(self) -> None:
            raise AssertionError("this test disables the probe")

        def transcribe(self, image, preprocessing_method: str) -> Transcription:
            return Transcription(
                text=None,
                model=MODEL_ID,
                status="error",
                preprocessing_method=preprocessing_method,
                latency_ms=1.0,
                error="local failure",
            )

    report = run_real_images(image_root, tmp_path / "out", _Boom(), require_probe=False)
    assert report.inference == INFERENCE_ID
    assert report.model_ran_locally is True
    assert report.model == MODEL_ID
    assert report.base_url.startswith("file:")
    assert "openrouter" not in report.base_url
    assert "api.mistral.ai" not in report.base_url


def test_batch_saves_exact_text_and_continues_after_a_failure(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _page(image_root / "a.png")
    _page(image_root / "b.png")
    (image_root / "notes.txt").write_text("ignore me", encoding="utf-8")
    raw = "Amoxicillin 500 mg\n1 tablet every 8 hours"
    client = _ScriptedClient(
        [
            Transcription(
                text=raw,
                model=MODEL_ID,
                status="success",
                preprocessing_method="original",
                latency_ms=12.5,
            ),
            Transcription(
                text=None,
                model=MODEL_ID,
                status="error",
                preprocessing_method="original",
                latency_ms=4.0,
                error="decoder stopped",
            ),
        ]
    )
    report = run_real_images(
        image_root,
        tmp_path / "out",
        client,
        variant="original",
        preprocess_config=PreprocessConfig(enable_upscale=False, enable_deskew=False),
        require_probe=False,
    )
    assert report.model_ran_locally is True
    assert report.inference == INFERENCE_ID
    assert report.discovered == 2
    assert report.succeeded == 1
    assert report.failed == 1
    assert client.calls == ["original", "original"]
    first, second = report.images
    saved = (tmp_path / "out" / first.text_file).read_text(encoding="utf-8")
    assert saved == raw
    payload = json.loads((tmp_path / "out" / first.result_file).read_text(encoding="utf-8"))
    assert payload["model"] == MODEL_ID
    assert payload["preprocessing_method"] == "original"
    assert payload["latency_ms"] == 12.5
    assert "confidence" not in payload
    assert (tmp_path / "out" / "a.png" / "original.png").is_file()
    assert second.status == "error"
    assert second.error == "decoder stopped"
    assert second.text_file is None
    assert (tmp_path / "out" / second.result_file).is_file()
    rows = list(csv.DictReader((tmp_path / "out" / "summary.csv").open(encoding="utf-8")))
    assert [row["image_path"] for row in rows] == ["a.png", "b.png"]
    assert rows[1]["text_file"] == ""


def test_qwen_backend_module_is_not_on_the_default_path() -> None:
    try:
        import ocr.local_qwen_ocr_client  # noqa: F401
    except ImportError:
        return
    raise AssertionError("Qwen OCR module must not remain on the default path")
