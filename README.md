# Arabic-English handwritten OCR (local Pixtral 12B)

The default OCR path is **Pixtral 12B running locally** through **Hugging Face Transformers**. It does not call OpenRouter, Mistral OCR, Qwen, PaddleOCR, TrOCR, or any other remote OCR API.

This edit pass did not run commands, installs, tests, downloads, or inference. Local accuracy on handwritten Arabic-English prescriptions is **not guaranteed** and was not measured here.

## Selected backend

| Item | Value |
| --- | --- |
| Model | Pixtral 12B (local weights) |
| Framework | Hugging Face Transformers `AutoModelForVision2Seq` + `AutoProcessor` |
| Why this framework | This project's local Transformers tree already maps `pixtral` to `LlavaForConditionalGeneration` and `PixtralProcessor`. `PixtralProcessor.__call__(images=..., text=...)` accepts PIL images and expands the processor `image_token` (default `[IMG]`). |
| Default checkpoint directory | `D:\OCR\models\pixtral-12b` |
| Config | `LOCAL_PIXTRAL_MODEL_PATH` |
| Network at inference | None. `from_pretrained(..., local_files_only=True)` plus `HF_HUB_OFFLINE=1` |
| Auto-download | Disabled. Missing files raise an error naming the expected directory |

**Unverified in this repository:** the exact Hugging Face repository name and shard filenames for a Pixtral 12B Transformers checkpoint. Transformers' own Pixtral config docstring mentions `pixtral-hf/pixtral-9b`, which is **not** the 12B checkpoint this project targets. Do not treat an OpenRouter id such as `mistralai/pixtral-12b:free` as a local directory.

Place a complete Transformers-format Pixtral 12B snapshot in `D:\OCR\models\pixtral-12b` (or the path you set). The client requires `config.json` and at least one of:

- `model.safetensors.index.json`
- `model.safetensors`
- `pytorch_model.bin.index.json`
- `pytorch_model.bin`

Tokenizer / processor files (`preprocessor_config.json`, tokenizer files, `generation_config.json`, and any `*.safetensors` shards listed by an index) must match whatever that checkpoint actually ships. Exact shard names are **not invented here**.

## Privacy

OCR inference is intended to run only on this machine, provided Pixtral weights and the Transformers/PyTorch stack are installed locally. Prescription images and recognized text are not posted to OpenRouter or another OCR HTTP API. The loader is not permitted to download weights. Transcriptions are not medical advice.

## Configure

1. Copy `.env.example` to `.env`. `.env` is gitignored.
2. Set `LOCAL_PIXTRAL_MODEL_PATH=D:\OCR\models\pixtral-12b` (or another local directory). Relative paths resolve against `D:\OCR`, not the current working directory.
3. Optionally set `LOCAL_PIXTRAL_DEVICE` to `auto`, `cpu`, or `cuda`. `cuda` is used only if the installed `torch` reports CUDA.
4. Obtain Pixtral 12B weights yourself. This project will not fetch them.
5. Install preprocessing deps from `requirements.txt` and the local inference stack from `requirements-local.txt` yourself. Packages were not installed in this edit pass.

```bash
pip install -r requirements.txt
pip install -r requirements-local.txt
```

There is no cloud fallback if the checkpoint, CUDA, or Transformers import fails.

## Hardware

Pixtral 12B is a large multimodal model. Realistic limits (not benchmarked here):

- **Storage:** a 12B Transformers checkpoint is typically tens of gigabytes. Confirm after you place the files.
- **Memory (float32):** weight storage alone is on the order of tens of GB, plus activations. 32 GB system RAM is likely tight or insufficient for comfortable CPU float32 inference.
- **GPU:** supported only through `torch.cuda.is_available()`. This source tree does not assume an NVIDIA GPU is present.
- **Quantization:** not implemented. Transformers/bitsandbytes 4-bit loading is **unverified** for this checkpoint and is not wired in.
- **CPU-only:** expected to be slow and possibly impractical for prescription batches. This was not timed.
- **Accuracy:** local Arabic-English handwriting quality must be evaluated later. Do not assume it matches any hosted Pixtral demo.

## Preprocessing

Supported files: PNG, JPEG, TIFF, BMP, and WebP. Variants: `original`, `grayscale`, `denoised`, `contrast`, `otsu`, `adaptive`, `combined`. Images are converted to RGB PIL for `PixtralProcessor` without forcing every variant to black-and-white. Optional longest-side cap: `LOCAL_PIXTRAL_MAX_IMAGE_SIDE` (default 1280). Source files are not rewritten.

Batch summaries set `inference` to `local-pixtral` (or `local-pixtral-cpu` / `local-pixtral-cuda` after a successful load) and `model_ran_locally` to `true` when the local probe succeeds.

## Run one image

```bash
python -m ocr transcribe data/images/azza_mohamed_prescription2.png --out outputs/real_images/one --variants original
```

## Run every image

```bash
python -m ocr run-images --input data/images --out outputs/real_images --variant original
```

## Local upload page

```bash
python -m ocr serve --host 127.0.0.1 --port 8765 --out outputs/sessions
```

Existing images, datasets, and outputs were not deleted.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Local Pixtral checkpoint directory is missing` | Create `D:\OCR\models\pixtral-12b` and copy a Transformers Pixtral 12B snapshot there. |
| Missing `config.json` or weight files | The folder is incomplete. The client will not download the rest. |
| `No module named 'transformers'` / PyTorch import error | Install `requirements-local.txt` yourself. Version compatibility with your Python is unverified. |
| `LOCAL_PIXTRAL_DEVICE=cuda but CUDA is not available` | Use `cpu` or install a CUDA torch build. CPU may still be too slow or too large. |
| `from_pretrained` architecture mismatch | The files in the directory may not be a Transformers Pixtral checkpoint (for example an OpenRouter id, GGUF-only dump, or a different model). Do not switch this project to Qwen or another OCR engine to work around that. |
| Out of memory | Reduce `LOCAL_PIXTRAL_MAX_IMAGE_SIDE`, close other programs, or use a machine with more RAM/VRAM. Quantization is not enabled in this code. |

## Tests

```bash
python -m pytest
```

Tests mock local model loading and inference. They were not executed during this edit pass.
