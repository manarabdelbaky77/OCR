from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocr.batch import run_real_images
from ocr.config import VARIANT_NAMES
from ocr.evaluation import load_dataset, score_predictions
from ocr.pixtral_ocr_client import LocalPixtralOcrClient, LocalPixtralOcrConfig
from ocr.pipeline import compare_image, prepare
from ocr.preprocessing import PreprocessConfig
from ocr.report import write_report
from ocr.sample_dataset import write_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocr",
        description="Arabic-English handwritten OCR with a local Pixtral 12B checkpoint",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess", help="Write preprocessing variants for one image")
    preprocess.add_argument("image", type=Path)
    preprocess.add_argument("--out", type=Path, required=True)
    preprocess.add_argument("--variants", default=",".join(VARIANT_NAMES))
    _add_preprocess_flags(preprocess)

    transcribe = sub.add_parser("transcribe", help="Transcribe one image with local Pixtral")
    transcribe.add_argument("image", type=Path)
    transcribe.add_argument("--out", type=Path, required=True)
    transcribe.add_argument("--variants", default="original")
    _add_preprocess_flags(transcribe)

    evaluate = sub.add_parser("evaluate", help="Score a predictions JSON file against the dataset")
    evaluate.add_argument("--dataset", type=Path, default=Path("data/eval"))
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)

    build = sub.add_parser("build-dataset", help="Render the synthetic evaluation images")
    build.add_argument("--out", type=Path, default=Path("data/eval"))

    real = sub.add_parser("run-images", help="Transcribe images already stored under data/images")
    real.add_argument("--input", type=Path, default=Path("data/images"))
    real.add_argument("--out", type=Path, default=Path("outputs/real_images"))
    real.add_argument("--variant", default="original", choices=list(VARIANT_NAMES))
    _add_preprocess_flags(real)

    serve = sub.add_parser("serve", help="Local page for upload, variants, and local Pixtral OCR")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--out", type=Path, default=Path("outputs/sessions"))

    args = parser.parse_args(argv)
    if args.command == "build-dataset":
        path = write_dataset(args.out)
        print(path)
        return 0
    if args.command == "preprocess":
        variants, _metadata = prepare(
            args.image,
            config=_config_from_args(args),
            variants=_names(args.variants),
            output_dir=args.out,
        )
        print(f"Wrote {len(variants)} variants to {args.out}")
        return 0
    if args.command == "transcribe":
        config = LocalPixtralOcrConfig.from_env()
        result = compare_image(
            args.image,
            LocalPixtralOcrClient(config),
            variants=_names(args.variants),
            config=_config_from_args(args),
            output_dir=args.out,
        )
        for item in result.transcriptions:
            if item.status == "success" and item.text is not None:
                (args.out / f"{item.preprocessing_method}.txt").write_text(item.text, encoding="utf-8")
        report = write_report(result, args.out)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        print(f"Report: {report}")
        return 0 if all(item.status == "success" for item in result.transcriptions) else 1
    if args.command == "evaluate":
        samples = load_dataset(args.dataset)
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        report = score_predictions(samples, predictions)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"variants": [item.to_dict() for item in report.variants]}, indent=2))
        return 0
    if args.command == "run-images":
        config = LocalPixtralOcrConfig.from_env()
        report = run_real_images(
            args.input,
            args.out,
            LocalPixtralOcrClient(config),
            variant=args.variant,
            preprocess_config=_config_from_args(args),
        )
        print(
            json.dumps(
                {
                    "discovered": report.discovered,
                    "succeeded": report.succeeded,
                    "failed": report.failed,
                    "not_run": report.not_run,
                    "model_ran_locally": report.model_ran_locally,
                    "inference": report.inference,
                    "elapsed_ms": report.elapsed_ms,
                    "summary_csv": report.summary_csv,
                    "blocker": report.blocker,
                },
                indent=2,
            )
        )
        if report.blocker:
            return 2
        return 0 if report.failed == 0 else 1
    if args.command == "serve":
        from ocr.server import serve

        serve(args.host, args.port, args.out)
        return 0
    parser.error(f"Unknown command {args.command}")
    return 2


def _names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in VARIANT_NAMES]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}. Choose from {', '.join(VARIANT_NAMES)}")
    return names


def _add_preprocess_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-upscale", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--denoise", choices=["median", "nlmeans"], default="median")
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--illumination", action="store_true")


def _config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        enable_upscale=not args.no_upscale,
        enable_denoise=not args.no_denoise,
        enable_clahe=not args.no_clahe,
        enable_deskew=not args.no_deskew,
        denoise_method=args.denoise,
        enable_sharpen=args.sharpen,
        enable_illumination=args.illumination,
    )


if __name__ == "__main__":
    sys.exit(main())
