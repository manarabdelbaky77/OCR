from __future__ import annotations

import html
from pathlib import Path

from ocr.pipeline import CompareResult


def write_report(result: CompareResult, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    by_method = {item.preprocessing_method: item for item in result.transcriptions}
    cards = []
    for meta in result.variant_metadata:
        name = meta["name"]
        transcription = by_method.get(name)
        if transcription is None:
            body = "<p>Not transcribed.</p>"
            status = "not-run"
            latency = ""
        elif transcription.status != "success":
            body = f"<pre class=\"error\">{html.escape(transcription.error or 'error')}</pre>"
            status = "error"
            latency = f"{transcription.latency_ms:.0f} ms"
        else:
            flags = transcription.quality_warnings or []
            warning_html = ""
            if flags:
                items = "".join(
                    f"<li>{html.escape(flag)}</li>" for flag in flags
                )
                warning_html = (
                    "<p class=\"quality\">Quality warnings (heuristic, not "
                    f"verified OCR errors):</p><ul class=\"quality\">{items}</ul>"
                )
            body = (
                f"{warning_html}<pre>{html.escape(transcription.text or '')}</pre>"
            )
            status = "success"
            latency = f"{transcription.latency_ms:.0f} ms"
        steps = ", ".join(meta.get("steps") or [])
        cards.append(
            f"""
            <article>
              <h2>{html.escape(name)}</h2>
              <p class="meta">{html.escape(status)} {html.escape(latency)} · {html.escape(steps)}</p>
              <img src="{html.escape(name)}.png" alt="{html.escape(name)} variant">
              {body}
            </article>
            """
        )
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCR comparison</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; background: #f6f4ef; color: #1c1c1c; }}
    article {{ background: white; padding: 16px; margin: 0 0 16px; border: 1px solid #ddd; }}
    img {{ max-width: 100%; height: auto; background: #eee; }}
    pre {{ white-space: pre-wrap; font-family: Consolas, monospace; }}
    .error {{ color: #8a1f1f; }}
    .meta {{ color: #555; }}
    .quality {{ color: #8a5a12; }}
  </style>
</head>
<body>
  <h1>Handwritten OCR</h1>
  <p>Transcriptions are produced by local Pixtral 12B on this machine. Prepared variant images are not sent to OpenRouter or any other OCR API.</p>
  <p>Source file name is omitted from this report header. Variant images are beside their raw transcriptions.</p>
  {''.join(cards)}
</body>
</html>
"""
    path = directory / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
