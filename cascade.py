"""
Cascade OCR: run engines from lightest to heaviest, stopping when
confidence is sufficient.

Engine order (ascending compute weight):
  1. tesseract   — pure CPU, <1s/page, native word confidence
  2. paddleocr   — CPU inference, ~1-2s/page, native box confidence
  3. glm-ocr     — Ollama 0.9B, ~3-5s/page, heuristic confidence
  4. docling     — IBM transformer, ~5-10s/page, heuristic confidence
  5. chandra     — 9B VLM, opt-in, heuristic confidence

Confidence is per-page. A page passes if its score >= threshold.
A document passes if ALL pages pass.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DEFAULT_THRESHOLD = 0.90
LADDER = ["tesseract", "paddleocr", "glm-ocr", "docling", "chandra"]


@dataclass
class PageResult:
    page_num: int
    text: str
    confidence: float
    engine: str
    elapsed: float


@dataclass
class CascadeResult:
    pages: List[PageResult]
    final_engine: str
    all_attempts: List[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n--- PAGE BREAK ---\n\n".join(p.text for p in self.pages)

    @property
    def mean_confidence(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.confidence for p in self.pages) / len(self.pages)


def cascade_ocr(
    pdf_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    apply_preprocess: bool = False,
    handle_redactions: bool = True,
    dpi: int = 300,
    verbose: bool = True,
    progress_callback=None,
    use_gpu: bool = False,
) -> CascadeResult:
    """
    Run engines in ascending compute order, stopping once all pages
    exceed `threshold` confidence.

    If handle_redactions=True, redaction detection runs once up front
    and the marked images are used by all engines.

    progress_callback: optional callable(event_dict) invoked at key moments:
      - engine_start, page_done, engine_done, escalate, complete
    """
    from engines import get_engine
    import preprocess as _pre

    def _emit(evt: dict):
        if progress_callback is not None:
            try:
                progress_callback(evt)
            except Exception:
                pass

    if verbose:
        gpu_str = "GPU" if use_gpu else "CPU"
        print(f"Cascade OCR: threshold={threshold:.0%}, redactions={handle_redactions}, device={gpu_str}")

    # Pre-process images once (PDF→images + optional redaction marking).
    # Per-page detection lets us emit a `redactions_marked` SSE event so
    # the UI can show a per-page count.
    images = _pre.pdf_to_images(pdf_path, dpi=dpi)
    if handle_redactions:
        marked: list = []
        total_marked_pages = 0
        for page_index, orig in enumerate(images):
            boxes = _pre.detect_redactions(orig)
            if boxes:
                marked.append(_pre._paint_redaction_markers(orig, boxes))
                total_marked_pages += 1
                _emit({
                    "event": "redactions_marked",
                    "page": page_index,
                    "count": len(boxes),
                    "boxes": [list(b) for b in boxes],
                })
            else:
                marked.append(orig)
        if verbose and total_marked_pages:
            print(f"  Redaction markers applied on {total_marked_pages} page(s)")
        images = marked

    # Save marked images to a temporary PDF so engines can consume a file path
    import tempfile
    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_pdf_path = tmp_pdf.name
    tmp_pdf.close()
    _images_to_pdf(images, tmp_pdf_path)

    all_attempts = []
    # Start with no result
    best_pages: Optional[List[PageResult]] = None
    best_engine = "none"
    t_total = time.perf_counter()

    for ladder_idx, engine_name in enumerate(LADDER):
        try:
            engine = get_engine(engine_name, use_gpu=use_gpu)
        except ImportError as e:
            if verbose:
                print(f"  [{engine_name}] skipped — {e}")
            continue
        except Exception as e:
            # GPU init crashes (CUDA OOM, driver mismatch, missing CUDA torch
            # build, etc.) shouldn't kill the whole cascade — log and move on.
            if verbose:
                print(f"  [{engine_name}] init failed — {e}")
            all_attempts.append({
                "engine": engine_name,
                "elapsed": 0.0,
                "mean_conf": None,
                "status": f"init error: {e}",
            })
            continue

        if verbose:
            print(f"  [{engine_name}] running...", end=" ", flush=True)

        _emit({"event": "engine_start", "engine": engine_name})

        t0 = time.perf_counter()
        try:
            pages_text, page_confs = engine.extract_with_confidence(
                tmp_pdf_path, apply_preprocess=apply_preprocess
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"ERROR ({e})")
            all_attempts.append({
                "engine": engine_name,
                "elapsed": elapsed,
                "mean_conf": None,
                "status": f"error: {e}",
            })
            _emit({"event": "engine_done", "engine": engine_name,
                   "mean_conf": 0.0, "elapsed": round(elapsed, 2), "passed": False})
            continue

        elapsed = time.perf_counter() - t0
        mean_conf = sum(page_confs) / len(page_confs) if page_confs else 0.0

        if verbose:
            print(f"conf={mean_conf:.1%}, {elapsed:.1f}s")

        # Emit per-page events
        per_page_elapsed = elapsed / max(len(pages_text), 1)
        for i, (text, conf) in enumerate(zip(pages_text, page_confs)):
            _emit({
                "event": "page_done",
                "engine": engine_name,
                "page": i + 1,
                "confidence": round(conf, 4),
            })

        passed = mean_conf >= threshold
        _emit({
            "event": "engine_done",
            "engine": engine_name,
            "mean_conf": round(mean_conf, 4),
            "elapsed": round(elapsed, 2),
            "passed": passed,
        })

        all_attempts.append({
            "engine": engine_name,
            "elapsed": elapsed,
            "mean_conf": mean_conf,
            "status": "ok",
        })

        page_results = [
            PageResult(
                page_num=i,
                text=text,
                confidence=conf,
                engine=engine_name,
                elapsed=per_page_elapsed,
            )
            for i, (text, conf) in enumerate(zip(pages_text, page_confs))
        ]

        best_pages = page_results
        best_engine = engine_name

        if passed:
            if verbose:
                print(f"  Sufficient confidence ({mean_conf:.1%} >= {threshold:.0%}). Stopping.")
            break
        else:
            if verbose:
                print(f"  Below threshold ({mean_conf:.1%} < {threshold:.0%}). Trying next engine.")
            # Emit escalate event if there is a next engine in the ladder
            next_engines = [n for n in LADDER[LADDER.index(engine_name) + 1:]]
            if next_engines:
                _emit({
                    "event": "escalate",
                    "from": engine_name,
                    "to": next_engines[0],
                    "reason": f"conf {mean_conf:.2f} < threshold {threshold:.2f}",
                })

    import os
    try:
        os.unlink(tmp_pdf_path)
    except OSError:
        pass

    if best_pages is None:
        best_pages = []

    elapsed_total = round(time.perf_counter() - t_total, 2)
    final_mean_conf = (
        sum(p.confidence for p in best_pages) / len(best_pages)
        if best_pages else 0.0
    )
    _emit({
        "event": "complete",
        "engine": best_engine,
        "mean_conf": round(final_mean_conf, 4),
        "pages": len(best_pages),
        "elapsed_total": elapsed_total,
    })

    return CascadeResult(
        pages=best_pages,
        final_engine=best_engine,
        all_attempts=all_attempts,
    )


def _images_to_pdf(images: list, out_path: str):
    """Save a list of PIL Images as a multi-page PDF (image-only, for OCR engines)."""
    import fitz
    import io

    doc = fitz.open()
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        img_doc = fitz.open(stream=png_bytes, filetype="png")
        rect = img_doc[0].rect
        page = doc.new_page(width=rect.width, height=rect.height)
        page.insert_image(rect, stream=png_bytes)
        img_doc.close()
    doc.save(out_path)
    doc.close()


def print_cascade_report(result: CascadeResult):
    print()
    print("=" * 60)
    print("CASCADE OCR REPORT")
    print("=" * 60)
    print(f"Final engine used:  {result.final_engine}")
    print(f"Mean confidence:    {result.mean_confidence:.1%}")
    print(f"Pages:              {len(result.pages)}")
    print()
    print("Attempt ladder:")
    for attempt in result.all_attempts:
        conf_str = f"{attempt['mean_conf']:.1%}" if attempt["mean_conf"] is not None else "N/A"
        print(f"  {attempt['engine']:<12} conf={conf_str:<8} {attempt['elapsed']:.1f}s  [{attempt['status']}]")
    print("=" * 60)
    print()
