#!/usr/bin/env python3
"""
OCR Pipeline — multi-engine and cascade tool.

Usage:
  python ocr_pipeline.py <pdf_path> [options]

Options:
  --engine   glm-ocr | docling | paddleocr | chandra | tesseract | all | cascade | regions
             (default: cascade)
  --preprocess   Apply OpenCV preprocessing (grayscale/denoise/deskew/binarize)
  --dpi      DPI for PDF-to-image conversion (default: 300)
  --output   Output directory (default: results/)
  --compare  Print side-by-side comparison table (auto-enabled with --engine all)
  --threshold      Confidence threshold for cascade / regions mode (default: 0.90)
  --no-redactions  Disable automatic redaction detection/masking
  --region-config  Path to a custom region_config.yaml (regions mode only)

Engines:
  tesseract  Tesseract 5 — fast CPU baseline, native word confidence
  paddleocr  PaddleOCR PP-OCRv5 — multi-column, native box confidence
  glm-ocr    Ollama glm-ocr 0.9B — OCR-specialized, #1 on OmniDocBench V1.5
  docling    IBM Docling — transformer layout, best for tables/structure
  chandra    Chandra 2 (9B VLM) — best accuracy; install: pip install 'chandra-ocr[hf]'
  all        Run all engines and compare
  cascade    (default) Run engines lightest→heaviest, stop when confident enough
  regions    Layout-aware: detect regions with Docling, route each to a
             configurable per-label cascade. See region_config.yaml.
"""

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="OCR a PDF with one or more engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pdf", help="Path to the PDF file")
    p.add_argument(
        "--engine",
        default="cascade",
        help="Engine: tesseract|paddleocr|glm-ocr|docling|chandra|all|cascade",
    )
    p.add_argument("--preprocess", action="store_true", default=False)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output", default="results")
    p.add_argument("--compare", action="store_true", default=False)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Confidence threshold for cascade mode (0.0–1.0, default 0.90)",
    )
    p.add_argument(
        "--no-redactions",
        dest="redactions",
        action="store_false",
        default=True,
        help="Disable automatic redaction detection",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable GPU acceleration (requires CUDA-capable GPU and appropriate packages)",
    )
    p.add_argument(
        "--region-config",
        dest="region_config",
        default=None,
        help="Path to a custom region_config.yaml for regions mode",
    )
    return p.parse_args()


def _patch_dpi(dpi: int):
    import preprocess as _pre
    _orig = _pre.pdf_to_images
    def _patched(path, **kw):
        kw.setdefault("dpi", dpi)
        return _orig(path, **kw)
    _pre.pdf_to_images = _patched


def _apply_redactions_to_pdf(pdf_path: str, dpi: int) -> tuple:
    """
    Detect redactions, mark them as [REDACTED] in all page images,
    save to a temp PDF. Returns (temp_path, n_redactions_found).
    Caller is responsible for deleting temp_path when done.
    """
    import tempfile
    import preprocess as _pre
    from cascade import _images_to_pdf

    images = _pre.pdf_to_images(pdf_path, dpi=dpi)
    marked = [_pre.apply_redaction_markers(img) for img in images]
    # Count pages that changed
    changed = sum(1 for o, m in zip(images, marked) if o is not m)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _images_to_pdf(marked, tmp_path)
    return tmp_path, changed


def run_engine(engine_name, pdf_path, apply_preprocess, use_gpu=False):
    from engines import get_engine

    print(f"  [{engine_name}] loading...", flush=True)
    try:
        engine = get_engine(engine_name, use_gpu=use_gpu)
    except ImportError as e:
        print(f"  [{engine_name}] SKIPPED — {e}")
        return engine_name, None, None, 0.0

    print(f"  [{engine_name}] extracting...", flush=True)
    t0 = time.perf_counter()
    try:
        pages, confs = engine.extract_with_confidence(pdf_path, apply_preprocess)
    except Exception as e:
        print(f"  [{engine_name}] ERROR — {e}")
        return engine_name, None, None, time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    mean_conf = sum(confs) / len(confs) if confs else 0.0

    print(
        f"  [{engine_name}] {len(pages)} page(s), "
        f"conf={mean_conf:.1%}, {elapsed:.1f}s"
    )
    return engine_name, pages, confs, elapsed


def save_results(engine_name, pages, output_dir, pdf_stem):
    out_path = Path(output_dir) / f"{pdf_stem}_{engine_name}.txt"
    text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def print_comparison(results, pdf_path):
    hdr = f"{'Engine':<14} {'Pages':>5} {'Chars':>7} {'Conf':>7} {'Sec':>6} {'Sec/pg':>7}  Sample"
    sep = "=" * (len(hdr) + 40)
    print()
    print(sep)
    print(hdr)
    print(sep)

    for name, pages, confs, elapsed in results:
        if pages is None:
            print(f"{name:<14} {'—':>5} {'—':>7} {'—':>7} {'—':>6} {'—':>7}  FAILED/SKIPPED")
            continue
        chars = sum(len(p) for p in pages)
        sec_per_pg = elapsed / max(len(pages), 1)
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        sample = pages[0][:80].replace("\n", " ").strip() if pages else ""
        print(
            f"{name:<14} {len(pages):>5} {chars:>7} {mean_conf:>7.1%} "
            f"{elapsed:>6.1f} {sec_per_pg:>7.2f}  {sample}"
        )
    print(sep)
    print()


def write_report(results, output_dir, pdf_stem):
    lines = [f"# OCR Comparison Report: {pdf_stem}\n"]
    lines.append("| Engine | Pages | Chars | Confidence | Time (s) | Sec/page | Status |")
    lines.append("|--------|-------|-------|-----------|----------|----------|--------|")

    for name, pages, confs, elapsed in results:
        if pages is None:
            lines.append(f"| {name} | — | — | — | — | — | FAILED/SKIPPED |")
            continue
        chars = sum(len(p) for p in pages)
        sec_per_pg = elapsed / max(len(pages), 1)
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        lines.append(
            f"| {name} | {len(pages)} | {chars} | {mean_conf:.1%} "
            f"| {elapsed:.1f} | {sec_per_pg:.2f} | OK |"
        )

    lines.append("")
    lines.append("## Sample Output — Page 1\n")
    for name, pages, confs, elapsed in results:
        if pages is None:
            continue
        sample = pages[0][:500] if pages else ""
        lines.append(f"### {name}\n```\n{sample}\n```\n")

    report_path = Path(output_dir) / f"{pdf_stem}_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_regions_mode(
    pdf_path, threshold, apply_preprocess, handle_redactions, dpi,
    output_dir, pdf_stem, use_gpu=False, region_config_path=None,
):
    from region_cascade import region_cascade_ocr, print_region_report

    result = region_cascade_ocr(
        pdf_path,
        threshold=threshold,
        apply_preprocess=apply_preprocess,
        handle_redactions=handle_redactions,
        dpi=dpi,
        verbose=True,
        use_gpu=use_gpu,
        region_config_path=region_config_path,
    )

    print_region_report(result)

    out_path = Path(output_dir) / f"{pdf_stem}_regions.txt"
    out_path.write_text(result.text, encoding="utf-8")
    print(f"Output → {out_path}")

    # Per-page + per-region breakdown
    detail_path = Path(output_dir) / f"{pdf_stem}_regions_detail.txt"
    lines = []
    for p in result.pages:
        lines.append(f"=== Page {p.page_num + 1} — page_conf={p.confidence:.1%} ===")
        for ri, r in enumerate(p.regions):
            lines.append(
                f"  region {ri:>2} {r.label:<20s} engine={r.engine or 'skip':<10s} "
                f"conf={r.ocr_conf:.1%} bbox={r.bbox}"
            )
        lines.append("")
    lines.append(f"Mean confidence: {result.mean_confidence:.1%}")
    detail_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Region detail → {detail_path}")

    # Word-format output with page breaks, per-page heading + per-region table
    from docx_writer import write_docx
    docx_path = Path(output_dir) / f"{pdf_stem}_regions.docx"

    # Convert page-level Annotation objects into the simple dicts the docx
    # writer wants. For Phase 2: prefer the lead-line target y over the
    # annotation's own y when available, and compute proportional position
    # against the actual body text extent (union of OCR line bboxes) rather
    # than the body region's bounding box — the region often covers the
    # whole page even though body text only occupies a fraction of it.
    annotations_per_page = []
    for p in result.pages:
        page_h = p.page_size_px[1] if p.page_size_px else 1
        # Find body text extent from the merge engine's per-line bboxes.
        body_lines = [ln for r in p.regions for ln in (r.lines or [])]
        if body_lines:
            body_y0 = min(ln["bbox"][1] for ln in body_lines)
            body_y1 = max(ln["bbox"][3] for ln in body_lines)
        else:
            body_y0, body_y1 = (p.body_y_range or (0, page_h))
        body_span = max(1, body_y1 - body_y0)

        page_anns = []
        for a in p.annotations:
            # Prefer the lead-line target y; fall back to annotation y_center.
            if a.anchor_xy is not None:
                target_y = a.anchor_xy[1]
            else:
                target_y = a.y_center
            y_rel = (target_y - body_y0) / body_span
            page_anns.append({
                "text": a.text,
                "anchor_y_relative": y_rel,
                "author": a.author,
                "initials": a.initials,
            })
        annotations_per_page.append(page_anns)

    # Collect per-page line spans from body regions in reading order so the
    # docx writer can apply per-line strike formatting.
    line_spans_per_page = []
    for p in result.pages:
        spans = []
        for r in p.regions:
            spans.extend(r.lines or [])
        line_spans_per_page.append(spans)

    write_docx(
        str(docx_path),
        [p.text for p in result.pages],
        annotations=annotations_per_page,
        line_spans_per_page=line_spans_per_page,
    )
    print(f"Word → {docx_path}")


def run_cascade_mode(pdf_path, threshold, apply_preprocess, handle_redactions, dpi, output_dir, pdf_stem, use_gpu=False):
    from cascade import cascade_ocr, print_cascade_report

    result = cascade_ocr(
        pdf_path,
        threshold=threshold,
        apply_preprocess=apply_preprocess,
        handle_redactions=handle_redactions,
        dpi=dpi,
        verbose=True,
        use_gpu=use_gpu,
    )

    print_cascade_report(result)

    out_path = Path(output_dir) / f"{pdf_stem}_cascade.txt"
    out_path.write_text(result.text, encoding="utf-8")
    print(f"Output → {out_path}")

    conf_path = Path(output_dir) / f"{pdf_stem}_cascade_confidence.txt"
    conf_lines = [
        f"Page {p.page_num + 1}: engine={p.engine}, confidence={p.confidence:.1%}"
        for p in result.pages
    ]
    conf_lines.append(f"\nOverall: engine={result.final_engine}, mean_conf={result.mean_confidence:.1%}")
    conf_path.write_text("\n".join(conf_lines), encoding="utf-8")
    print(f"Confidence report → {conf_path}")


def main():
    args = parse_args()

    # Hide CUDA from torch in CPU mode. Docling pulls in the default torch
    # wheel (cu130), which auto-uses CUDA when libcuda.so is visible. On older
    # cards (sm_61 and below) this crashes with cudaErrorNoKernelImageForDevice.
    # Must be set before any engine imports torch.
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pdf_path = str(Path(args.pdf).resolve())
    if not os.path.exists(pdf_path):
        print(f"ERROR: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = Path(pdf_path).stem

    _patch_dpi(args.dpi)

    engine_arg = args.engine.lower()

    print(f"\nOCR Pipeline")
    print(f"  PDF:        {pdf_path}")
    print(f"  Engine:     {engine_arg}")
    print(f"  DPI:        {args.dpi}")
    print(f"  Preproc:    {args.preprocess}")
    print(f"  Redactions: {args.redactions}")
    print(f"  GPU:        {args.gpu}")
    if engine_arg == "cascade":
        print(f"  Threshold:  {args.threshold:.0%}")
    print(f"  Output:     {output_dir}/")
    print()

    if engine_arg == "cascade":
        run_cascade_mode(
            pdf_path, args.threshold, args.preprocess, args.redactions, args.dpi, output_dir, pdf_stem,
            use_gpu=args.gpu,
        )
        return

    if engine_arg == "regions":
        run_regions_mode(
            pdf_path, args.threshold, args.preprocess, args.redactions, args.dpi,
            output_dir, pdf_stem, use_gpu=args.gpu,
            region_config_path=args.region_config,
        )
        return

    if engine_arg == "all":
        from engines import all_engines
        engine_names = [name for name, _ in all_engines()]
        args.compare = True
    else:
        engine_names = [e.strip() for e in engine_arg.split(",")]

    # Apply redaction markers once up front for all direct-mode engine runs
    work_pdf = pdf_path
    tmp_pdf_to_clean = None
    if args.redactions:
        print("  Applying redaction detection...", flush=True)
        work_pdf, n_changed = _apply_redactions_to_pdf(pdf_path, args.dpi)
        tmp_pdf_to_clean = work_pdf
        if n_changed:
            print(f"  Redactions marked on {n_changed} page(s).")
        else:
            print("  No redactions detected.")

    all_results = []
    for name in engine_names:
        engine_name, pages, confs, elapsed = run_engine(name, work_pdf, args.preprocess, use_gpu=args.gpu)
        all_results.append((engine_name, pages, confs, elapsed))
        if pages is not None:
            out = save_results(engine_name, pages, output_dir, pdf_stem)
            print(f"  [{engine_name}] saved → {out}")

    if tmp_pdf_to_clean:
        import os as _os
        try:
            _os.unlink(tmp_pdf_to_clean)
        except OSError:
            pass

    if args.compare or len(engine_names) > 1:
        print_comparison(all_results, pdf_path)
        report = write_report(all_results, output_dir, pdf_stem)
        print(f"Comparison report → {report}")

    print("Done.")


if __name__ == "__main__":
    main()
