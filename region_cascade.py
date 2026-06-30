"""
Region-aware cascade OCR.

For each page:
  1. Detect regions with Docling (layout-only).
  2. For each region, run a mini-cascade over the region's configured engine
     ladder (region_config.yaml). Stop a region's mini-cascade once it meets
     --threshold; otherwise escalate to the next engine in its ladder.
  3. Assemble per-region texts in reading order to produce page text.
  4. Compute area-weighted per-page confidence.

Emits per-region progress events identical in shape to cascade.py's so the
web UI can stream them via SSE:
  - region_detected      {page, regions: [{bbox, label, detect_conf}, ...]}
  - region_engine_start  {page, region_index, engine, label, bbox}
  - region_done          {page, region_index, engine, label, bbox, ocr_conf,
                          elapsed, passed}
  - region_escalate      {page, region_index, from, to, reason}
  - page_done            {page, page_conf, regions_total, elapsed}
  - complete             {pages, mean_conf, elapsed_total}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from regions import (
    Region,
    detect_regions,
    reading_order_sort,
    assemble_page,
    page_confidence,
)
from region_config import RegionConfig, load_region_config
from annotations import (
    Annotation,
    ANNOTATION_SOURCE_DOCLING,
    detect_annotations,
    ocr_annotation,
)

DEFAULT_THRESHOLD = 0.90


@dataclass
class RegionPageResult:
    page_num: int
    regions: List[Region]
    text: str
    confidence: float
    elapsed: float
    annotations: List[Annotation] = field(default_factory=list)
    page_size_px: tuple = (0, 0)
    body_y_range: tuple = (0, 0)   # (y0, y1) of the main body region, used for
                                    # anchoring annotations proportionally


@dataclass
class RegionCascadeResult:
    pages: List[RegionPageResult]
    config_path: Optional[str] = None
    all_region_attempts: List[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n--- PAGE BREAK ---\n\n".join(p.text for p in self.pages)

    @property
    def mean_confidence(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.confidence for p in self.pages) / len(self.pages)


def _ocr_region(
    image, region: Region, ladder: List[str], threshold: float, use_gpu: bool,
    engine_cache: dict, apply_preprocess: bool, dpi: int,
    emit: Callable[[dict], None], page_index: int, region_index: int,
    attempts_log: List[dict],
    strike_anchors_page: Optional[List[tuple]] = None,
) -> Region:
    """Run the mini-cascade for one region. Crops `image`, walks `ladder`,
    stops on first engine meeting `threshold`. Mutates and returns `region`.

    `strike_anchors_page` is the list of `(x, y)` annotation lead-line
    endpoints in PAGE coords; we translate to region-crop coords and pass
    to engines that accept a `strike_anchors` kwarg."""
    from engines import get_engine

    crop = image.crop(region.bbox)

    # Translate page-coord anchors into region-crop coords; drop any that
    # fall outside this region.
    crop_anchors: List[tuple] = []
    if strike_anchors_page:
        rx0, ry0, rx1, ry1 = region.bbox
        for ax, ay in strike_anchors_page:
            if rx0 <= ax <= rx1 and ry0 <= ay <= ry1:
                crop_anchors.append((ax - rx0, ay - ry0))

    if not ladder:
        # Explicitly-empty ladder (e.g. empty_value) — skip OCR
        region.engine = None
        region.text = ""
        region.ocr_conf = 0.0
        region.elapsed = 0.0
        return region

    best_text, best_conf, best_engine, best_elapsed = "", 0.0, None, 0.0
    best_lines: list = []
    for engine_name in ladder:
        if engine_name not in engine_cache:
            try:
                engine_cache[engine_name] = get_engine(engine_name, use_gpu=use_gpu)
            except Exception as e:
                attempts_log.append({
                    "page": page_index, "region": region_index, "label": region.label,
                    "engine": engine_name, "status": f"init error: {e}",
                    "ocr_conf": 0.0, "elapsed": 0.0,
                })
                continue

        engine = engine_cache[engine_name]

        emit({
            "event": "region_engine_start",
            "page": page_index,
            "region_index": region_index,
            "engine": engine_name,
            "label": region.label,
            "bbox": list(region.bbox),
        })

        t0 = time.perf_counter()
        try:
            text, conf, line_spans = engine.extract_image_structured(
                crop, apply_preprocess=apply_preprocess, dpi=dpi,
                strike_anchors=crop_anchors or None,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            attempts_log.append({
                "page": page_index, "region": region_index, "label": region.label,
                "engine": engine_name, "status": f"error: {e}",
                "ocr_conf": 0.0, "elapsed": round(elapsed, 2),
            })
            emit({
                "event": "region_done", "page": page_index,
                "region_index": region_index, "engine": engine_name,
                "label": region.label, "bbox": list(region.bbox),
                "ocr_conf": 0.0, "elapsed": round(elapsed, 2), "passed": False,
            })
            continue
        elapsed = time.perf_counter() - t0
        # Translate per-line bboxes from crop-coords to page-coords; carry
        # along any auxiliary fields (struck, struck_fraction, etc.) so
        # downstream consumers (docx writer) can use them without round-trip.
        ox, oy = region.bbox[0], region.bbox[1]
        translated_lines = []
        for ls in (line_spans or []):
            new_ls = dict(ls)
            x0, y0, x1, y1 = ls["bbox"]
            new_ls["bbox"] = (x0 + ox, y0 + oy, x1 + ox, y1 + oy)
            translated_lines.append(new_ls)

        attempts_log.append({
            "page": page_index, "region": region_index, "label": region.label,
            "engine": engine_name, "status": "ok",
            "ocr_conf": round(conf, 4), "elapsed": round(elapsed, 2),
        })

        passed = conf >= threshold
        emit({
            "event": "region_done", "page": page_index,
            "region_index": region_index, "engine": engine_name,
            "label": region.label, "bbox": list(region.bbox),
            "ocr_conf": round(conf, 4), "elapsed": round(elapsed, 2), "passed": passed,
        })

        if conf > best_conf:
            best_text, best_conf, best_engine, best_elapsed = text, conf, engine_name, elapsed
            best_lines = translated_lines

        if passed:
            break
        else:
            # escalate if there's another rung
            idx = ladder.index(engine_name)
            if idx + 1 < len(ladder):
                emit({
                    "event": "region_escalate", "page": page_index,
                    "region_index": region_index, "from": engine_name,
                    "to": ladder[idx + 1],
                    "reason": f"conf {conf:.2f} < threshold {threshold:.2f}",
                })

    region.engine = best_engine
    region.text = best_text
    region.ocr_conf = best_conf
    region.elapsed = best_elapsed
    region.lines = best_lines
    return region


def region_cascade_ocr(
    pdf_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    apply_preprocess: bool = False,
    handle_redactions: bool = True,
    dpi: int = 300,
    verbose: bool = True,
    progress_callback: Optional[Callable[[dict], None]] = None,
    use_gpu: bool = False,
    region_config_path: Optional[str] = None,
) -> RegionCascadeResult:
    """Region-routed cascade. See module docstring for event shape."""
    import preprocess as _pre

    def emit(evt: dict):
        if progress_callback is not None:
            try:
                progress_callback(evt)
            except Exception:
                pass

    cfg: RegionConfig = load_region_config(region_config_path)

    if verbose:
        gpu_str = "GPU" if use_gpu else "CPU"
        print(f"Region-cascade OCR: threshold={threshold:.0%}, "
              f"redactions={handle_redactions}, device={gpu_str}")
        cfg_path = region_config_path or "(defaults)"
        print(f"  Region config: {cfg_path}")

    # 1. Render pages, apply redaction markers (same as cascade.py).
    # Per-page detection runs first so we can emit a `redactions_marked`
    # SSE event with counts; the marker pass uses those boxes directly.
    images = _pre.pdf_to_images(pdf_path, dpi=dpi)
    if handle_redactions:
        marked: list = []
        total_marked_pages = 0
        for page_index, orig in enumerate(images):
            boxes = _pre.detect_redactions(orig)
            if boxes:
                marked.append(_pre._paint_redaction_markers(orig, boxes))
                total_marked_pages += 1
                emit({
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

    # 2. Detect regions (Docling layout-only) on the ORIGINAL PDF.
    # We use the pdf path so Docling sees real page dimensions; redaction
    # boxes mostly fall under `picture` and don't change layout much.
    if verbose:
        print("  [layout] detecting regions...", flush=True)
    t_layout = time.perf_counter()
    regions_by_page = detect_regions(pdf_path, dpi=dpi)
    if verbose:
        total_regions = sum(len(p) for p in regions_by_page)
        print(f"  [layout] {total_regions} regions across "
              f"{len(regions_by_page)} pages ({time.perf_counter() - t_layout:.1f}s)")

    # 3. Per-page mini-cascade per region.
    engine_cache: dict = {}
    attempts_log: List[dict] = []
    page_results: List[RegionPageResult] = []
    t_total = time.perf_counter()

    for page_index, page_regions in enumerate(regions_by_page):
        if page_index >= len(images):
            break
        image = images[page_index]

        # Detect handwritten margin annotations on this page. Docling-sourced
        # annotations get dropped from the body OCR (their text becomes a
        # Word comment, not inline text).
        page_annotations = detect_annotations(image, page_index, page_regions)
        docling_annotation_bboxes = {
            tuple(a.bbox) for a in page_annotations
            if a.source == ANNOTATION_SOURCE_DOCLING
        }
        body_regions = [
            r for r in page_regions if tuple(r.bbox) not in docling_annotation_bboxes
        ]
        # Anchor points = body-side endpoints of margin-annotation lead lines.
        # Engines that detect strikes (merge-tess-paddle) use them as priors
        # to catch short strikes that the line-wide row-density scan misses.
        page_strike_anchors = [
            a.anchor_xy for a in page_annotations if a.anchor_xy is not None
        ]

        emit({
            "event": "region_detected",
            "page": page_index,
            "regions": [
                {"bbox": list(r.bbox), "label": r.label, "detect_conf": r.detect_conf}
                for r in body_regions
            ],
        })

        if verbose:
            print(f"  [page {page_index + 1}] {len(body_regions)} regions"
                  + (f", {len(page_annotations)} annotation(s)" if page_annotations else ""))

        t_page = time.perf_counter()
        for ri, region in enumerate(body_regions):
            ladder = cfg.ladder_for(region.label)
            _ocr_region(
                image=image, region=region, ladder=ladder, threshold=threshold,
                use_gpu=use_gpu, engine_cache=engine_cache,
                apply_preprocess=apply_preprocess, dpi=dpi,
                emit=emit, page_index=page_index, region_index=ri,
                attempts_log=attempts_log,
                strike_anchors_page=page_strike_anchors,
            )
            if verbose:
                eng = region.engine or "skip"
                print(f"    region {ri:>2} {region.label:<20s} -> "
                      f"{eng:<10s} conf={region.ocr_conf:.1%} "
                      f"({region.elapsed:.1f}s)")

        # OCR each annotation with glm-ocr (handwriting is the only reliable
        # engine for these).
        if page_annotations:
            from engines import get_engine
            if "glm-ocr" not in engine_cache:
                engine_cache["glm-ocr"] = get_engine("glm-ocr", use_gpu=use_gpu)
            glm = engine_cache["glm-ocr"]
            for ann_i, ann in enumerate(page_annotations):
                t_ann = time.perf_counter()
                try:
                    ann.text = ocr_annotation(ann, image, glm)
                except Exception as e:
                    ann.text = ""
                    if verbose:
                        print(f"    annotation {ann_i} OCR error: {e}")
                if verbose:
                    elapsed_ann = time.perf_counter() - t_ann
                    print(f"    annotation {ann_i:>2} [{ann.source:7s}] "
                          f"bbox={ann.bbox} ({elapsed_ann:.1f}s) → "
                          f"{ann.text[:60]!r}")

        page_elapsed = time.perf_counter() - t_page
        ordered = reading_order_sort(body_regions)
        page_text = assemble_page(ordered)
        page_conf = page_confidence(body_regions)

        emit({
            "event": "page_done", "page": page_index,
            "page_conf": round(page_conf, 4),
            "regions_total": len(page_regions),
            "elapsed": round(page_elapsed, 2),
        })

        # Determine body y-range for proportional annotation anchoring.
        if body_regions:
            body_y0 = min(r.bbox[1] for r in body_regions)
            body_y1 = max(r.bbox[3] for r in body_regions)
        else:
            body_y0, body_y1 = 0, image.size[1]

        page_results.append(RegionPageResult(
            page_num=page_index,
            regions=ordered,
            text=page_text,
            confidence=page_conf,
            elapsed=page_elapsed,
            annotations=page_annotations,
            page_size_px=image.size,
            body_y_range=(body_y0, body_y1),
        ))

    elapsed_total = round(time.perf_counter() - t_total, 2)
    mean_conf = (
        sum(p.confidence for p in page_results) / len(page_results)
        if page_results else 0.0
    )
    emit({
        "event": "complete",
        "pages": len(page_results),
        "mean_conf": round(mean_conf, 4),
        "elapsed_total": elapsed_total,
    })

    return RegionCascadeResult(
        pages=page_results,
        config_path=region_config_path,
        all_region_attempts=attempts_log,
    )


def print_region_report(result: RegionCascadeResult):
    print()
    print("=" * 70)
    print("REGION-CASCADE OCR REPORT")
    print("=" * 70)
    print(f"Mean confidence:   {result.mean_confidence:.1%}")
    print(f"Pages:             {len(result.pages)}")
    print(f"Region attempts:   {len(result.all_region_attempts)}")
    print(f"Config:            {result.config_path or '(defaults)'}")
    print()
    # Aggregate engine usage
    from collections import Counter
    engine_use = Counter()
    for p in result.pages:
        for r in p.regions:
            if r.engine:
                engine_use[r.engine] += 1
    print("Engine usage across regions:")
    for eng, n in engine_use.most_common():
        print(f"  {eng:<12s} {n:>4d} regions")
    print("=" * 70)
    print()
