"""
Layout-aware region detection for the regions-mode pipeline.

Wraps Docling's DocumentConverter in layout-only mode (do_ocr=False,
do_table_structure=False, CPU accelerator) and returns per-page lists of
Region objects whose bbox coordinates are scaled to PIXEL space at the
caller's DPI — so a downstream crop is `image.crop(region.bbox)` with no
further conversion.

Also handles overlap dedup: when two clusters cover the same area (IoU
> 0.5), the higher-confidence one wins.

Engines are NOT loaded here — this module only does layout detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

BAND_HEIGHT_PX = 50  # for reading-order sort: rows within this band are L-to-R


@dataclass
class Region:
    page_index: int
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel coords at render DPI
    label: str                        # lowercase DocItemLabel value
    detect_conf: float
    engine: Optional[str] = None      # set after routing
    text: str = ""
    ocr_conf: float = 0.0
    elapsed: float = 0.0
    # Per-line layout in PAGE pixel coords (translated from crop coords).
    # Populated only when the OCR engine supports structured output.
    lines: list = field(default_factory=list)

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedup_overlapping(regions: List[Region], iou_threshold: float = 0.5) -> List[Region]:
    """Greedy NMS: walk highest-conf-first; drop any later region with IoU > threshold."""
    sorted_regions = sorted(regions, key=lambda r: r.detect_conf, reverse=True)
    kept: List[Region] = []
    for r in sorted_regions:
        overlap = any(_iou(r.bbox, k.bbox) > iou_threshold for k in kept)
        if not overlap:
            kept.append(r)
    return kept


def _build_converter():
    """Build a Docling DocumentConverter configured for layout-only on CPU."""
    # Force CPU before any torch CUDA init — shared venv has a CUDA-enabled torch
    # whose kernels don't run on older NVIDIA cards (e.g. Quadro P2000, sm_61).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        AcceleratorOptions,
        AcceleratorDevice,
    )
    from docling.datamodel.base_models import InputFormat

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    opts.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU, num_threads=4
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


_CONVERTER = None


def _get_converter():
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = _build_converter()
    return _CONVERTER


def detect_regions(pdf_path: str, dpi: int = 300, fill_gaps: bool = True) -> List[List[Region]]:
    """
    Run Docling layout-only on `pdf_path` and return a list of per-page region
    lists. Bbox coords are scaled to pixel space at the given render DPI so
    callers can crop the corresponding `pdf_to_images(pdf_path, dpi)` image
    directly without further conversion.

    If `fill_gaps` is True, each page also gets synthetic `_uncovered` regions
    covering any area Docling did not assign to a cluster — these route through
    the _default ladder so text in undetected areas isn't silently dropped.
    """
    converter = _get_converter()
    try:
        result = converter.convert(pdf_path)
    except Exception as first_err:
        # Some scanned-form PDFs (e.g. court PACER exports) trip docling-parse's
        # PDFium parser with "Data format error" even though PyMuPDF reads
        # them fine. Round-trip the PDF through PyMuPDF to produce a
        # structurally clean copy, then retry once. The normalized tmp file
        # is cleaned up after the retry succeeds.
        from preprocess import normalize_pdf
        normalized = normalize_pdf(pdf_path)
        if normalized == pdf_path:
            raise
        try:
            result = converter.convert(normalized)
        except Exception:
            raise first_err
        finally:
            import os
            try:
                os.unlink(normalized)
            except OSError:
                pass

    pages_out: List[List[Region]] = []
    for page_index, page in enumerate(result.pages):
        page_w_pt = page.size.width if page.size else 612.0
        page_h_pt = page.size.height if page.size else 792.0
        sx = dpi / 72.0
        sy = dpi / 72.0
        page_w_px = int(round(page_w_pt * sx))
        page_h_px = int(round(page_h_pt * sy))

        regions: List[Region] = []
        if page.predictions and page.predictions.layout:
            for c in page.predictions.layout.clusters:
                bbox = c.bbox
                # Docling bbox may use TOPLEFT or BOTTOMLEFT coord origin
                origin = str(getattr(bbox, "coord_origin", "TOPLEFT")).upper()
                if "BOTTOM" in origin:
                    y0_pt = page_h_pt - bbox.b
                    y1_pt = page_h_pt - bbox.t
                else:
                    y0_pt = bbox.t
                    y1_pt = bbox.b
                x0_pt, x1_pt = bbox.l, bbox.r

                x0 = max(0, int(round(x0_pt * sx)))
                y0 = max(0, int(round(y0_pt * sy)))
                x1 = min(page_w_px, int(round(x1_pt * sx)))
                y1 = min(page_h_px, int(round(y1_pt * sy)))
                if x1 <= x0 or y1 <= y0:
                    continue

                label = str(c.label).lower() if c.label else "text"
                conf = float(getattr(c, "confidence", 0.0) or 0.0)
                regions.append(
                    Region(
                        page_index=page_index,
                        bbox=(x0, y0, x1, y1),
                        label=label,
                        detect_conf=conf,
                    )
                )

        regions = _dedup_overlapping(regions)
        if fill_gaps:
            regions.extend(_uncovered_regions(
                regions, page_index, page_w_px, page_h_px
            ))
        pages_out.append(regions)

    return pages_out


def _uncovered_regions(
    regions: List[Region], page_index: int, page_w: int, page_h: int,
    min_coverage_fraction: float = 0.05,
) -> List[Region]:
    """
    Synthesize regions covering rows of the page that Docling left uncovered.
    Approach: rasterize a coarse row-mask (one bit per pixel-row), mark rows
    covered by any cluster, then emit one synthetic region per contiguous
    uncovered horizontal band. Each band spans the full page width.

    Bands shorter than `min_coverage_fraction` of the page height are dropped
    to avoid spamming the cascade with sliver regions.
    """
    if page_h <= 0 or page_w <= 0:
        return []
    covered = bytearray(page_h)
    for r in regions:
        y0 = max(0, min(page_h - 1, r.bbox[1]))
        y1 = max(0, min(page_h, r.bbox[3]))
        for y in range(y0, y1):
            covered[y] = 1

    out: List[Region] = []
    min_h = max(1, int(min_coverage_fraction * page_h))
    i = 0
    while i < page_h:
        if covered[i]:
            i += 1
            continue
        start = i
        while i < page_h and not covered[i]:
            i += 1
        end = i  # exclusive
        if end - start >= min_h:
            out.append(Region(
                page_index=page_index,
                bbox=(0, start, page_w, end),
                label="_uncovered",
                detect_conf=0.0,
            ))
    return out


def reading_order_sort(regions: List[Region]) -> List[Region]:
    """Top-to-bottom, left-to-right with horizontal banding so side-by-side
    blocks aren't split across rows."""
    return sorted(regions, key=lambda r: (r.bbox[1] // BAND_HEIGHT_PX, r.bbox[0]))


def assemble_page(regions: List[Region]) -> str:
    """Sort regions in reading order and join their text with blank lines."""
    parts: List[str] = []
    for r in reading_order_sort(regions):
        if r.text:
            parts.append(r.text.strip())
    return "\n\n".join(p for p in parts if p)


def page_confidence(regions: List[Region]) -> float:
    """Area-weighted mean of region OCR confidences."""
    total_area = sum(r.area for r in regions if r.area > 0)
    if total_area == 0:
        return 0.0
    weighted = sum(r.ocr_conf * r.area for r in regions if r.area > 0)
    return weighted / total_area
