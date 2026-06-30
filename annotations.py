"""
Margin-annotation detection.

A handwritten margin annotation is a small block of ink that lives outside
the printed body column — typically a Justice's editorial mark on a draft
opinion. We find them via two passes:

  1. Docling-label pass — re-use Docling regions whose label is
     `handwritten_text` or (`picture` outside the body column). These are
     candidates we already have bboxes for "for free."
  2. OpenCV fallback pass — threshold the page, find connected components
     in the left/right margin strips that look like handwriting, and
     dedupe against pass 1.

Returned annotations carry only bboxes; the caller is responsible for
OCR'ing them (typically via glm-ocr).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Anything whose x-center sits in the outer 22% of either side of the
# page is considered a margin candidate. The "IN A" annotation on Vinson p6
# spans x=2020–2180 on a 2550-wide page; 22% margin (right_edge=1989)
# captures the leading "IN" without intruding into body text (which lives
# at x<1900 in centered SCOTUS bodies).
MARGIN_X_FRACTION = 0.22
ANNOTATION_SOURCE_DOCLING = "docling"
ANNOTATION_SOURCE_OPENCV = "opencv"


@dataclass
class Annotation:
    page_index: int
    bbox: Tuple[int, int, int, int]   # (x0, y0, x1, y1) in page-pixel coords
    text: str = ""
    source: str = ANNOTATION_SOURCE_DOCLING
    anchor_xy: Optional[Tuple[int, int]] = None   # filled in Phase 2 only
    author: str = "[handwritten]"
    initials: str = "HW"

    @property
    def y_center(self) -> int:
        return (self.bbox[1] + self.bbox[3]) // 2

    @property
    def x_center(self) -> int:
        return (self.bbox[0] + self.bbox[2]) // 2


def detect_annotations(
    page_image,
    page_index: int,
    docling_regions: list,
) -> List[Annotation]:
    """Return candidate annotations for `page_image`. Doesn't OCR — caller
    runs glm-ocr on each Annotation.bbox separately."""
    w, h = page_image.size
    margin_x = int(w * MARGIN_X_FRACTION)
    left_edge = margin_x
    right_edge = w - margin_x
    page_area = w * h

    out: List[Annotation] = []
    claimed_bboxes: List[Tuple[int, int, int, int]] = []

    # Pass 1: Docling regions classified as annotations.
    #  - `handwritten_text` is an annotation only when the region sits in
    #    the outer margin (its x-centroid is in the left or right MARGIN_X
    #    strip) OR when it's small enough to plausibly be an interlinear
    #    edit (< 10% of page area). A handwritten_text region centered in
    #    the body column is the body itself — for a wholly handwritten
    #    letter, treating it as an annotation moves the whole letter into
    #    comments. Leave those as body regions.
    #  - `picture` is an annotation if it covers < 20% of the page area
    #    (excludes wide pictures like the SCOTUS routing slip whose bbox
    #    spans most of the right side).
    for r in docling_regions:
        x0, y0, x1, y1 = r.bbox
        area = max(0, (x1 - x0) * (y1 - y0))
        cx = (x0 + x1) // 2
        in_margin = (cx < left_edge) or (cx > right_edge)
        small_enough = area < 0.10 * page_area

        is_handwritten_annotation = (
            r.label == "handwritten_text" and (in_margin or small_enough)
        )
        is_picture_annotation = (
            r.label == "picture" and area < 0.20 * page_area
        )
        if is_handwritten_annotation or is_picture_annotation:
            out.append(Annotation(
                page_index=page_index,
                bbox=tuple(r.bbox),
                source=ANNOTATION_SOURCE_DOCLING,
            ))
            claimed_bboxes.append(tuple(r.bbox))

    # Pass 2: OpenCV fallback for margin blobs Docling missed.
    cv_bboxes = _opencv_margin_blobs(page_image, left_edge, right_edge)
    for bbox in cv_bboxes:
        # Skip blobs that are mostly contained in an already-claimed bbox
        # (e.g. fragments of the SCOTUS routing slip that Docling already
        # covered as a single picture region).
        if any(_contained_in(bbox, b, fraction=0.6) for b in claimed_bboxes):
            continue
        out.append(Annotation(
            page_index=page_index,
            bbox=bbox,
            source=ANNOTATION_SOURCE_OPENCV,
        ))
        claimed_bboxes.append(bbox)

    # Phase 2: trace lead lines to refine anchor targets where possible.
    for ann in out:
        target = trace_lead_line(page_image, ann.bbox)
        if target is not None:
            ann.anchor_xy = (int(target[0]), int(target[1]))

    return out


def ocr_annotation(annotation: Annotation, page_image, glm_engine) -> str:
    """Crop `page_image` to the annotation bbox (with a small pad) and run
    glm-ocr on it. Returns the transcribed text (stripped)."""
    pad = 12
    x0, y0, x1, y1 = annotation.bbox
    w, h = page_image.size
    crop = page_image.crop((
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    ))
    text, _ = glm_engine.extract_image(crop)
    return (text or "").strip()


def trace_lead_line(
    page_image,
    annotation_bbox: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int]]:
    """Find the lead line drawn from a margin annotation to a target body
    location. Returns `(x, y)` of the body-side endpoint, in page-pixel
    coordinates, or None if no line found.

    Trace strategy:
      - Determine which margin side the annotation is on.
      - Extract a horizontal strip near the annotation's y-band that runs
        from the body column toward the annotation.
      - Threshold, edge-detect, run HoughLinesP looking for short near-
        horizontal segments.
      - Of those, pick the one extending furthest into the body. Its
        body-side endpoint is the target.
    """
    import cv2
    import numpy as np

    x0, y0, x1, y1 = annotation_bbox
    arr = np.asarray(page_image.convert("L"))
    h, w = arr.shape

    margin_x = int(w * MARGIN_X_FRACTION)
    body_left = margin_x
    body_right = w - margin_x

    ann_cx = (x0 + x1) // 2
    in_right_margin = ann_cx > w // 2

    if in_right_margin:
        strip_x0 = max(0, body_left - 50)
        strip_x1 = max(strip_x0 + 1, x0)
    else:
        strip_x0 = min(w - 1, x1)
        strip_x1 = min(w, body_right + 50)

    pad_y = 35
    strip_y0 = max(0, y0 - pad_y)
    strip_y1 = min(h, y1 + pad_y)

    if strip_x1 - strip_x0 < 30 or strip_y1 - strip_y0 < 5:
        return None

    strip = arr[strip_y0:strip_y1, strip_x0:strip_x1]
    _, binary = cv2.threshold(strip, 200, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 40, 140)

    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180, threshold=18,
        minLineLength=25, maxLineGap=12,
    )
    if lines is None:
        return None

    best_target: Optional[Tuple[int, int]] = None
    best_dx = 0
    for line in lines:
        lx0, ly0, lx1, ly1 = line[0]
        dx = abs(lx1 - lx0)
        dy = abs(ly1 - ly0)
        if dx < 25 or dy > 6:
            continue
        if in_right_margin:
            if lx0 < lx1:
                body_x, body_y = lx0, ly0
            else:
                body_x, body_y = lx1, ly1
        else:
            if lx0 > lx1:
                body_x, body_y = lx0, ly0
            else:
                body_x, body_y = lx1, ly1
        if dx > best_dx:
            best_dx = dx
            best_target = (body_x + strip_x0, body_y + strip_y0)
    return best_target


def _opencv_margin_blobs(
    page_image, left_edge: int, right_edge: int
) -> List[Tuple[int, int, int, int]]:
    """Find connected-component blobs in the left/right margin strips that
    look like handwriting (not page-edge artifacts, not signature lines)."""
    import cv2
    import numpy as np

    arr = np.array(page_image.convert("L"))
    h, w = arr.shape

    # Fixed-threshold binarization: pencil handwriting on white paper
    # typically sits at intensity 100–180; printed ink is < 100; clean white
    # is > 200. A fixed threshold of 200 reliably catches both. We can't use
    # Otsu globally (biased by dark printed body) and we can't use Otsu per
    # strip (margin is overwhelmingly white → degenerate threshold).
    _, binary = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)

    # Mask to the margin strips only.
    margin_only = np.zeros_like(binary)
    margin_only[:, :left_edge] = binary[:, :left_edge]
    margin_only[:, right_edge:] = binary[:, right_edge:]

    # Close nearby strokes so each annotation comes out as one blob.
    # Wider kernel: typical handwritten annotations have intra-letter gaps
    # ~10-20 px at 300 DPI. We close horizontally further than vertically so
    # words on the same line merge but separate annotations on different
    # lines stay distinct.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 11))
    closed = cv2.morphologyEx(margin_only, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    out: List[Tuple[int, int, int, int]] = []
    min_bbox_area = 600
    max_bbox_area = 0.05 * w * h
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        bbox_area = ww * hh
        if not (min_bbox_area <= bbox_area <= max_bbox_area):
            continue
        if hh > 0.5 * h or ww < 16 or hh < 12:
            continue
        if ww > 0.35 * w:
            continue
        # Reject wide-and-short small blobs — page-edge scuffs, fold marks,
        # binder-punch artifacts. On the Vinson p6 scan these come through
        # as ~50×17 marks (aspect > 3) with bbox area < 1500. Real annotated
        # words on the same scan are squarer (the "A" in "IN A" is 30×28).
        aspect = ww / max(1, hh)
        if aspect > 2.5 and bbox_area < 1500:
            continue
        pad = 6
        out.append((
            max(0, x - pad),
            max(0, y - pad),
            min(w, x + ww + pad),
            min(h, y + hh + pad),
        ))
    return out


def _contained_in(inner, outer, *, fraction: float = 0.6) -> bool:
    """True if at least `fraction` of `inner`'s area sits inside `outer`."""
    ax0, ay0, ax1, ay1 = inner
    bx0, by0, bx1, by1 = outer
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    inner_area = max(1, (ax1 - ax0) * (ay1 - ay0))
    return inter / inner_area >= fraction
