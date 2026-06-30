"""
Strikethrough detection — v1.2 (line-level, multi-span).

v1.1 found the single longest high-density run per line. That worked for
lines with one strike, but Vinson p6 has a line with TWO strikes ("has not
been made"… "by"), so a single-run model leaves the second strike unmarked
and incorrectly absorbs the unstruck words between them.

v1.2 scans at the line level (same y-row max-density approach), then
extracts EVERY high-density run on that row that passes the coverage-gap
filter — yielding a list of pixel x-spans per line. Two public entry points:

  detect_strike_spans(image, tess_data) -> Dict[line_key, List[(x_l, x_r)]]
      raw pixel spans grouped by tesseract (block, par, line) key.

  detect_strikethrough(image, word_bboxes, *, tess_data=None) -> Set[int]
      legacy compatibility — returns indices of word_bboxes whose center
      falls inside any strike span (used by callers that only need per-word
      booleans).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Defaults tuned on Vinson p6 — struck body line max_row ≈ 0.37, unstruck
# body lines stay below 0.25. 0.28 gives a comfortable margin.
DEFAULT_DARK_THRESHOLD = 130
DEFAULT_MIN_ROW_FRACTION = 0.28
DEFAULT_MIN_LINE_WIDTH = 200
DEFAULT_MIN_LINE_HEIGHT = 15

# Rolling-window scan params. A 40-px window with 0.40 density-threshold
# catches mid-gap strikes (where tesseract failed) AND end-of-line strikes
# (which extend into the right margin), without firing on dense all-caps
# titles that stay below 0.40 over any 40-px run.
_RUN_WINDOW = 40
_RUN_THRESH = 0.40
_RUN_MIN_LEN = 30
_RUN_MERGE_GAP = 80
_RUN_MIN_FINAL_WIDTH = 35
_GAP_MIN_WIDTH = 80
_RIGHT_MARGIN_TOLERANCE = 20


def detect_strike_spans(
    image,
    tess_data: dict,
    *,
    anchors: Optional[Sequence[Tuple[int, int]]] = None,
    dark_threshold: int = DEFAULT_DARK_THRESHOLD,
    min_row_fraction: float = DEFAULT_MIN_ROW_FRACTION,
    min_line_width: int = DEFAULT_MIN_LINE_WIDTH,
    min_line_height: int = DEFAULT_MIN_LINE_HEIGHT,
) -> Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]:
    """Return per-line strike pixel spans keyed by tesseract (block, par,
    line). Each span is `(x_left, x_right, strike_y)` in `image` coords.

    Multiple spans per line are supported — a line with two separate
    hand-drawn strikes (e.g. "has not been made" followed later by "by")
    yields two entries.

    `anchors` is an optional list of `(x, y)` image-coord points marking
    body-side endpoints of annotation lead-lines. Tess lines that contain
    any anchor's y get a permissive second-pass scan tuned for short
    strikes (one or two words) that the line-wide scan misses because
    most of the line is clean printed text.
    """
    arr = np.asarray(image.convert("L"))
    img_h, img_w = arr.shape

    lines: Dict[Tuple[int, int, int], List[Tuple[int, int, int, int]]] = defaultdict(list)
    for i in range(len(tess_data.get("text", []))):
        if int(tess_data["level"][i]) != 5:
            continue
        text = str(tess_data["text"][i]).strip()
        if not text:
            continue
        key = (
            int(tess_data["block_num"][i]),
            int(tess_data["par_num"][i]),
            int(tess_data["line_num"][i]),
        )
        left = int(tess_data["left"][i])
        top = int(tess_data["top"][i])
        right = left + int(tess_data["width"][i])
        bottom = top + int(tess_data["height"][i])
        lines[key].append((left, top, right, bottom))

    spans_per_line: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}
    for key, members in lines.items():
        spans = _scan_line_for_strikes(
            arr, img_h, members,
            dark_threshold=dark_threshold,
            min_row_fraction=min_row_fraction,
            min_line_width=min_line_width,
            min_line_height=min_line_height,
        )
        if spans:
            spans_per_line[key] = spans

    # Anchor-driven second pass is DISABLED. Margin-annotation
    # `anchor_xy` points returned by `annotations.trace_lead_line` are
    # not reliable enough to drive an additional scan: the body endpoint
    # often lands on the wrong y, and even when it lands on the struck
    # line, the underlying hand-drawn strike is too wavy to register at
    # the same intensity threshold as the printed letters. A more reliable
    # signal would require either:
    #   (a) annotation OCR available at strike time (so the strike is
    #       known to correspond to a real edit), or
    #   (b) a tighter detector (e.g. medium-intensity pen-mask + vertical
    #       dilation) that doesn't false-fire on letter rows.
    # Both are out of scope for this fork right now. The `anchors` kwarg
    # is kept as a future-extension hook; passing it currently does
    # nothing.
    _ = anchors  # explicitly unused
    return spans_per_line


def _anchored_strike_scan(
    arr: np.ndarray,
    img_h: int,
    members: List[Tuple[int, int, int, int]],
    anchor_x: int,
    anchor_y: int,
) -> List[Tuple[int, int, int]]:
    """Permissive strike scan within a narrow y-band around the anchor's y.
    Uses a wider dark threshold (180, not 130) so light pen strokes
    register, and a smaller rolling window (30) to surface short strikes.
    """
    if not members:
        return []
    line_l = min(m[0] for m in members)
    line_r = max(m[2] for m in members)
    band_y0 = max(0, anchor_y - 6)
    band_y1 = min(img_h, anchor_y + 7)
    if band_y1 - band_y0 < 3:
        return []
    sub = arr[band_y0:band_y1, line_l:line_r] < 180
    # Any-dark per x within the y band (collapses wavy strikes onto one row).
    any_dark = sub.any(axis=0).astype(np.int32)
    density = _rolling_mean(any_dark, 30)
    runs = _all_true_runs(density >= 0.50, min_length=30)
    if not runs:
        return []
    # Only accept runs that anchor at or before anchor_x (the lead line
    # points into the strike's right end).
    spans: List[Tuple[int, int, int]] = []
    for s, e in runs:
        sx_l = line_l + s
        sx_r = line_l + e
        if sx_l > anchor_x + 40:
            continue  # strike must start near or before the anchor's x
        spans.append((sx_l, sx_r, anchor_y))
    return spans


def detect_strikethrough(
    image,
    word_bboxes: Sequence,
    *,
    tess_data: Optional[dict] = None,
    dark_threshold: int = DEFAULT_DARK_THRESHOLD,
    min_row_fraction: float = DEFAULT_MIN_ROW_FRACTION,
    min_line_width: int = DEFAULT_MIN_LINE_WIDTH,
    min_line_height: int = DEFAULT_MIN_LINE_HEIGHT,
) -> Set[int]:
    """Legacy per-word API. Returns indices of `word_bboxes` whose center
    pixel-x falls inside any detected strike span on its line.

    Note: words that fall in a strike's gap (tesseract failed to recognize
    them, so they don't appear in `word_bboxes`) are not represented in the
    return value. Use `detect_strike_spans` to capture those.
    """
    if tess_data is None:
        return set()

    spans_per_line = detect_strike_spans(
        image, tess_data,
        dark_threshold=dark_threshold,
        min_row_fraction=min_row_fraction,
        min_line_width=min_line_width,
        min_line_height=min_line_height,
    )
    if not spans_per_line:
        return set()

    # Map each (block, par, line) key back to its members for the per-word
    # marking.
    bbox_to_idx: dict = {tuple(b): i for i, b in enumerate(word_bboxes)}
    members_by_key: Dict[Tuple[int, int, int], List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
    for i in range(len(tess_data.get("text", []))):
        if int(tess_data["level"][i]) != 5:
            continue
        text = str(tess_data["text"][i]).strip()
        if not text:
            continue
        key = (
            int(tess_data["block_num"][i]),
            int(tess_data["par_num"][i]),
            int(tess_data["line_num"][i]),
        )
        left = int(tess_data["left"][i])
        top = int(tess_data["top"][i])
        right = left + int(tess_data["width"][i])
        bottom = top + int(tess_data["height"][i])
        idx = bbox_to_idx.get((left, top, right, bottom))
        if idx is None:
            continue
        members_by_key[key].append((idx, (left, top, right, bottom)))

    struck: Set[int] = set()
    for key, spans in spans_per_line.items():
        members = members_by_key.get(key, [])
        for span_l, span_r, strike_y in spans:
            for idx, (x0, y0, x1, y1) in members:
                cx = (x0 + x1) // 2
                if span_l <= cx <= span_r and (y0 - 4 <= strike_y <= y1 + 4):
                    struck.add(idx)
    return struck


def _scan_line_for_strikes(
    arr: np.ndarray,
    img_h: int,
    members: List[Tuple[int, int, int, int]],
    *,
    dark_threshold: int,
    min_row_fraction: float,
    min_line_width: int,
    min_line_height: int,
) -> List[Tuple[int, int, int]]:
    """Return list of `(x_left, x_right, strike_y)` strike spans on one line."""
    if not members:
        return []
    line_l = min(b[0] for b in members)
    line_t = min(b[1] for b in members)
    line_r = max(b[2] for b in members)
    line_b = max(b[3] for b in members)
    line_w = line_r - line_l
    line_h = line_b - line_t
    if line_w < min_line_width or line_h < min_line_height:
        return []

    band_y0 = max(0, line_t + int(line_h * 0.20))
    band_y1 = min(img_h, line_b - int(line_h * 0.20))
    if band_y1 - band_y0 < 3:
        return []

    sub = arr[band_y0:band_y1, line_l:line_r] < dark_threshold
    row_fractions = sub.sum(axis=1) / max(1, line_w)
    max_idx = int(row_fractions.argmax())
    max_frac = float(row_fractions[max_idx])
    if max_frac < min_row_fraction:
        return []

    strike_y = band_y0 + max_idx
    strike_row = sub[max_idx]
    density = _rolling_mean(strike_row.astype(np.int32), _RUN_WINDOW)
    runs = _all_true_runs(density >= _RUN_THRESH, min_length=_RUN_MIN_LEN)
    if not runs:
        return []

    # Pixel positions, then merge runs whose intra-gap is small. A single
    # hand-drawn strike often produces several sub-runs (a dip below the
    # density threshold mid-stroke) that all belong together.
    abs_runs = [(line_l + s, line_l + e) for s, e in runs]
    abs_runs = _merge_close(abs_runs, _RUN_MERGE_GAP)
    abs_runs = [(s, e) for s, e in abs_runs if e - s >= _RUN_MIN_FINAL_WIDTH]
    if not abs_runs:
        return []

    # False-positive guard: an all-caps centered heading
    # ("SUPREME COURT OF THE UNITED STATES") produces a similar dense
    # horizontal stripe at the cap-line, but its words run continuously
    # with no big tesseract-coverage gap. A real strike either covers a
    # gap (struck words that tesseract failed on) or extends past the
    # line's right margin (an end-of-line strike whose lead-line trails
    # into the annotation margin).
    sorted_members = sorted(members, key=lambda b: b[0])
    gaps: List[Tuple[int, int]] = []
    prev_right = line_l
    for bbox in sorted_members:
        gap_l = prev_right
        gap_r = bbox[0]
        if (gap_r - gap_l) >= _GAP_MIN_WIDTH:
            gaps.append((gap_l, gap_r))
        prev_right = max(prev_right, bbox[2])
    right_margin_start = prev_right - _RIGHT_MARGIN_TOLERANCE

    spans: List[Tuple[int, int, int]] = []
    for sx_l, sx_r in abs_runs:
        overlaps_gap = False
        for gl, gr in gaps:
            if not (sx_r < gl or sx_l > gr):
                overlaps_gap = True
                break
        if not overlaps_gap and sx_l < right_margin_start:
            continue
        spans.append((sx_l, sx_r, strike_y))
    return spans


def _merge_close(runs: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    if not runs:
        return []
    runs = sorted(runs)
    out: List[List[int]] = [list(runs[0])]
    for s, e in runs[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean of a 1-D int array with the same length as
    input. Uses cumulative-sum trick for O(n)."""
    n = arr.shape[0]
    if window <= 1 or n <= 1:
        return arr.astype(np.float32)
    pad = window // 2
    padded = np.pad(arr, (pad, window - pad), mode="edge")
    csum = np.cumsum(padded, dtype=np.float64)
    sums = csum[window:] - csum[:-window]
    return (sums / window).astype(np.float32)[:n]


def _all_true_runs(mask: np.ndarray, *, min_length: int = 1) -> List[Tuple[int, int]]:
    """Return all (start, end_exclusive) True-runs at least `min_length` long."""
    n = int(mask.shape[0]) if hasattr(mask, "shape") else len(mask)
    runs: List[Tuple[int, int]] = []
    if n == 0:
        return runs
    cur_start: Optional[int] = None
    for i in range(n):
        if mask[i]:
            if cur_start is None:
                cur_start = i
        elif cur_start is not None:
            if i - cur_start >= min_length:
                runs.append((cur_start, i))
            cur_start = None
    if cur_start is not None and n - cur_start >= min_length:
        runs.append((cur_start, n))
    return runs
