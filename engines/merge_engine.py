"""
MergeTesseractPaddleEngine — ensemble OCR engine.

Combines Tesseract's word-level layout (via image_to_data) with PaddleOCR's
text. For each tesseract line we find paddleocr text whose bbox overlaps the
line, then splice them so paddleocr's characters land at positions tesseract's
spaces dictate. Net effect: PaddleOCR's character accuracy + Tesseract's
whitespace fidelity.

Confidence: paddleocr's per-box average over lines where it took over;
tesseract's word-conf average for lines where paddleocr produced nothing.
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, TYPE_CHECKING

from .base import OCREngine

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class MergeTesseractPaddleEngine(OCREngine):
    name = "merge-tess-paddle"

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._paddle = None

    def _get_paddle(self):
        if self._paddle is None:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(
                use_angle_cls=True, lang="en", show_log=False
            )
        return self._paddle

    def extract(self, pdf_path: str, apply_preprocess: bool = False) -> List[str]:
        pages, _ = self.extract_with_confidence(pdf_path, apply_preprocess)
        return pages

    def extract_with_confidence(
        self, pdf_path: str, apply_preprocess: bool = False
    ) -> Tuple[List[str], List[float]]:
        from preprocess import pdf_to_images, preprocess as preprocess_fn

        images = pdf_to_images(pdf_path)
        texts, confs = [], []
        for img in images:
            processed = preprocess_fn(img) if apply_preprocess else img
            text, conf = self._merge_image(processed)
            texts.append(text)
            confs.append(conf)
        return texts, confs

    def extract_image(
        self, image: "PILImage", *, apply_preprocess: bool = False, dpi: int = 300
    ) -> Tuple[str, float]:
        from preprocess import preprocess as preprocess_fn
        processed = preprocess_fn(image) if apply_preprocess else image
        text, conf, _ = self._merge_image(processed)
        return text, conf

    def extract_image_structured(
        self, image: "PILImage", *, apply_preprocess: bool = False, dpi: int = 300,
        strike_anchors: Optional[list] = None,
    ) -> Tuple[str, float, list]:
        from preprocess import preprocess as preprocess_fn
        processed = preprocess_fn(image) if apply_preprocess else image
        return self._merge_image(processed, strike_anchors=strike_anchors)

    def _merge_image(
        self, image: "PILImage", *, strike_anchors: Optional[list] = None,
    ) -> Tuple[str, float, list]:
        tess_words, tess_data = self._tesseract_words(image)
        paddle_items = self._paddle_items(image)
        # Strike pixel spans grouped by tesseract (block, par, line) key.
        # Per-paddle-box mapping happens inside _merge so each line span
        # carries `struck_segments: List[(start, end)]` in merged-text coords.
        strike_spans_per_line: dict = {}
        try:
            from strike import detect_strike_spans
            strike_spans_per_line = detect_strike_spans(
                image, tess_data, anchors=strike_anchors,
            )
        except Exception:
            strike_spans_per_line = {}
        return _merge(tess_words, paddle_items, tess_data, strike_spans_per_line)

    def _tesseract_words(self, image: "PILImage"):
        """Return (words, raw_data) where words is a flat list of word
        dicts and raw_data is the full pytesseract DICT (so callers like
        the strike detector can use the block/par/line grouping)."""
        import pytesseract
        data = pytesseract.image_to_data(
            image, config="--psm 6 --oem 3",
            output_type=pytesseract.Output.DICT,
        )
        out: List[dict] = []
        for i in range(len(data["text"])):
            if int(data["level"][i]) != 5:  # 5 = word
                continue
            word = str(data["text"][i]).strip()
            if not word:
                continue
            left = int(data["left"][i])
            top = int(data["top"][i])
            right = left + int(data["width"][i])
            bottom = top + int(data["height"][i])
            conf = int(data["conf"][i])
            line_key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            out.append({
                "text": word,
                "bbox": (left, top, right, bottom),
                "conf": conf,
                "line_key": line_key,
            })
        return out, data

    def _paddle_items(self, image: "PILImage") -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """Return [(text, (x0, y0, x1, y1), conf), ...]."""
        ocr = self._get_paddle()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            img = image if image.mode == "RGB" else image.convert("RGB")
            img.save(path)
            raw = ocr.ocr(path, cls=True)
        items: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        if not raw or raw[0] is None:
            return items
        for line in raw[0]:
            if not line or len(line) != 2:
                continue
            bbox_pts, payload = line
            text, conf = payload
            xs = [pt[0] for pt in bbox_pts]
            ys = [pt[1] for pt in bbox_pts]
            items.append((
                str(text),
                (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                float(conf),
            ))
        return items


def _merge(
    tess_words: List[dict],
    paddle_items: List[Tuple[str, Tuple[int, int, int, int], float]],
    tess_data: Optional[dict] = None,
    strike_spans_per_line: Optional[dict] = None,
) -> Tuple[str, float, list]:
    """Walk paddleocr boxes in reading order (the layout authority — handles
    multi-column blocks because each column is a separate box). For each
    paddle box, find the tesseract words *inside* it and use their joined
    text as the whitespace template to splice paddle's characters into.

    Returns `(text, conf, lines)` where `lines` is a list of
    `{text, bbox, conf, struck_segments}` per output line. `struck_segments`
    is a list of `(char_start, char_end)` ranges in merged-text coords for
    portions that the strike detector flagged as crossed out."""
    line_spans: list = []
    strike_spans_per_line = strike_spans_per_line or {}

    if not paddle_items:
        # No paddle output — fall back to tesseract banded lines.
        confs: List[float] = []
        for line_text, line_conf, line_bbox in _tess_words_to_lines(tess_words):
            line_spans.append({
                "text": line_text, "bbox": line_bbox, "conf": line_conf,
                "struck_segments": [],
            })
            confs.append(line_conf)
        final = "\n".join(span["text"] for span in line_spans)
        mean = round(sum(confs) / len(confs), 4) if confs else 0.0
        return final, mean, line_spans

    band_h = _estimate_line_height(paddle_items)
    sorted_paddle = sorted(
        paddle_items,
        key=lambda p: (((p[1][1] + p[1][3]) // 2) // band_h, p[1][0]),
    )

    used_tess = [False] * len(tess_words)
    confs: List[float] = []

    for ptext, pbox, pconf in sorted_paddle:
        inside_idx = [
            i for i, w in enumerate(tess_words)
            if _is_center_inside(w["bbox"], pbox)
        ]
        inside_idx.sort(key=lambda i: tess_words[i]["bbox"][0])

        struck_segments: List[Tuple[int, int]] = []
        if inside_idx:
            tess_text = " ".join(tess_words[i]["text"] for i in inside_idx)
            for i in inside_idx:
                used_tess[i] = True
            merged = _splice_whitespace(tess_text, ptext)

            # Identify which tesseract (block, par, line) keys the paddle box
            # covers, then look up strike spans for those keys.
            key_counts: dict = defaultdict(int)
            for i in inside_idx:
                k = tess_words[i].get("line_key")
                if k is not None:
                    key_counts[k] += 1
            box_strike_spans: List[Tuple[int, int, int]] = []
            for k in key_counts:
                box_strike_spans.extend(strike_spans_per_line.get(k, []))

            if box_strike_spans:
                struck_segments = _map_strike_spans_to_merged(
                    box_strike_spans,
                    inside_idx,
                    tess_words,
                    tess_text,
                    merged,
                )
        else:
            merged = ptext

        struck_fraction = (
            sum(e - s for s, e in struck_segments) / max(1, len(merged))
            if struck_segments else 0.0
        )

        line_spans.append({
            "text": merged,
            "bbox": tuple(int(v) for v in pbox),
            "conf": float(pconf),
            "struck_fraction": struck_fraction,
            "struck": bool(struck_segments),
            "struck_segments": struck_segments,
        })
        confs.append(pconf)

    # Tesseract words that paddle missed — append as supplementary lines.
    leftover = [tess_words[i] for i in range(len(tess_words)) if not used_tess[i]]
    if leftover:
        for line_text, line_conf, line_bbox in _tess_words_to_lines(leftover):
            line_spans.append({
                "text": line_text, "bbox": line_bbox, "conf": line_conf,
                "struck_segments": [],
            })
            confs.append(line_conf)

    final_text = "\n".join(span["text"] for span in line_spans)
    final_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
    return final_text, final_conf, line_spans


def _is_center_inside(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
    """True if the center of `inner` falls within `outer`."""
    cx = (inner[0] + inner[2]) // 2
    cy = (inner[1] + inner[3]) // 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _estimate_line_height(paddle_items: List[Tuple[str, Tuple[int, int, int, int], float]]) -> int:
    """Median paddleocr box height; used as the reading-order banding factor."""
    heights = [p[1][3] - p[1][1] for p in paddle_items if p[1][3] > p[1][1]]
    if not heights:
        return 50
    heights.sort()
    median = heights[len(heights) // 2]
    return max(20, int(median * 0.7))


def _tess_words_to_lines(words: List[dict]) -> List[Tuple[str, float]]:
    """Reconstruct line strings from a list of tesseract word dicts using a
    band-by-y sort (same banding heuristic as the paddle-box ordering)."""
    if not words:
        return []
    heights = [w["bbox"][3] - w["bbox"][1] for w in words if w["bbox"][3] > w["bbox"][1]]
    band = max(20, int((sorted(heights)[len(heights) // 2] if heights else 50) * 0.7))
    sorted_words = sorted(
        words,
        key=lambda w: (((w["bbox"][1] + w["bbox"][3]) // 2) // band, w["bbox"][0]),
    )
    lines: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    current: List[dict] = []
    current_band = None

    def _emit(current_words):
        text = " ".join(x["text"] for x in current_words)
        conf = sum(x["conf"] for x in current_words) / (len(current_words) * 100.0)
        x0 = min(x["bbox"][0] for x in current_words)
        y0 = min(x["bbox"][1] for x in current_words)
        x1 = max(x["bbox"][2] for x in current_words)
        y1 = max(x["bbox"][3] for x in current_words)
        lines.append((text, round(conf, 4), (x0, y0, x1, y1)))

    for w in sorted_words:
        cy = (w["bbox"][1] + w["bbox"][3]) // 2
        b = cy // band
        if current_band is None or b == current_band:
            current.append(w)
            current_band = b
        else:
            _emit(current)
            current = [w]
            current_band = b
    if current:
        _emit(current)
    return lines


def _splice_whitespace(tess: str, paddle: str) -> str:
    """Align paddle's characters to tess via SequenceMatcher; insert a space
    in the output wherever tess has a space immediately before the aligned
    position. Net effect: paddle's character accuracy + tess's whitespace."""
    sm = SequenceMatcher(None, tess, paddle, autojunk=False)
    paddle_to_tess: dict = {}
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            paddle_to_tess[block.b + k] = block.a + k

    out: List[str] = []
    for i, c in enumerate(paddle):
        if i in paddle_to_tess:
            t_idx = paddle_to_tess[i]
            if t_idx > 0 and tess[t_idx - 1] == " " and (not out or out[-1] != " "):
                out.append(" ")
        out.append(c)
    return "".join(out)


def _map_strike_spans_to_merged(
    box_strike_spans: List[Tuple[int, int, int]],
    inside_idx: List[int],
    tess_words: List[dict],
    tess_text: str,
    merged_text: str,
) -> List[Tuple[int, int]]:
    """Convert pixel-space strike spans into merged-text char ranges.

    For each pixel span `(sx_l, sx_r, _y)`:
      * tess char anchors come from the words on either side of the span —
        last word ending before `sx_l` gives `tess_left`; first word
        starting after `sx_r` gives `tess_right`. Words centered inside
        the span are recognized-struck words; their char range is included.
      * The tess char range is then mapped to merged via SequenceMatcher
        opcodes — `insert`s inside the range (paddle-only text where tesseract
        saw nothing) get included, so badly-OCR'd struck content like
        "Laas net bean me a" lands in the struck merged range even though
        tesseract returned blank.
    """
    if not box_strike_spans:
        return []

    # Build (tess_char_start, tess_char_end) per inside word, in the order
    # they appear in `tess_text` (which is " ".join of inside words by x0).
    word_char_ranges: List[Tuple[int, int, int, int]] = []  # (cs, ce, bx0, bx1)
    cursor = 0
    for i in inside_idx:
        w = tess_words[i]
        wt = w["text"]
        ce = cursor + len(wt)
        word_char_ranges.append((cursor, ce, w["bbox"][0], w["bbox"][2]))
        cursor = ce + 1  # +1 for the joining space

    opcodes = SequenceMatcher(None, tess_text, merged_text, autojunk=False).get_opcodes()

    out: List[Tuple[int, int]] = []
    for sx_l, sx_r, _strike_y in box_strike_spans:
        left_anchor = 0
        right_anchor = len(tess_text)
        contained_first: Optional[int] = None
        contained_last: Optional[int] = None
        for cs, ce, bx0, bx1 in word_char_ranges:
            bcx = (bx0 + bx1) // 2
            if bcx < sx_l:
                left_anchor = ce
            elif bcx > sx_r:
                right_anchor = min(right_anchor, cs)
                break
            else:
                if contained_first is None:
                    contained_first = cs
                contained_last = ce
        # If words are contained in the strike, expand anchors to cover them.
        if contained_first is not None:
            tess_start = min(left_anchor, contained_first)
            tess_end = max(right_anchor, contained_last)
        else:
            tess_start = left_anchor
            tess_end = right_anchor
        tess_start = max(0, min(len(tess_text), tess_start))
        tess_end = max(tess_start, min(len(tess_text), tess_end))

        merged_start, merged_end = _tess_range_to_merged_range(
            tess_start, tess_end, opcodes, len(merged_text)
        )
        if merged_end > merged_start:
            # Trim leading/trailing whitespace so we don't render a struck
            # space hanging off the end of an otherwise unstruck word.
            while merged_start < merged_end and merged_text[merged_start] == " ":
                merged_start += 1
            while merged_end > merged_start and merged_text[merged_end - 1] == " ":
                merged_end -= 1
            if merged_end > merged_start:
                out.append((merged_start, merged_end))
    # Merge overlapping segments.
    if not out:
        return []
    out.sort()
    merged_out: List[Tuple[int, int]] = [out[0]]
    for s, e in out[1:]:
        ps, pe = merged_out[-1]
        if s <= pe:
            merged_out[-1] = (ps, max(pe, e))
        else:
            merged_out.append((s, e))
    return merged_out


def _tess_range_to_merged_range(
    tess_start: int,
    tess_end: int,
    opcodes,
    merged_len: int,
) -> Tuple[int, int]:
    """Map a tess-char range to a merged-char range. Paddle-only inserts
    whose tess position falls inside the range are absorbed into the
    merged range (so badly-OCR'd struck text is included)."""
    merged_start: Optional[int] = None
    merged_end: Optional[int] = None
    for tag, i1, i2, j1, j2 in opcodes:
        # Insert: i1 == i2; it occurs "between" tess positions i1 and i1.
        if tag == "insert":
            if tess_start <= i1 <= tess_end:
                if merged_start is None:
                    merged_start = j1
                merged_end = j2
            continue
        # Skip opcodes entirely outside [tess_start, tess_end].
        if i2 <= tess_start or i1 >= tess_end:
            continue
        local_s = max(0, tess_start - i1)
        local_e = min(i2 - i1, tess_end - i1)
        if tag == "equal":
            cand_s = j1 + local_s
            cand_e = j1 + local_e
        else:  # replace / delete
            # Map proportionally across the paddle side of the replace.
            tess_len = max(1, i2 - i1)
            paddle_len = j2 - j1
            cand_s = j1 + round(local_s / tess_len * paddle_len)
            cand_e = j1 + round(local_e / tess_len * paddle_len)
        if merged_start is None:
            merged_start = cand_s
        merged_end = cand_e
    if merged_start is None:
        return 0, 0
    return max(0, merged_start), min(merged_len, merged_end if merged_end is not None else merged_start)
