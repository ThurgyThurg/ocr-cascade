"""
Public entry point for the .docx writer.

`write_docx(out_path, pages, **kwargs)` keeps the same signature as the old
python-docx implementation so callers don't need to change. The new
implementation builds OOXML directly.

Heuristics applied per page (same as the python-docx version):
  * Detected case-caption blocks (Plaintiff / Petitioner / v. / Defendant /
    et al. with optional docket lines like "On Writ of Certiorari to...")
    are rendered as a two-column table.
  * `[Month —, 1950.]`-style bracketed dates are centered.
  * Other lines are flowing paragraphs.
  * No metadata table or per-page heading.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .blocks import Block, ParagraphBlock, TableBlock, PageBreakBlock, TextRun
from .comments import Comment, build_comments_xml
from .document import build_document_xml
from .pack import pack_docx
from .template import (
    content_types_xml,
    document_rels_xml,
    root_rels_xml,
    styles_xml,
)


def write_docx(
    out_path: str,
    pages: Sequence[str],
    *,
    annotations: Optional[Sequence[Sequence[dict]]] = None,
    line_spans_per_page: Optional[Sequence[Sequence[dict]]] = None,
    **_ignored,
) -> None:
    """Render OCR pages to a Word document at `out_path`.

    annotations — optional, one list per page. Each annotation is a dict
        {text, anchor_y_relative, author, initials}. Each annotation is
        attached as a Word comment to the page paragraph at the proportional
        y-position `anchor_y_relative` (0.0 = first paragraph, 1.0 = last).
        Empty / falsy text annotations are skipped.

    line_spans_per_page — optional, one list of line dicts per page in the
        same order non-empty lines appear in `pages[i]`. Each line dict may
        carry `{struck: bool, struck_fraction: float, bbox: ...}`. Used to
        apply strikethrough formatting to the matching paragraphs.

    Extra kwargs (title, metadata, per_page, region_detail) are accepted
    for backward compatibility with old callers but no longer rendered.
    """
    blocks: List[Block] = []
    comments_to_emit: List[Comment] = []
    next_comment_id = 0

    for i, page_text in enumerate(pages):
        if i > 0:
            blocks.append(PageBreakBlock())
        page_spans = (
            line_spans_per_page[i]
            if line_spans_per_page and i < len(line_spans_per_page)
            else None
        )
        page_blocks = _render_page_to_blocks(page_text or "", page_spans)

        if annotations and i < len(annotations):
            next_comment_id = _attach_annotations(
                page_blocks,
                annotations[i] or [],
                comments_to_emit,
                next_comment_id,
            )

        blocks.extend(page_blocks)

    include_comments = bool(comments_to_emit)

    parts = {
        "[Content_Types].xml": content_types_xml(include_comments=include_comments),
        "_rels/.rels": root_rels_xml(),
        "word/document.xml": build_document_xml(blocks),
        "word/_rels/document.xml.rels": document_rels_xml(include_comments=include_comments),
        "word/styles.xml": styles_xml(),
    }
    if include_comments:
        parts["word/comments.xml"] = build_comments_xml(comments_to_emit)

    pack_docx(out_path, parts)


def _attach_annotations(
    page_blocks: List[Block],
    page_annotations: Sequence[dict],
    comments_out: List[Comment],
    next_id: int,
) -> int:
    """For each annotation on this page, locate the target paragraph by
    `anchor_y_relative`, attach a comment_id to its first run, and append a
    Comment to `comments_out`. Returns the next available comment id.

    Two annotations on the same page can't share a paragraph (the
    commentRange markers would conflict), so we deduplicate by bumping the
    target index forward when a paragraph's already claimed. To make that
    bumping produce sensible ordering, we process annotations in top-to-
    bottom order by y-anchor.
    """
    para_blocks = [
        b for b in page_blocks
        if isinstance(b, ParagraphBlock) and b.runs
    ]
    n_paras = len(para_blocks)
    if n_paras == 0:
        return next_id

    sorted_annotations = sorted(
        page_annotations,
        key=lambda a: float(a.get("anchor_y_relative", 0.0)),
    )

    used_paragraph_indices: set = set()

    for ann in sorted_annotations:
        text = (ann.get("text") or "").strip()
        if not text:
            continue
        y_rel = ann.get("anchor_y_relative", 0.0)
        y_rel = max(0.0, min(1.0, float(y_rel)))
        target_idx = min(int(y_rel * n_paras), n_paras - 1)
        while target_idx in used_paragraph_indices and target_idx < n_paras - 1:
            target_idx += 1
        used_paragraph_indices.add(target_idx)
        target = para_blocks[target_idx]
        target.runs[0].comment_id = next_id
        comments_out.append(Comment(
            id=next_id,
            text=text,
            author=ann.get("author", "[handwritten]"),
            initials=ann.get("initials", "HW"),
        ))
        next_id += 1
    return next_id


# ── Heuristic page renderer (same logic as the previous python-docx writer) ──

def _render_page_to_blocks(
    page_text: str,
    line_spans: Optional[Sequence[dict]] = None,
) -> List[Block]:
    raw_lines = [ln.rstrip() for ln in page_text.split("\n")]
    spans_iter = iter(line_spans or [])

    def _next_span():
        return next(spans_iter, None)

    def _runs_for_line(line_text: str, span):
        """Build the TextRun list for one rendered line.

        The span may carry `struck_segments: List[(start, end)]` — char ranges
        in `line_text` that the strike detector flagged as crossed out. We
        walk those segments left-to-right and emit alternating unstruck /
        struck runs. A line with no segments renders as one plain run.

        Backwards-compatible fallback: if only the legacy `struck` flag is
        set (no segments), the whole line is marked struck.
        """
        if span is None:
            return [TextRun(text=line_text)]
        raw_segments = span.get("struck_segments")
        if raw_segments is None:
            if bool(span.get("struck")):
                return [TextRun(text=line_text, strike=True)]
            return [TextRun(text=line_text)]

        # Clamp + sort + drop empties.
        segs: list = []
        n = len(line_text)
        for s, e in raw_segments:
            s = max(0, min(n, int(s)))
            e = max(0, min(n, int(e)))
            if e > s:
                segs.append((s, e))
        segs.sort()
        if not segs:
            return [TextRun(text=line_text)]

        runs: list = []
        cursor = 0
        for s, e in segs:
            if s > cursor:
                runs.append(TextRun(text=line_text[cursor:s]))
            runs.append(TextRun(text=line_text[s:e], strike=True))
            cursor = e
        if cursor < n:
            runs.append(TextRun(text=line_text[cursor:]))
        return runs

    blocks: List[Block] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        stripped = line.strip()

        if not stripped:
            blocks.append(ParagraphBlock(runs=[]))
            i += 1
            continue

        end = _detect_case_caption_end(raw_lines, i)
        if end > i:
            blocks.append(_make_caption_table(raw_lines[i:end + 1]))
            # Each non-empty raw line in the caption slice consumed a span.
            for k in range(i, end + 1):
                if raw_lines[k].strip():
                    next(spans_iter, None)
            i = end + 1
            continue

        if _looks_like_centered_bracket(stripped):
            span = _next_span()
            blocks.append(ParagraphBlock(
                runs=_runs_for_line(stripped, span),
                alignment="center",
            ))
            i += 1
            continue

        span = _next_span()
        blocks.append(ParagraphBlock(runs=_runs_for_line(line, span)))
        i += 1

    return blocks


_CAPTION_START_EXCLUDE_PREFIXES = (
    "see ", "to:", "from:", "circulated", "revised", "re:",
)


def _detect_case_caption_end(lines: List[str], start: int) -> int:
    """If `lines[start:]` opens a case caption block, return the inclusive
    end-line index. Else -1.

    Heuristic: within a small window from `start`, look for
        (Plaintiff lines) → "Petitioner," → "v."/"vs."/"U." (paddle misread)
        → (Defendant lines) → "et al."

    `start` must look like the petitioner's name line — reject obvious
    routing-slip / instruction lines so they don't get pulled into the table.
    """
    s = lines[start].strip()
    if not s:
        return -1
    if s.lower().startswith(_CAPTION_START_EXCLUDE_PREFIXES):
        return -1

    pet_idx = -1
    for j in range(start, min(start + 4, len(lines))):
        if "Petitioner" in lines[j]:
            pet_idx = j
            break
    if pet_idx < 0:
        return -1

    v_idx = -1
    for j in range(pet_idx + 1, min(pet_idx + 4, len(lines))):
        s = lines[j].strip().rstrip(".").lower()
        if s in ("v", "vs", "u"):
            v_idx = j
            break
    if v_idx < 0:
        return -1

    end_idx = -1
    for j in range(v_idx + 1, min(v_idx + 7, len(lines))):
        if "et al" in lines[j].lower():
            end_idx = j
            break
    return end_idx


RIGHT_COL_PREFIXES = (
    "on writ",
    "the supreme",
    "the state",
    "the district",
    "the united states",
    "the court of",
    "the circuit",
    "the high court",
)


def _make_caption_table(caption_lines: List[str]) -> TableBlock:
    left: List[str] = []
    right: List[str] = []
    for raw in caption_lines:
        s = raw.strip()
        if not s:
            continue
        if s.lower().startswith(RIGHT_COL_PREFIXES):
            right.append(s)
        else:
            left.append(s)

    rows = max(len(left), len(right), 1)
    grid: List[List[str]] = []
    for r in range(rows):
        grid.append([
            left[r] if r < len(left) else "",
            right[r] if r < len(right) else "",
        ])
    return TableBlock(rows=grid)


def _looks_like_centered_bracket(s: str) -> bool:
    return s.startswith("[") and s.endswith("]") and len(s) < 40
