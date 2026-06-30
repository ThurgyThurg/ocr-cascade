"""PDF-to-images conversion, OpenCV preprocessing, and redaction detection."""

import io
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """Convert each page of a PDF to a PIL Image at the given DPI."""
    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def normalize_pdf(pdf_path: str) -> str:
    """Round-trip a PDF through PyMuPDF to produce a `garbage=4 deflate clean`
    rewrite that PDFium can reliably parse. Some scan-tool output (PDF/A
    streams, unusual xref tables) trips docling-parse's PDFium with a
    "Data format error" even though PyMuPDF reads it cleanly. Rewriting
    through PyMuPDF produces a structurally identical document that PDFium
    accepts.

    Returns the path to the normalized PDF (a new tmp file). The caller is
    responsible for deleting it. Returns the original path unchanged if the
    rewrite fails for any reason — better to let downstream surface the
    real error than to silently swallow a normalization failure.
    """
    import tempfile
    try:
        doc = fitz.open(pdf_path)
        out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        out.close()
        doc.save(out.name, garbage=4, deflate=True, clean=True)
        doc.close()
        return out.name
    except Exception:
        return pdf_path


def preprocess(img: Image.Image) -> Image.Image:
    """
    Enhance a scanned document image for OCR:
    grayscale → denoise → deskew → Otsu binarize
    """
    import cv2

    arr = np.array(img.convert("RGB"))
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    denoised = cv2.medianBlur(gray, 3)
    deskewed = _deskew(denoised)
    _, binary = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Detect and correct skew angle using Hough line transform."""
    import cv2

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return gray

    angles = []
    for line in lines[:50]:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def detect_redactions(img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Detect redacted (solid dark rectangle) regions in a document image.

    Returns a list of (x, y, w, h) bounding boxes for each redaction.

    Heuristic — a redaction bar is a directly-thresholded dark blob that's
    much denser than nearby letter shapes. We deliberately do NOT apply a
    horizontal close before contour-finding: on handwritten forms, a close
    kernel bridges the redaction bar to adjacent handwriting strokes,
    producing a long low-density contour that fails downstream filters. Filters:
      * area in [0.05%, 5%] of the page (smaller end catches inline-name
        bars; upper end rejects whole-page photocopier noise)
      * `w >= h` (width-or-square, not tall-and-narrow)
      * `h >= 8 px` (excludes form ruled lines which contour as 1–3 px tall)
      * dark-fill density >= 50% inside the bbox
    """
    import cv2

    arr = np.array(img.convert("L"))  # grayscale

    # Threshold for dark regions (redactions are near-black; 80 catches
    # slightly grayer photocopy bars too without sweeping in body text).
    _, dark = cv2.threshold(arr, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.width * img.height
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < img_area * 0.0005 or area > img_area * 0.05:
            continue
        if w < h:
            continue
        if h < 8:
            continue
        roi = dark[y : y + h, x : x + w]
        density = roi.sum() / (255 * area)
        if density < 0.50:
            continue
        boxes.append((x, y, w, h))

    return boxes


def apply_redaction_markers(img: Image.Image) -> Image.Image:
    """
    Replace detected redaction regions with a clearly-visible "[REDACTED]"
    label that survives downstream OCR engines, including vision models that
    downsample their input (glm-ocr at 1280px max-edge, etc.).

    Implementation notes:
      * Font size is decoupled from box height. A small inline redaction
        bar (e.g. 30 px tall) painted with a 24 px font becomes ~9 px after
        glm-ocr's downsample and disappears. We use a min font size of 48
        and let the text extend vertically past the original bar — better
        a slightly overflowing label than an unreadable one.
      * The label is centered horizontally on the bar and may extend left/
        right if the bar is narrower than the rendered text.
      * A thin black border around the original bbox makes the redaction
        visually obvious in the rendered docx preview and to humans
        reviewing the marked page.
    """
    boxes = detect_redactions(img)
    if not boxes:
        return img
    return _paint_redaction_markers(img, boxes)


def _paint_redaction_markers(img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    result = img.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    img_w, img_h = result.size

    # Per-DPI scale guess — at 300 DPI we want font ~64 px so it survives
    # glm-ocr's 0.4× downsample to ~25 px. At 150 DPI 32 px suffices. Use
    # img height as a proxy.
    base_font = max(48, int(img_h / 50))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", base_font
        )
    except (IOError, OSError):
        font = ImageFont.load_default()

    text = "[REDACTED]"
    try:
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
        tw, th = tx1 - tx0, ty1 - ty0
    except AttributeError:  # very old Pillow
        tw, th = draw.textsize(text, font=font)

    for x, y, w, h in boxes:
        # White-out the bar (slight pad keeps the original edges from showing).
        draw.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], fill=(255, 255, 255))

        # Center the label on the bar's center; allow vertical/horizontal
        # overflow so the text never gets cropped to invisibility.
        cx, cy = x + w // 2, y + h // 2
        ltx = max(0, min(img_w - tw, cx - tw // 2))
        lty = max(0, min(img_h - th, cy - th // 2))

        # White halo behind the text in case it overflows onto handwriting.
        pad = max(4, base_font // 8)
        draw.rectangle(
            [ltx - pad, lty - pad, ltx + tw + pad, lty + th + pad],
            fill=(255, 255, 255),
        )
        # Thin border around the *original* bar (not the halo) so reviewers
        # can see where the redaction actually sat.
        draw.rectangle(
            [x, y, x + w, y + h],
            outline=(0, 0, 0), width=2,
        )
        draw.text((ltx, lty), text, fill=(0, 0, 0), font=font)
    return result


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    import base64
    return base64.b64encode(image_to_bytes(img, fmt)).decode("utf-8")
