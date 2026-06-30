import os
import re
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class OCREngine(ABC):
    name: str
    use_gpu: bool = False

    @abstractmethod
    def extract(self, pdf_path: str, apply_preprocess: bool = False) -> List[str]:
        """Extract text from each page of the PDF. Returns one string per page."""
        ...

    def extract_with_confidence(
        self, pdf_path: str, apply_preprocess: bool = False
    ) -> Tuple[List[str], List[float]]:
        """
        Extract text and return per-page confidence scores (0.0–1.0).
        Default: calls extract() then scores via heuristic.
        Engines with native confidence should override this.
        """
        pages = self.extract(pdf_path, apply_preprocess)
        scores = [heuristic_confidence(p) for p in pages]
        return pages, scores

    def extract_image(
        self, image: "PILImage", *, apply_preprocess: bool = False, dpi: int = 300
    ) -> Tuple[str, float]:
        """
        Extract text + confidence from a single cropped PIL Image.
        Default: write a single-page temp PDF and delegate to
        extract_with_confidence. Engines that natively accept images
        (tesseract, glm-ocr, paddleocr) should override this to skip
        the PDF round-trip.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "region.pdf")
            _pil_to_single_page_pdf(image, pdf_path, dpi=dpi)
            pages, confs = self.extract_with_confidence(pdf_path, apply_preprocess)
        text = pages[0] if pages else ""
        conf = confs[0] if confs else 0.0
        return text, conf

    def extract_image_structured(
        self, image: "PILImage", *, apply_preprocess: bool = False, dpi: int = 300,
        **kwargs,
    ) -> Tuple[str, float, list]:
        """
        Extract text + confidence + per-line layout from a cropped PIL Image.

        `lines` is a list of dicts: {text, bbox, conf}, one per output line,
        bboxes in the input image's coordinate system. Engines that don't
        track per-line layouts return an empty list.

        Engines may accept extra kwargs (e.g. `strike_anchors`) and ignore
        the ones they don't use.
        """
        text, conf = self.extract_image(image, apply_preprocess=apply_preprocess, dpi=dpi)
        return text, conf, []

    def __repr__(self) -> str:
        return f"<OCREngine: {self.name}>"


def _pil_to_single_page_pdf(image: "PILImage", out_path: str, dpi: int = 300) -> None:
    """Write a PIL image as a single-page PDF at the given DPI."""
    img = image
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(out_path, format="PDF", resolution=float(dpi))


def heuristic_confidence(text: str) -> float:
    """
    Estimate text quality heuristically for engines that don't emit native confidence.

    Combines:
    - Printable character ratio (garbled OCR → lots of non-printable chars)
    - Alphabetic word ratio (real text → mostly words, not symbol soup)
    """
    if not text or not text.strip():
        return 0.0

    total = len(text)
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t\r")
    printable_ratio = printable / total

    non_space = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if not non_space:
        return 0.0
    alpha_chars = sum(1 for c in non_space if c.isalpha())
    alpha_ratio = alpha_chars / len(non_space)

    return round(printable_ratio * 0.4 + alpha_ratio * 0.6, 4)
