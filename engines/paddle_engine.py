"""
OCR via PaddleOCR PP-OCRv5.

Native per-box confidence scores. Strong on multi-column layouts, fast,
decent handwriting. Returns boxes sorted in reading order.

Handles PaddleOCR v2.x (ocr(path, cls=True) API) and v3.x (predict() API).
"""

import logging
import os
import tempfile
from typing import List, Tuple, TYPE_CHECKING

from .base import OCREngine

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

for _name in ("ppocr", "paddleocr", "paddle"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class PaddleEngine(OCREngine):
    name = "paddleocr"

    def __init__(self, use_gpu: bool = False):
        self._ocr = None
        self.use_gpu = use_gpu
        self._api_version = None  # detected on first call

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            kwargs = {"use_angle_cls": True, "lang": "en", "show_log": False}
            if self.use_gpu:
                try:
                    self._ocr = PaddleOCR(**kwargs, use_gpu=True)
                except Exception:
                    print("  [paddleocr] GPU init failed, falling back to CPU (install paddlepaddle-gpu for GPU)")
                    self._ocr = PaddleOCR(**kwargs)
            else:
                self._ocr = PaddleOCR(**kwargs)
        return self._ocr

    def extract(self, pdf_path: str, apply_preprocess: bool = False) -> List[str]:
        pages, _ = self.extract_with_confidence(pdf_path, apply_preprocess)
        return pages

    def extract_with_confidence(
        self, pdf_path: str, apply_preprocess: bool = False
    ) -> Tuple[List[str], List[float]]:
        from preprocess import pdf_to_images, preprocess

        ocr = self._get_ocr()
        images = pdf_to_images(pdf_path)
        all_text, all_conf = [], []

        with tempfile.TemporaryDirectory() as tmp:
            for i, img in enumerate(images):
                if apply_preprocess:
                    img = preprocess(img)
                page_path = os.path.join(tmp, f"page_{i}.png")
                img.save(page_path)

                text, conf = self._ocr_page(ocr, page_path)
                all_text.append(text)
                all_conf.append(conf)

        return all_text, all_conf

    def extract_image(
        self, image: "PILImage", *, apply_preprocess: bool = False, dpi: int = 300
    ) -> Tuple[str, float]:
        from preprocess import preprocess as preprocess_fn
        ocr = self._get_ocr()
        img = preprocess_fn(image) if apply_preprocess else image
        with tempfile.TemporaryDirectory() as tmp:
            page_path = os.path.join(tmp, "region.png")
            img.save(page_path)
            return self._ocr_page(ocr, page_path)

    def _ocr_page(self, ocr, page_path: str) -> Tuple[str, float]:
        # Try v2.x API: ocr(path, cls=True)
        if self._api_version in (None, 2):
            try:
                raw = ocr.ocr(page_path, cls=True)
                result = self._assemble_v2(raw)
                self._api_version = 2
                return result
            except TypeError:
                self._api_version = 3

        # v3.x API: predict(path)
        try:
            results = list(ocr.predict(page_path))
            return self._assemble_v3(results)
        except Exception:
            # Last resort: ocr() without cls kwarg
            try:
                raw = ocr.ocr(page_path)
                return self._assemble_v2(raw)
            except Exception:
                return "", 0.0

    def _assemble_v2(self, raw) -> Tuple[str, float]:
        """PaddleOCR v2.x: [[[bbox, [text, conf]], ...]]"""
        if not raw or raw[0] is None:
            return "", 0.0
        lines, confs = [], []
        for line in raw[0]:
            if line and len(line) == 2:
                text, conf = line[1]
                lines.append(str(text))
                confs.append(float(conf))
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return "\n".join(lines), round(avg_conf, 4)

    def _assemble_v3(self, results) -> Tuple[str, float]:
        """PaddleOCR v3.x (PaddleX): list of result objects with rec_texts/rec_scores."""
        lines, confs = [], []
        for res in results:
            texts = scores = []
            if hasattr(res, "rec_texts") and hasattr(res, "rec_scores"):
                texts = res.rec_texts
                scores = res.rec_scores
            elif isinstance(res, dict):
                texts = res.get("rec_texts", res.get("texts", []))
                scores = res.get("rec_scores", res.get("scores", []))
            for text, conf in zip(texts, scores):
                lines.append(str(text))
                confs.append(float(conf))
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return "\n".join(lines), round(avg_conf, 4)
