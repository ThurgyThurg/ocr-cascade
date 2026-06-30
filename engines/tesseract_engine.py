"""
OCR via Tesseract 5 + OpenCV preprocessing.

Native word-level confidence via pytesseract.image_to_data().
Good baseline for clean printed text; poor on handwriting.
"""

from typing import List, Tuple, TYPE_CHECKING
from .base import OCREngine

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class TesseractEngine(OCREngine):
    name = "tesseract"

    def extract(self, pdf_path: str, apply_preprocess: bool = True) -> List[str]:
        pages, _ = self.extract_with_confidence(pdf_path, apply_preprocess)
        return pages

    def extract_with_confidence(
        self, pdf_path: str, apply_preprocess: bool = True
    ) -> Tuple[List[str], List[float]]:
        from preprocess import pdf_to_images, preprocess

        images = pdf_to_images(pdf_path)
        all_text, all_conf = [], []

        for img in images:
            processed = preprocess(img) if apply_preprocess else img
            text, conf = self._ocr_pil(processed)
            all_text.append(text)
            all_conf.append(conf)

        return all_text, all_conf

    def extract_image(
        self, image: "PILImage", *, apply_preprocess: bool = False, dpi: int = 300
    ) -> Tuple[str, float]:
        from preprocess import preprocess as preprocess_fn
        processed = preprocess_fn(image) if apply_preprocess else image
        return self._ocr_pil(processed)

    def _ocr_pil(self, img: "PILImage") -> Tuple[str, float]:
        import pytesseract
        data = pytesseract.image_to_data(
            img,
            config="--psm 6 --oem 3",
            output_type=pytesseract.Output.DICT,
        )
        word_confs = [int(c) for c in data["conf"] if int(c) > 0]
        page_text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
        conf = sum(word_confs) / (len(word_confs) * 100) if word_confs else 0.0
        return page_text, round(conf, 4)
