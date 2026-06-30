"""
OCR via Chandra 2 (Qwen-3-VL 9B, HuggingFace backend).

Chandra 2 scores 85.9 on the olmocr benchmark, outperforming GPT-4o (69.9%)
and Gemini Flash 2 (63.8%). Excellent handwriting and complex layout support.

Install (opt-in, ~5 GB model download):
  pip install "chandra-ocr[hf]"

This engine is automatically skipped if chandra_ocr is not installed.
"""

import os
import tempfile
from typing import List

from .base import OCREngine


class ChandraEngine(OCREngine):
    name = "chandra"

    def __init__(self, use_gpu: bool = False):
        try:
            from chandra.inference import InferenceManager
        except ImportError:
            raise ImportError(
                "chandra-ocr is not installed. Run: pip install 'chandra-ocr[hf]'"
            )
        self.use_gpu = use_gpu
        device = "cuda" if use_gpu else "cpu"
        try:
            self._manager = InferenceManager(method="hf", device=device)
        except TypeError:
            # Older chandra API without device param
            self._manager = InferenceManager(method="hf")

    def extract(self, pdf_path: str, apply_preprocess: bool = False) -> List[str]:
        from preprocess import pdf_to_images, preprocess

        images = pdf_to_images(pdf_path)
        results = []

        with tempfile.TemporaryDirectory() as tmp:
            for i, img in enumerate(images):
                if apply_preprocess:
                    img = preprocess(img)
                page_path = os.path.join(tmp, f"page_{i}.png")
                img.save(page_path)

                # Chandra BatchInputItem wraps image path or PIL image
                try:
                    from chandra.inference import BatchInputItem
                    items = [BatchInputItem(image=page_path)]
                except ImportError:
                    items = [{"image": page_path}]

                output = self._manager.generate(items)
                text = output[0] if output else ""
                results.append(text)

        return results
