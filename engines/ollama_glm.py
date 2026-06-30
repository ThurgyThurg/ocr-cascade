"""
OCR via Ollama glm-ocr (1.1B, MIT).

glm-ocr is an OCR-specialized vision model, ranked #1 on OmniDocBench V1.5
(94.62/100). Uses the Ollama REST API directly — the Python ollama library
has a known issue with image encoding for custom model architectures.

Large images (>~1MB base64) cause the vision encoder to degrade into
prompt-echo / repetition. We downsample to a max edge of MAX_EDGE_PX and
re-encode as JPEG before sending.

API: POST http://localhost:11434/api/generate
"""

import base64
import io
import json
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

from .base import OCREngine, heuristic_confidence

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

OLLAMA_URL = "http://localhost:11434"
MAX_EDGE_PX = 1280
JPEG_QUALITY = 88


class OllamaGLMEngine(OCREngine):
    name = "glm-ocr"
    MODEL = "glm-ocr"

    def __init__(self, use_gpu: bool = False):
        self._server_started = False
        self.use_gpu = use_gpu  # Ollama auto-detects NVIDIA GPU via libcuda.so

    def _ensure_server(self):
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
            return
        except Exception:
            pass
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait up to 10 seconds for Ollama to start
        for _ in range(10):
            time.sleep(1)
            try:
                urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
                self._server_started = True
                return
            except Exception:
                continue
        raise RuntimeError("Ollama server did not start in time")

    def _generate(self, b64_image: str) -> str:
        """Call Ollama REST API with a base64 image, return generated text."""
        payload = json.dumps({
            "model": self.MODEL,
            "prompt": (
                "Transcribe every word visible in this image exactly as written. "
                "Include handwriting, stamps, printed text, and any marks. "
                "Do not summarize, interpret, or add any words not present in the image. "
                "Output only the raw transcription, preserving line breaks."
            ),
            "images": [b64_image],
            "stream": False,
            "options": {"temperature": 0},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return _strip_echoed_prompt(data.get("response", ""))

    @staticmethod
    def _encode_for_glm(img) -> str:
        """Downsample to MAX_EDGE_PX and JPEG-encode; return base64 string."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > MAX_EDGE_PX:
            scale = MAX_EDGE_PX / longest
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def extract(self, pdf_path: str, apply_preprocess: bool = False) -> List[str]:
        from preprocess import pdf_to_images, preprocess

        self._ensure_server()

        images = pdf_to_images(pdf_path)
        results = []

        for i, img in enumerate(images):
            if apply_preprocess:
                img = preprocess(img)
            b64 = self._encode_for_glm(img)
            results.append(self._generate(b64))

        return results

    def extract_image(
        self, image: "PILImage", *, apply_preprocess: bool = False, dpi: int = 300
    ) -> Tuple[str, float]:
        from preprocess import preprocess as preprocess_fn
        self._ensure_server()
        img = preprocess_fn(image) if apply_preprocess else image
        b64 = self._encode_for_glm(img)
        text = self._generate(b64)
        return text, heuristic_confidence(text)


# Phrases the model is known to echo from the prompt. Matched case-
# insensitively as a substring at the end of a line; once we find one we
# drop it and keep walking backwards until we hit a non-prompt line.
_PROMPT_ECHO_MARKERS = (
    "transcribe every word",
    "exactly as written",
    "preserving line breaks",
    "include handwriting",
    "do not summarize",
    "do not interpret",
    "output only the raw transcription",
    "any marks",
)


def _strip_echoed_prompt(text: str) -> str:
    """Some glm-ocr generations append (parts of) the prompt to the end of
    their response. Trim trailing lines whose content matches any known
    prompt phrase so it doesn't leak into the final transcription."""
    if not text:
        return text
    lines = text.rstrip().split("\n")
    while lines:
        last = lines[-1].strip().lower()
        if not last:
            lines.pop()
            continue
        if any(marker in last for marker in _PROMPT_ECHO_MARKERS):
            lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()
