from .base import OCREngine
from .ollama_glm import OllamaGLMEngine
from .docling_engine import DoclingEngine
from .paddle_engine import PaddleEngine
from .tesseract_engine import TesseractEngine
from .merge_engine import MergeTesseractPaddleEngine

__all__ = [
    "OCREngine", "OllamaGLMEngine", "DoclingEngine",
    "PaddleEngine", "TesseractEngine", "MergeTesseractPaddleEngine",
]


def get_engine(name: str, use_gpu: bool = False) -> "OCREngine":
    engines = {
        "glm-ocr": OllamaGLMEngine,
        "docling": DoclingEngine,
        "paddleocr": PaddleEngine,
        "tesseract": TesseractEngine,
        "merge-tess-paddle": MergeTesseractPaddleEngine,
    }
    if name == "chandra":
        from .chandra_engine import ChandraEngine
        return ChandraEngine(use_gpu=use_gpu)
    if name not in engines:
        raise ValueError(f"Unknown engine: {name}. Choose from: {', '.join(engines)}, chandra")
    cls = engines[name]
    # Engines that accept use_gpu; tesseract is always CPU
    if name in ("docling", "paddleocr", "glm-ocr", "merge-tess-paddle"):
        return cls(use_gpu=use_gpu)
    return cls()


def all_engines(use_gpu: bool = False) -> list:
    available = []
    for name, cls in [
        ("glm-ocr", OllamaGLMEngine),
        ("docling", DoclingEngine),
        ("paddleocr", PaddleEngine),
        ("tesseract", TesseractEngine),
    ]:
        if name in ("docling", "paddleocr", "glm-ocr"):
            available.append((name, cls(use_gpu=use_gpu)))
        else:
            available.append((name, cls()))

    try:
        from .chandra_engine import ChandraEngine
        available.append(("chandra", ChandraEngine(use_gpu=use_gpu)))
    except ImportError:
        pass

    return available
