"""
OCR via IBM Docling.

Handles PDF natively using a transformer-based layout analyser. Excellent on
multi-column layouts and tables (97.9% table extraction accuracy). Returns the
full document as markdown, split back into page-sized chunks for comparison.
"""

from typing import List
from .base import OCREngine


class DoclingEngine(OCREngine):
    name = "docling"

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

    def extract(self, pdf_path: str, apply_preprocess: bool = False) -> List[str]:
        pages, _ = self.extract_with_confidence(pdf_path, apply_preprocess)
        return pages

    def extract_with_confidence(self, pdf_path: str, apply_preprocess: bool = False):
        from docling.document_converter import DocumentConverter

        # Try docling 2.x API first; fall back to minimal API for older versions
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True

            # Force the Tesseract OCR backend. Docling's default RapidOCR
            # auto-picks the torch backend and tries to load PP-OCRv6.det.small,
            # which isn't shipped — it crashes with "Unsupported configuration:
            # torch.PP-OCRv6.det.small". We already have tesseract in nix-shell.
            try:
                from docling.datamodel.pipeline_options import TesseractCliOcrOptions
                pipeline_options.ocr_options = TesseractCliOcrOptions()
            except ImportError:
                try:
                    from docling.datamodel.pipeline_options import TesseractOcrOptions
                    pipeline_options.ocr_options = TesseractOcrOptions()
                except ImportError:
                    pass  # leave default and hope for the best

            if self.use_gpu:
                try:
                    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
                    pipeline_options.accelerator_options = AcceleratorOptions(
                        device=AcceleratorDevice.CUDA
                    )
                except (ImportError, AttributeError):
                    pass

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        except (ImportError, TypeError):
            # Older / simplified docling build
            converter = DocumentConverter()

        try:
            result = converter.convert(pdf_path)
        except Exception as first_err:
            # Retry with a PyMuPDF-normalized copy — same workaround used in
            # regions.detect_regions for PDFs that trip PDFium.
            from preprocess import normalize_pdf
            normalized = normalize_pdf(pdf_path)
            if normalized == pdf_path:
                raise
            import os
            try:
                result = converter.convert(normalized)
            except Exception:
                raise first_err
            finally:
                try:
                    os.unlink(normalized)
                except OSError:
                    pass
        doc = result.document
        md = doc.export_to_markdown()

        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1
        pages = self._split_by_pages(md, page_count)
        from .base import heuristic_confidence
        confs = [heuristic_confidence(p) for p in pages]
        return pages, confs

    def _split_by_pages(self, md: str, page_count: int) -> List[str]:
        if page_count <= 1:
            return [md]
        lines = md.splitlines()
        chunk_size = max(1, len(lines) // page_count)
        pages = []
        for i in range(page_count):
            start = i * chunk_size
            end = start + chunk_size if i < page_count - 1 else len(lines)
            pages.append("\n".join(lines[start:end]))
        return pages
