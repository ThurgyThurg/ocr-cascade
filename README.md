# Cascade

A multi-engine OCR pipeline for PDF documents with per-region routing, hand-edit detection, redaction marking, and a side-by-side web viewer.

Cascade runs each page through an ordered list of OCR engines — cheapest first, heaviest last — and stops at the first engine that clears a confidence threshold. **Regions mode** takes it further: Docling's layout model identifies blocks on the page (printed text, handwritten text, tables, pictures), and each block is routed through its own configured engine ladder. Mixed pages — a printed letterhead with a handwritten body, a court filing with margin annotations — only pay for the engine each piece actually needs.

---

## What it produces

- **`.txt`** — plain transcription, in reading order.
- **`.docx`** — Word output that preserves document structure: case captions render as two-column tables, handwritten margin annotations attach as **Word comments** anchored to the line they point to, struck text on edited drafts renders with strikethrough formatting, redaction bars are replaced with the literal token `[REDACTED]`.
- Confidence numbers per page and per region.

---

## Quick start

```bash
# 1. Drop into the project shell (NixOS — provides ollama, tesseract, python 3.13, libGL, glib)
nix-shell

# 2. Create the venv, install Python deps, pull the glm-ocr model (~2.2 GB)
./setup.sh

# 3a. Run the web UI
uvicorn app:app --host 127.0.0.1 --port 8765
#     → http://127.0.0.1:8765

# 3b. Or run a single document on the CLI
python ocr_pipeline.py path/to/file.pdf --engine regions --output results/
```

The first regions-mode run downloads two small Docling model checkpoints from Hugging Face (~700 MB combined). After that, everything runs locally.

---

## Modes

### `--engine cascade`
One engine per page. Starts with Tesseract (fast, fine on clean printed scans), escalates to PaddleOCR, then to `glm-ocr` (a 1.1B vision-language model served via Ollama) if confidence is still below the threshold. Heavy fallbacks (Docling for full OCR, optional Chandra 9B) follow.

### `--engine regions` *(default for the web UI)*
1. Docling's layout transformer runs in layout-only mode (no OCR) on the PDF and returns labeled bounding boxes per page (`text`, `handwritten_text`, `table`, `picture`, `page_header`, etc.).
2. Each region runs through its own cascade defined in `region_config.yaml`.
3. The page text is assembled in reading order from the per-region transcriptions; page confidence is the area-weighted mean of region confidences.

### Cascade vs regions

| | Cascade | Regions |
|---|---|---|
| Layout detection | none | Docling |
| Per-region routing | no | yes |
| Per-region engine ladders | n/a | yes, configurable |
| Annotation / comment detection | no | yes (margin handwriting → Word comments) |
| Strike-through detection | no | yes (struck text gets strikethrough in the .docx) |
| Redaction marking | yes | yes |
| Best for | clean printed documents | mixed-content pages, court filings, edited drafts |

---

## Region configuration

`region_config.yaml` maps each Docling label to an ordered engine ladder. The cascade stops at the first engine that clears `--threshold`.

```yaml
text:
  ladder: [tesseract, merge-tess-paddle, glm-ocr]

handwritten_text:
  ladder: [glm-ocr]

table:
  ladder: [docling, paddleocr, glm-ocr]

picture:
  ladder: [glm-ocr]

# … and so on for all DocItemLabels
```

Edit the file directly or use the web UI's **Cascade Rules** view to reorder, add, or remove engines per label. The next run picks up the changes automatically — no server restart needed.

---

## Web UI

Drop a single PDF or a whole folder onto the upload card. Watch each engine and region scroll past in real time on a pleading-paper-style trace. When the job finishes, the side-by-side viewer shows the **source PDF** on the left (pdf.js) and the **rendered transcription** on the right (docx-preview), proportionally scroll-synced. Light and dark themes toggle from the masthead.

Other UI features:

- **Docket sidebar** — every completed run persists to `runs/<job-id>/` and re-appears in the sidebar after a server restart.
- **Strike-from-record** — hover any docket entry, click `×` twice to delete the run from memory and disk.
- **Batch upload** — drop multiple PDFs or a folder; each runs sequentially with its own row, status badge, and per-file `.txt` / `.docx` downloads, plus a "bundle all (.zip)" button when the batch finishes.
- **Redaction notice** — when the preprocessor detects redaction bars, a banner shows how many were marked per page.
- **Live cascade trace** — in regions mode, each region row keeps a chain of attempts (`tesseract 68.2% ▸ merge-tess-paddle 84.1% ▸ glm-ocr 92.3% ✓`) so you can see *why* the cascade escalated.

---

## CLI reference

```text
python ocr_pipeline.py <file.pdf> [options]

Engines / modes
  --engine regions            layout-routed cascade (recommended)
  --engine cascade            whole-page cascade
  --engine tesseract|paddleocr|glm-ocr|docling|chandra
                              run a single engine directly
  --engine all --compare      run every engine, print a side-by-side table

Output
  --output <dir>              write .txt / .docx (default: results/)
  --dpi <int>                 render DPI for OCR (default: 300)

Behavior
  --threshold 0.90            confidence at which the cascade stops
  --preprocess                deskew + binarize before OCR
  --no-redactions             skip redaction-bar detection
  --gpu                       use CUDA for engines that support it
  --region-config <path>      override region_config.yaml
```

---

## Engines

| Engine | What it is | Used for |
|---|---|---|
| **Tesseract 5** | Apache LSTM OCR | clean printed text, fast first pass |
| **PaddleOCR** (PP-OCRv4 EN) | Baidu's OCR stack | character recovery on noisy scans; cheap second opinion |
| **merge-tess-paddle** | This project | splices PaddleOCR's character output into Tesseract's word-bbox whitespace — catches typewriter-era errors without losing spacing |
| **Docling** | IBM, layout + TableFormer | layout detection in regions mode; full OCR fallback; table-structure parsing |
| **glm-ocr** | ZhipuAI's 1.1B VLM, served via Ollama | handwriting, degraded scans, pictures-with-text |
| **Chandra 2** (optional) | 9B VLM | heaviest fallback; requires `pip install 'chandra-ocr[hf]'` |

---

## Project layout

```
.
├── app.py                  FastAPI server + endpoints
├── static/                 web UI (index.html, app.css, app.js)
├── ocr_pipeline.py         CLI entry point
│
├── cascade.py              whole-page cascade orchestrator
├── region_cascade.py       per-region cascade orchestrator
├── regions.py              Docling layout-only wrapper
├── region_config.py        YAML loader
├── region_config.yaml      default label → ladder mapping
│
├── engines/                per-engine implementations
│   ├── base.py
│   ├── tesseract_engine.py
│   ├── paddle_engine.py
│   ├── merge_engine.py
│   ├── docling_engine.py
│   ├── ollama_glm.py
│   └── chandra_engine.py   (optional, loaded lazily)
│
├── preprocess.py           PDF → images, redaction detection
├── annotations.py          margin-handwriting detection + OCR
├── strike.py               strike-through detection (line-level + per-word)
│
├── docx_writer/            from-scratch OOXML writer (no python-docx)
│   ├── __init__.py
│   ├── document.py
│   ├── blocks.py
│   ├── runs.py
│   ├── comments.py
│   ├── template.py
│   └── pack.py
│
├── shell.nix               NixOS dev shell
└── setup.sh                installs Python deps + pulls glm-ocr
```

---

## Known limitations

- **glm-ocr on CPU is slow.** ~5–7 minutes per page for handwriting on a typical laptop. The cascade only invokes it when cheaper engines fail; in regions mode you can constrain `handwritten_text` and `picture` labels to specific engines via `region_config.yaml` to control runtime.
- **GPU support is incomplete on older NVIDIA cards.** Current PyTorch wheels require compute capability sm_70+. Quadro P-series (sm_61) and earlier are CPU-only.
- **Strikethrough detection is heuristic.** Long, clearly-drawn horizontal strikes that span multiple words are detected reliably (the gap-spanning algorithm in `strike.py`). Short or wavy strikes on individual words may not register — those will be transcribed as normal text instead of struck.
- **Docling label drift.** Docling occasionally mis-labels regions (a hand-edited form may come back as `table`; a signature may come back as `picture`). The default ladders for those labels include `glm-ocr` as a final fallback so the cascade can still produce something usable, at the cost of running the VLM.

---

## Acknowledgments

Cascade stands on the shoulders of:

- [Tesseract](https://github.com/tesseract-ocr/tesseract) — Apache 2.0
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — Apache 2.0
- [IBM Docling](https://github.com/DS4SD/docling) — MIT
- [glm-ocr](https://huggingface.co/zhipuai/glm-ocr) (via [Ollama](https://ollama.com)) — MIT
- [pdf.js](https://github.com/mozilla/pdf.js) and [docx-preview](https://github.com/VolodymyrBaydalka/docxjs) — viewer panes
- [FastAPI](https://github.com/tiangolo/fastapi) + [Uvicorn](https://github.com/encode/uvicorn) — web server
