"""
FastAPI web UI for the OCR cascade pipeline.

Endpoints:
  GET  /                              — serve the single-page UI
  POST /ocr                           — single PDF upload, returns {"job_id"}
  GET  /stream/{id}                   — SSE stream of cascade progress events
  GET  /result/{id}                   — return final {"text", "engine", ...}
  POST /batch                         — multi-file batch upload, returns {"batch_id"}
  GET  /batch/{id}/stream             — SSE stream of per-file batch events
  GET  /batch/{id}/file/{idx}.{txt|docx}  — per-file download
  GET  /batch/{id}/all.zip            — bundle of all completed files
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

# Hide CUDA from torch unless the operator opts in by setting LQ_OCR_GPU=1
# before launching uvicorn. The default torch wheel auto-uses CUDA when
# libcuda.so is present, which crashes on older GPUs (sm < 75). The GPU
# checkbox in the UI is best-effort per-job; this flag controls the process.
if os.environ.get("LQ_OCR_GPU", "").lower() not in ("1", "true", "yes", "on"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import io
import zipfile
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(title="OCR Cascade UI")
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Per-job artifacts (source PDF, rendered docx, metadata) are written under
# RUNS_DIR so the viewer keeps working across server restarts. Without this
# the in-memory _jobs dict is wiped on every uvicorn relaunch and any job
# that finished in a previous lifecycle becomes a 404.
RUNS_DIR = Path(__file__).parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def _run_dir(job_id: str) -> Path:
    p = RUNS_DIR / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _persist_job(job_id: str, meta: dict, pdf_bytes: bytes,
                 text: str, docx_bytes: bytes | None) -> None:
    """Write everything a future request to /source/, /result/, or
    /result/{id}.docx might need. Safe to call multiple times; later calls
    overwrite. Best-effort: failures here must not fail the OCR pipeline."""
    try:
        d = _run_dir(job_id)
        if pdf_bytes:
            (d / "source.pdf").write_bytes(pdf_bytes)
        if text is not None:
            (d / "result.txt").write_text(text, encoding="utf-8")
        if docx_bytes:
            (d / "result.docx").write_bytes(docx_bytes)
        (d / "meta.json").write_text(json.dumps(meta, default=str), encoding="utf-8")
    except Exception:
        pass


def _read_persisted(job_id: str) -> dict | None:
    """Return a dict shaped like _jobs[job_id] sourced from disk, or None."""
    d = RUNS_DIR / job_id
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pdf_path = d / "source.pdf"
    docx_path = d / "result.docx"
    txt_path = d / "result.txt"
    text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
    docx = docx_path.read_bytes() if docx_path.exists() else None
    return {
        "result": None,                        # CascadeResult is not persisted
        "error": None,
        "done": True,
        "pdf_path": str(pdf_path) if pdf_path.exists() else "",
        "use_gpu": False,
        "mode": meta.get("mode", "regions"),
        "threshold": meta.get("threshold", 0.90),
        "name": meta.get("name", ""),
        "mean_conf": meta.get("mean_conf", 0.0),
        "engine_use": meta.get("engine_use", {}),
        "engine": meta.get("engine", ""),
        "pages": meta.get("pages", 0),
        "text": text,
        "docx": docx,
        "queue": None,
        "from_disk": True,
    }


# ── In-memory job store ────────────────────────────────────────────────────────
# Each job: {"queue": Queue, "result": CascadeResult|None, "error": str|None,
#            "done": bool, "pdf_path": str}
_jobs: dict[str, dict] = {}
_batches: dict[str, dict] = {}
_SENTINEL = object()  # signals SSE generator that the worker thread is done


def _get_job(job_id: str) -> dict | None:
    """Look up a job in memory; fall back to disk. Caches the disk lookup
    into _jobs so subsequent reads avoid IO."""
    if job_id in _jobs:
        return _jobs[job_id]
    j = _read_persisted(job_id)
    if j is not None:
        _jobs[job_id] = j
        return j
    return None


# ── POST /ocr ─────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.90  # confidence at or above which the cascade stops


def _clamp_threshold(raw: str) -> float:
    """Parse a `threshold` form value (in [0,1] or [0,100]) and clamp to
    [0.10, 1.00]. Out-of-range or unparseable → fall back to default."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    if v > 1.0:
        v = v / 100.0   # accept "90" as 0.90
    if v < 0.10:
        v = 0.10
    if v > 1.00:
        v = 1.00
    return v


@app.post("/ocr")
async def upload_pdf(
    file: UploadFile = File(...),
    use_gpu: str = Form("false"),
    mode: str = Form("cascade"),
    threshold: str = Form(str(DEFAULT_THRESHOLD)),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    gpu = use_gpu.lower() in ("true", "1", "on", "yes")
    mode = mode.lower().strip()
    if mode not in ("cascade", "regions"):
        mode = "cascade"
    threshold_f = _clamp_threshold(threshold)

    # Save upload to a temp file; the worker thread owns cleanup.
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    job_id = str(uuid.uuid4())
    q: Queue = Queue()
    _jobs[job_id] = {
        "queue": q,
        "result": None,
        "error": None,
        "done": False,
        "pdf_path": tmp_path,
        "use_gpu": gpu,
        "mode": mode,
        "threshold": threshold_f,
        "name": file.filename or "filing.pdf",
        "started_at": time.time(),
    }

    worker = _run_regions if mode == "regions" else _run_cascade
    t = threading.Thread(target=worker, args=(job_id, tmp_path, q, gpu, threshold_f), daemon=True)
    t.start()

    return JSONResponse({"job_id": job_id, "mode": mode, "threshold": threshold_f})


# ── Cascade worker thread ─────────────────────────────────────────────────────

def _run_cascade(job_id: str, pdf_path: str, q: Queue, use_gpu: bool = False,
                 threshold: float = DEFAULT_THRESHOLD):
    """Run cascade_ocr with a progress callback, putting SSE event dicts onto q."""
    try:
        from cascade import cascade_ocr, LADDER

        # Peek at page count before we start
        try:
            import fitz
            doc = fitz.open(pdf_path)
            n_pages = len(doc)
            doc.close()
        except Exception:
            n_pages = 1

        q.put({
            "event": "start", "mode": "cascade",
            "engines": LADDER, "pages": n_pages,
            "threshold": threshold,
        })

        def callback(evt: dict):
            q.put(evt)

        result = cascade_ocr(
            pdf_path,
            threshold=threshold,
            apply_preprocess=False,
            handle_redactions=True,
            dpi=300,
            verbose=False,
            progress_callback=callback,
            use_gpu=use_gpu,
        )

        _jobs[job_id]["result"] = result
        _jobs[job_id]["mean_conf"] = round(result.mean_confidence, 4)
        _jobs[job_id]["engine"] = getattr(result, "final_engine", "cascade")
        _jobs[job_id]["pages"] = len(result.pages)
        _jobs[job_id]["text"] = result.text

    except Exception as exc:
        _jobs[job_id]["error"] = str(exc)
        q.put({"event": "error", "message": str(exc)})
    finally:
        _jobs[job_id]["done"] = True
        q.put(_SENTINEL)
        _persist_completed_job(job_id, pdf_path)
        # NOTE: PDF is intentionally kept for the side-by-side viewer's
        # /source/{id} endpoint. The local-only server cleans up on restart.


def _run_regions(job_id: str, pdf_path: str, q: Queue, use_gpu: bool = False,
                 threshold: float = DEFAULT_THRESHOLD):
    """Run region_cascade_ocr with a progress callback."""
    try:
        from region_cascade import region_cascade_ocr

        try:
            import fitz
            doc = fitz.open(pdf_path)
            n_pages = len(doc)
            doc.close()
        except Exception:
            n_pages = 1

        q.put({
            "event": "start", "mode": "regions",
            "pages": n_pages, "threshold": threshold,
        })

        def callback(evt: dict):
            q.put(evt)

        result = region_cascade_ocr(
            pdf_path,
            threshold=threshold,
            apply_preprocess=False,
            handle_redactions=True,
            dpi=300,
            verbose=False,
            progress_callback=callback,
            use_gpu=use_gpu,
        )
        _jobs[job_id]["result"] = result
        _jobs[job_id]["mean_conf"] = round(result.mean_confidence, 4)
        _jobs[job_id]["pages"] = len(result.pages)
        _jobs[job_id]["text"] = result.text
        # engine breakdown (regions mode uses many engines per page)
        from collections import Counter
        engine_use: Counter = Counter()
        for p in result.pages:
            for r in p.regions:
                if r.engine:
                    engine_use[r.engine] += 1
        _jobs[job_id]["engine"] = "regions"
        _jobs[job_id]["engine_use"] = dict(engine_use)
        # Render the docx now so the side-by-side viewer can serve it
        # immediately without re-running per request.
        try:
            _jobs[job_id]["docx"] = _result_to_docx_bytes(result)
            print(f"[regions] job {job_id} docx bytes={len(_jobs[job_id]['docx'] or b'')}",
                  file=sys.stderr, flush=True)
        except Exception as docx_exc:
            _jobs[job_id]["docx"] = None
            import traceback
            print(f"[regions] job {job_id} docx render FAILED: {docx_exc}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    except Exception as exc:
        _jobs[job_id]["error"] = str(exc)
        q.put({"event": "error", "message": str(exc)})
        import traceback
        print(f"[regions] job {job_id} pipeline FAILED: {exc}",
              file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        _jobs[job_id]["done"] = True
        q.put(_SENTINEL)
        _persist_completed_job(job_id, pdf_path)
        print(f"[regions] job {job_id} finalized "
              f"(done={_jobs[job_id]['done']}, "
              f"docx={_jobs[job_id].get('docx') is not None}, "
              f"disk={(RUNS_DIR / job_id / 'result.docx').exists()})",
              file=sys.stderr, flush=True)
        # PDF is kept for /source/{id}; see _run_cascade for rationale.


def _persist_completed_job(job_id: str, pdf_path: str) -> None:
    """Write source.pdf + result.docx + meta.json into runs/{job_id}/ so the
    viewer keeps working after the server restarts. Called from the worker
    `finally` block, must never raise."""
    try:
        job = _jobs.get(job_id) or {}
        if job.get("error") or not job.get("done"):
            return
        pdf_bytes = b""
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        except OSError:
            pass
        meta = {
            "name": job.get("name", ""),
            "mode": job.get("mode", "regions"),
            "threshold": job.get("threshold", DEFAULT_THRESHOLD),
            "mean_conf": job.get("mean_conf", 0.0),
            "engine": job.get("engine", ""),
            "engine_use": job.get("engine_use", {}),
            "pages": job.get("pages", 0),
            "completed_at": time.time(),
        }
        _persist_job(
            job_id,
            meta=meta,
            pdf_bytes=pdf_bytes,
            text=job.get("text", ""),
            docx_bytes=job.get("docx"),
        )
    except Exception:
        pass


# ── docx rendering helper ─────────────────────────────────────────────────────

def _result_to_docx_bytes(result) -> bytes:
    """Render a `RegionCascadeResult` to .docx bytes (in-memory).

    Mirrors the per-page assembly that ocr_pipeline.run_regions_mode does:
    annotation anchor_y_relative is computed against the actual body text
    extent (union of OCR line bboxes), and per-line struck_segments are
    passed through so strikethrough renders correctly."""
    from docx_writer import write_docx

    annotations_per_page = []
    for p in result.pages:
        page_h = p.page_size_px[1] if p.page_size_px else 1
        body_lines = [ln for r in p.regions for ln in (r.lines or [])]
        if body_lines:
            body_y0 = min(ln["bbox"][1] for ln in body_lines)
            body_y1 = max(ln["bbox"][3] for ln in body_lines)
        else:
            body_y0, body_y1 = (p.body_y_range or (0, page_h))
        body_span = max(1, body_y1 - body_y0)

        page_anns = []
        for a in p.annotations:
            target_y = a.anchor_xy[1] if a.anchor_xy is not None else a.y_center
            y_rel = (target_y - body_y0) / body_span
            page_anns.append({
                "text": a.text,
                "anchor_y_relative": y_rel,
                "author": a.author,
                "initials": a.initials,
            })
        annotations_per_page.append(page_anns)

    line_spans_per_page = []
    for p in result.pages:
        spans = []
        for r in p.regions:
            spans.extend(r.lines or [])
        line_spans_per_page.append(spans)

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        out_path = tmp.name
    try:
        write_docx(
            out_path,
            [p.text for p in result.pages],
            annotations=annotations_per_page,
            line_spans_per_page=line_spans_per_page,
        )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


# ── Batch worker thread ───────────────────────────────────────────────────────

def _run_batch(batch_id: str):
    """Process every PDF in the batch sequentially. Each file gets its own
    SSE events (passed through with `file_index` + `file_name`) and a
    `_jobs[job_id]`-shaped result entry so the per-file download endpoints
    can reuse the same code path."""
    batch = _batches[batch_id]
    q: Queue = batch["queue"]
    files: list = batch["files"]  # [(file_name, tmp_pdf_path), ...]
    use_gpu: bool = batch["use_gpu"]
    mode: str = batch.get("mode", "regions")
    threshold: float = batch.get("threshold", DEFAULT_THRESHOLD)

    q.put({
        "event": "batch_start",
        "batch_id": batch_id,
        "total_files": len(files),
        "files": [name for name, _ in files],
        "mode": mode,
        "threshold": threshold,
    })

    for idx, (file_name, pdf_path) in enumerate(files):
        # Per-file queue so the existing _run_regions worker shape is reusable
        # via a callback wrapper that injects file_index + file_name into events.
        batch["current_index"] = idx
        q.put({
            "event": "file_start",
            "batch_id": batch_id,
            "file_index": idx,
            "file_name": file_name,
        })
        file_state = batch["files_state"][idx]
        file_state["status"] = "running"

        try:
            if mode == "regions":
                from region_cascade import region_cascade_ocr
                try:
                    import fitz
                    doc = fitz.open(pdf_path)
                    n_pages = len(doc)
                    doc.close()
                except Exception:
                    n_pages = 1
                q.put({
                    "event": "start", "mode": "regions",
                    "pages": n_pages,
                    "file_index": idx, "file_name": file_name,
                })

                def callback(evt: dict, _idx=idx, _name=file_name):
                    evt = dict(evt)
                    evt["file_index"] = _idx
                    evt["file_name"] = _name
                    q.put(evt)

                result = region_cascade_ocr(
                    pdf_path,
                    threshold=threshold,
                    apply_preprocess=False,
                    handle_redactions=True,
                    dpi=300,
                    verbose=False,
                    progress_callback=callback,
                    use_gpu=use_gpu,
                )
                file_state["text"] = result.text
                file_state["mean_conf"] = round(result.mean_confidence, 4)
                file_state["docx"] = _result_to_docx_bytes(result)
                file_state["status"] = "done"
                q.put({
                    "event": "file_done",
                    "batch_id": batch_id,
                    "file_index": idx,
                    "file_name": file_name,
                    "mean_conf": file_state["mean_conf"],
                })
            else:
                # cascade mode — no docx output in the existing cascade path
                from cascade import cascade_ocr
                try:
                    import fitz
                    doc = fitz.open(pdf_path)
                    n_pages = len(doc)
                    doc.close()
                except Exception:
                    n_pages = 1
                q.put({
                    "event": "start", "mode": "cascade",
                    "pages": n_pages,
                    "file_index": idx, "file_name": file_name,
                })

                def callback(evt: dict, _idx=idx, _name=file_name):
                    evt = dict(evt)
                    evt["file_index"] = _idx
                    evt["file_name"] = _name
                    q.put(evt)

                result = cascade_ocr(
                    pdf_path,
                    threshold=threshold,
                    apply_preprocess=False,
                    handle_redactions=True,
                    dpi=300,
                    verbose=False,
                    progress_callback=callback,
                    use_gpu=use_gpu,
                )
                file_state["text"] = result.text
                file_state["mean_conf"] = round(result.mean_confidence, 4)
                file_state["docx"] = None
                file_state["status"] = "done"
                q.put({
                    "event": "file_done",
                    "batch_id": batch_id,
                    "file_index": idx,
                    "file_name": file_name,
                    "mean_conf": file_state["mean_conf"],
                })
        except Exception as exc:
            file_state["status"] = "error"
            file_state["error"] = str(exc)
            q.put({
                "event": "file_error",
                "batch_id": batch_id,
                "file_index": idx,
                "file_name": file_name,
                "message": str(exc),
            })
        finally:
            try:
                os.unlink(pdf_path)
            except OSError:
                pass

    q.put({"event": "batch_done", "batch_id": batch_id})
    batch["done"] = True
    q.put(_SENTINEL)


# ── POST /batch ───────────────────────────────────────────────────────────────

@app.post("/batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    use_gpu: str = Form("false"),
    mode: str = Form("regions"),
    threshold: str = Form(str(DEFAULT_THRESHOLD)),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files in upload.")
    pdfs: list = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted (got: {f.filename}).",
            )
        contents = await f.read()
        if not contents:
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(contents)
            tmp.flush()
            pdfs.append((f.filename, tmp.name))
        finally:
            tmp.close()
    if not pdfs:
        raise HTTPException(status_code=400, detail="No usable PDF files.")

    gpu = use_gpu.lower() in ("true", "1", "on", "yes")
    mode = mode.lower().strip()
    if mode not in ("cascade", "regions"):
        mode = "regions"
    threshold_f = _clamp_threshold(threshold)

    batch_id = str(uuid.uuid4())
    _batches[batch_id] = {
        "queue": Queue(),
        "files": pdfs,
        "files_state": [
            {"name": name, "status": "queued", "text": "",
             "mean_conf": 0.0, "docx": None, "error": None}
            for name, _ in pdfs
        ],
        "use_gpu": gpu,
        "mode": mode,
        "threshold": threshold_f,
        "done": False,
        "current_index": -1,
    }
    threading.Thread(target=_run_batch, args=(batch_id,), daemon=True).start()
    return JSONResponse({
        "batch_id": batch_id,
        "files": [name for name, _ in pdfs],
        "mode": mode,
        "threshold": threshold_f,
    })


# ── GET /batch/{id}/stream ────────────────────────────────────────────────────

@app.get("/batch/{batch_id}/stream")
async def batch_stream(batch_id: str):
    if batch_id not in _batches:
        raise HTTPException(status_code=404, detail="Batch not found.")

    async def event_generator():
        q: Queue = _batches[batch_id]["queue"]
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=0.1))
            except Empty:
                yield ": keep-alive\n\n"
                continue
            if item is _SENTINEL:
                break
            yield f"data: {json.dumps(item)}\n\n"
        # drain
        while True:
            try:
                item = q.get_nowait()
                if item is _SENTINEL:
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except Empty:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── GET /batch/{id}/file/{idx}.{ext} ──────────────────────────────────────────

def _file_state_or_404(batch_id: str, idx: int) -> dict:
    if batch_id not in _batches:
        raise HTTPException(status_code=404, detail="Batch not found.")
    states = _batches[batch_id]["files_state"]
    if idx < 0 or idx >= len(states):
        raise HTTPException(status_code=404, detail="File index out of range.")
    fs = states[idx]
    if fs["status"] != "done":
        raise HTTPException(status_code=409, detail=f"File not ready ({fs['status']}).")
    return fs


def _docx_filename(original: str) -> str:
    stem = Path(original).stem if original else "result"
    return f"{stem}.docx"


@app.get("/batch/{batch_id}/file/{idx}.txt")
async def batch_file_txt(batch_id: str, idx: int):
    fs = _file_state_or_404(batch_id, idx)
    return Response(
        content=fs["text"],
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{Path(fs["name"]).stem}.txt"',
        },
    )


@app.get("/batch/{batch_id}/file/{idx}.docx")
async def batch_file_docx(batch_id: str, idx: int):
    fs = _file_state_or_404(batch_id, idx)
    if not fs.get("docx"):
        raise HTTPException(status_code=404, detail="No .docx for this file (cascade mode).")
    return Response(
        content=fs["docx"],
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{_docx_filename(fs["name"])}"',
        },
    )


# ── GET /batch/{id}/all.zip ───────────────────────────────────────────────────

@app.get("/batch/{batch_id}/all.zip")
async def batch_all_zip(batch_id: str):
    if batch_id not in _batches:
        raise HTTPException(status_code=404, detail="Batch not found.")
    states = _batches[batch_id]["files_state"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fs in states:
            if fs["status"] != "done":
                continue
            stem = Path(fs["name"]).stem
            zf.writestr(f"{stem}.txt", fs["text"] or "")
            if fs.get("docx"):
                zf.writestr(f"{stem}.docx", fs["docx"])
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="batch_{batch_id[:8]}.zip"',
        },
    )


# ── GET /stream/{job_id} ──────────────────────────────────────────────────────

@app.get("/stream/{job_id}")
async def stream_events(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_generator():
        job = _jobs[job_id]
        q: Queue = job["queue"]
        loop = asyncio.get_event_loop()

        while True:
            # Poll the queue without blocking the event loop
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=0.1))
            except Empty:
                # Nothing yet — keep the connection alive with a comment
                yield ": keep-alive\n\n"
                continue

            if item is _SENTINEL:
                break

            payload = json.dumps(item)
            yield f"data: {payload}\n\n"

        # Drain anything remaining
        while True:
            try:
                item = q.get_nowait()
                if item is _SENTINEL:
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except Empty:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── GET /result/{job_id}.docx — rendered Word output ──────────────────────────
# NOTE: This MUST be registered before `/result/{job_id}` (the JSON endpoint).
# FastAPI/Starlette match routes in registration order, and a path parameter
# happily eats trailing dots — so without this ordering, `GET /result/X.docx`
# was routed to `get_result` with `job_id = "X.docx"` and 404'd.

@app.api_route("/result/{job_id}.docx", methods=["GET", "HEAD"])
async def get_result_docx(job_id: str):
    job = _get_job(job_id)
    if job is None:
        print(f"[/result/{job_id}.docx] 404 — no memory or disk entry. "
              f"in_jobs={list(_jobs.keys())[:5]}... "
              f"disk={[p.name for p in RUNS_DIR.iterdir() if p.is_dir()][:5]}...",
              file=sys.stderr, flush=True)
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.get("done"):
        raise HTTPException(status_code=202, detail="Job still in progress.")
    docx = job.get("docx")
    if not docx:
        persisted = RUNS_DIR / job_id / "result.docx"
        if persisted.exists():
            docx = persisted.read_bytes()
    if not docx:
        print(f"[/result/{job_id}.docx] 404 — job exists "
              f"(mode={job.get('mode')}, done={job.get('done')}) but no docx",
              file=sys.stderr, flush=True)
        raise HTTPException(
            status_code=404,
            detail="No .docx for this job (cascade mode does not produce docx).",
        )
    return Response(
        content=docx,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'inline; filename="result.docx"'},
    )


# ── GET /result/{job_id} ─────────────────────────────────────────────────────
# Parameter is constrained to a path component without dots so a request for
# `/result/X.docx` can never be matched here. (Belt + suspenders with the
# ordering above — if someone refactors this, the routing still holds.)

@app.get("/result/{job_id:path}", include_in_schema=False)
async def get_result(job_id: str):
    # Reject anything with a dot in it — those are extension-suffixed routes
    # (e.g. .docx) which have their own handlers above.
    if "." in job_id:
        raise HTTPException(status_code=404, detail="Not Found")
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.get("done"):
        raise HTTPException(status_code=202, detail="Job still in progress.")
    if job.get("error"):
        raise HTTPException(status_code=500, detail=job["error"])

    mode = job.get("mode", "cascade")
    # Prefer cached/persisted fields. Fall back to deriving from a live
    # result object when present (in-memory immediately-after-completion).
    payload = {
        "mode": mode,
        "text": job.get("text", "") or "",
        "mean_conf": job.get("mean_conf", 0.0),
        "pages": job.get("pages", 0),
        "engine": job.get("engine", "regions" if mode == "regions" else "cascade"),
    }
    if mode == "regions":
        payload["engine_use"] = job.get("engine_use", {})

    result = job.get("result")
    if result is not None and not payload["text"]:
        # Worker just finished; cached fields aren't populated yet.
        payload["text"] = result.text
        payload["mean_conf"] = round(result.mean_confidence, 4)
        payload["pages"] = len(result.pages)
        if mode == "cascade":
            payload["engine"] = result.final_engine
        else:
            from collections import Counter
            eu = Counter()
            for p in result.pages:
                for r in p.regions:
                    if r.engine:
                        eu[r.engine] += 1
            payload["engine_use"] = dict(eu)
            payload["region_count"] = sum(len(p.regions) for p in result.pages)

    return JSONResponse(payload)


# ── GET /source/{job_id} — original uploaded PDF ──────────────────────────────

@app.api_route("/source/{job_id}", methods=["GET", "HEAD"])
async def get_source(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    pdf_path = job.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        # Try the persisted copy under runs/.
        persisted = RUNS_DIR / job_id / "source.pdf"
        if persisted.exists():
            pdf_path = str(persisted)
        else:
            raise HTTPException(status_code=404, detail="Source PDF no longer available.")
    with open(pdf_path, "rb") as f:
        data = f.read()
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="source.pdf"'},
    )


# ── GET /jobs — docket index for the UI ──────────────────────────────────────

@app.get("/jobs")
async def list_jobs():
    """Union of in-memory + persisted jobs. The UI uses this to rebuild the
    docket sidebar after a page reload or a server restart."""
    out = []
    seen: set = set()

    for jid, j in _jobs.items():
        seen.add(jid)
        if j.get("error"):
            status = "error"
        elif j.get("done"):
            status = "done"
        else:
            status = "running"
        out.append({
            "id": jid,
            "name": j.get("name", ""),
            "mode": j.get("mode", "regions"),
            "status": status,
            "mean_conf": j.get("mean_conf", 0.0),
            "started_at": j.get("started_at"),
            "kind": "single",
        })

    # Add anything persisted but not in memory (server-restart leftovers).
    if RUNS_DIR.exists():
        for sub in RUNS_DIR.iterdir():
            if not sub.is_dir() or sub.name in seen:
                continue
            meta_path = sub / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "id": sub.name,
                "name": meta.get("name", ""),
                "mode": meta.get("mode", "regions"),
                "status": "done",
                "mean_conf": meta.get("mean_conf", 0.0),
                "started_at": meta.get("completed_at"),
                "kind": "single",
            })

    # Most recent first.
    out.sort(key=lambda e: e.get("started_at") or 0, reverse=True)
    return JSONResponse({"jobs": out})


# ── DELETE /jobs/{job_id} — strike a filing from the record ──────────────────

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Remove a job from memory and delete its runs/{id}/ directory. Returns
    {status: ok} regardless of whether anything was deleted (idempotent)."""
    import shutil
    removed = {"memory": False, "disk": False}

    if job_id in _jobs:
        # Best-effort: drain the SSE queue + delete the temp PDF the worker
        # is still holding. The worker thread is daemon-only, we don't try
        # to join it — it will harmlessly write to a no-longer-watched queue.
        job = _jobs[job_id]
        pdf_path = job.get("pdf_path")
        if pdf_path and os.path.exists(pdf_path):
            # Don't remove if it's the persisted source.pdf — that's owned by
            # the runs/ dir which the shutil.rmtree below handles.
            if not pdf_path.startswith(str(RUNS_DIR)):
                try:
                    os.unlink(pdf_path)
                except OSError:
                    pass
        del _jobs[job_id]
        removed["memory"] = True

    run_dir = RUNS_DIR / job_id
    if run_dir.exists() and run_dir.is_dir():
        try:
            shutil.rmtree(run_dir)
            removed["disk"] = True
        except OSError:
            pass

    return JSONResponse({"status": "ok", "removed": removed})


# ── /config/regions ───────────────────────────────────────────────────────────

# Engines that the cascade can route to. Kept in display order: cheapest →
# heaviest. The UI's "add engine" dropdown is built from this list.
_AVAILABLE_ENGINES = [
    "tesseract",
    "paddleocr",
    "merge-tess-paddle",
    "docling",
    "glm-ocr",
    "chandra",
]


def _yaml_path() -> Path:
    from region_config import DEFAULT_CONFIG_PATH
    return Path(DEFAULT_CONFIG_PATH)


@app.get("/config/regions")
async def get_region_config():
    """Return the current region_config.yaml as a structured JSON payload
    plus the list of engines the UI may pick from."""
    import yaml
    path = _yaml_path()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    labels = []
    for label, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        labels.append({
            "label": label,
            "ladder": list(entry.get("ladder") or []),
            "rationale": (entry.get("rationale") or "").strip(),
            "citation": (entry.get("citation") or "").strip(),
            "is_default": label == "_default",
        })

    return JSONResponse({
        "labels": labels,
        "available_engines": _AVAILABLE_ENGINES,
    })


@app.put("/config/regions")
async def put_region_config(payload: dict):
    """Accept {"labels": [{label, ladder}, ...]} and write the new ladders
    back to region_config.yaml. Preserves the rationale + citation comments
    for each label by merging into the existing YAML structure rather than
    rewriting from scratch."""
    import yaml
    new_labels = payload.get("labels")
    if not isinstance(new_labels, list):
        raise HTTPException(status_code=400, detail="Expected `labels` list.")

    by_label = {}
    for item in new_labels:
        if not isinstance(item, dict):
            continue
        lbl = item.get("label")
        ladder = item.get("ladder")
        if not lbl or not isinstance(ladder, list):
            continue
        ladder = [e for e in ladder if isinstance(e, str) and e in _AVAILABLE_ENGINES]
        by_label[str(lbl)] = ladder

    path = _yaml_path()
    with open(path) as f:
        existing = yaml.safe_load(f) or {}

    for lbl, ladder in by_label.items():
        if lbl in existing and isinstance(existing[lbl], dict):
            existing[lbl]["ladder"] = ladder
        else:
            existing[lbl] = {"ladder": ladder}

    with open(path, "w") as f:
        # sort_keys=False preserves the label ordering the user (and the
        # rationales) expect.
        yaml.safe_dump(existing, f, sort_keys=False, default_flow_style=False)

    return JSONResponse({"status": "ok", "labels_written": len(by_label)})


# ── GET / ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))
