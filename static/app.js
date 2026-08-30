/**
 * OCR Cascade — modern UI.
 *
 * App states are driven by the current view: upload → progress | batch → viewer.
 * SSE delivers progress events; on completion we swap to the side-by-side viewer
 * (PDF rendered via pdf.js, docx via docx-preview, scrolls synced proportionally).
 */

// ─────────────────────────────────────────────────────────────────────── pdf.js ──
// Loaded as ESM module from CDN
import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.min.mjs";
pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.worker.min.mjs";

// ─────────────────────────────────────────────────────────────────── DOM refs ──
const $ = (id) => document.getElementById(id);

const dropZone        = $("drop-zone");
const fileInput       = $("file-input");
const folderInput     = $("folder-input");
const browseBtn       = $("browse-btn");
const folderBtn       = $("folder-btn");
const fileSummary     = $("file-summary");
const submitBtn       = $("submit-btn");
const newJobBtn       = $("new-job-btn");
const thresholdSlider = $("threshold-slider");
const thresholdInput  = $("threshold-input");
const gpuCheckbox     = $("gpu-checkbox");
const modeButtons     = document.querySelectorAll('.seg-control[data-name="mode"] button');

const viewUpload   = $("view-upload");
const viewProgress = $("view-progress");
const viewBatch    = $("view-batch");
const viewViewer   = $("view-viewer");
const viewSettings = $("view-settings");
const settingsBtn  = $("settings-btn");
const settingsList = $("settings-list");
const settingsSaveBtn  = $("settings-save-btn");
const settingsResetBtn = $("settings-reset-btn");

const progressFilename = $("progress-filename");
const progressStatus   = $("progress-status");
const progressRegions  = $("progress-regions");
const progressCascade  = $("progress-cascade");
const progressRegionCount = $("progress-region-count");

const redactionBanner  = $("redaction-banner");
const batchRedactionBanner = $("batch-redaction-banner");

const regionList     = $("region-list");
const regionMeterFill = $("region-meter-fill");
const regionMeterPct  = $("region-meter-pct");

const engineLadder = $("engine-ladder");
const meterFill    = $("meter-fill");
const meterPct     = $("meter-pct");
const escalateMsg  = $("escalate-msg");

const batchList   = $("batch-list");
const batchZipBtn = $("batch-zip-btn");

const viewerFilename = $("viewer-filename");
const viewerConf     = $("viewer-conf");
const viewerEngines  = $("viewer-engines");
const viewerDocxBtn  = $("viewer-download-docx");
const viewerTxtBtn   = $("viewer-download-txt");
const pdfBody        = $("pdf-body");
const docxBody       = $("docx-body");
const pdfPageInfo    = $("pdf-page-info");
const docxStatus     = $("docx-status");

const jobListEl    = $("docket-list");
const docketCount  = $("docket-count");
const errorToast   = $("error-toast");
const redactionBreakdown      = $("redaction-breakdown");
const batchRedactionBreakdown = $("batch-redaction-breakdown");

// ────────────────────────────────────────────────────────────────────── state ──
let selectedFiles = [];
let currentMode = "regions";
let currentJobId = null;
let currentBatchId = null;
let currentEventSource = null;
const jobs = {}; // job_id → {kind, mode, name, status, ...}

// ─────────────────────────────────────────────────────────── theme toggle ──
const themeToggle = $("theme-toggle");
if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const next = isDark ? "light" : "dark";
    if (next === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    try { localStorage.setItem("cascade-theme", next); } catch {}
  });
}

// ─────────────────────────────────────────────────── threshold slider binding ──
function clampThreshold(v) {
  v = parseInt(v, 10);
  if (!Number.isFinite(v)) return 90;
  return Math.min(99, Math.max(50, v));
}
thresholdSlider.addEventListener("input", () => { thresholdInput.value = thresholdSlider.value; });
thresholdInput.addEventListener("change", () => {
  const v = clampThreshold(thresholdInput.value);
  thresholdInput.value = v;
  thresholdSlider.value = v;
});
const thresholdFraction = () => clampThreshold(thresholdInput.value) / 100;

// ────────────────────────────────────────────────────────── seg control (mode) ──
modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    modeButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.value;
  });
});

// ───────────────────────────────────────────────────────────── file selection ──
const actionsHint = document.getElementById("actions-hint");

function setSelectedFiles(filesLike) {
  selectedFiles = Array.from(filesLike).filter(
    (f) => f && f.name && f.name.toLowerCase().endsWith(".pdf")
  );
  if (selectedFiles.length === 0) {
    fileSummary.textContent = "";
    submitBtn.disabled = true;
    if (actionsHint) actionsHint.textContent = "Pick a document first.";
    return;
  }
  if (selectedFiles.length === 1) {
    fileSummary.textContent = selectedFiles[0].name;
    if (actionsHint) actionsHint.textContent = "Ready to file.";
  } else {
    fileSummary.textContent = `${selectedFiles.length} filings queued — they run sequentially.`;
    if (actionsHint) actionsHint.textContent = "Batch intake.";
  }
  submitBtn.disabled = false;
}

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => setSelectedFiles(fileInput.files));
folderBtn.addEventListener("click", () => folderInput.click());
folderInput.addEventListener("change", () => setSelectedFiles(folderInput.files));
dropZone.addEventListener("click", (e) => {
  // Don't double-trigger when clicking the explicit browse/folder links
  if (e.target.closest(".link-btn")) return;
  fileInput.click();
});
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  setSelectedFiles(e.dataTransfer.files);
});

// ───────────────────────────────────────────────────────────────── view switch ──
function showView(name) {
  [viewUpload, viewProgress, viewBatch, viewViewer, viewSettings].forEach((v) => v.hidden = true);
  ({
    upload: viewUpload, progress: viewProgress, batch: viewBatch,
    viewer: viewViewer, settings: viewSettings,
  })[name].hidden = false;
}

newJobBtn.addEventListener("click", () => {
  // Clear state, return to upload screen
  if (currentEventSource) { currentEventSource.close(); currentEventSource = null; }
  selectedFiles = [];
  fileInput.value = "";
  folderInput.value = "";
  fileSummary.textContent = "";
  submitBtn.disabled = true;
  showView("upload");
});

// ───────────────────────────────────────────────────────────────── submit ──
submitBtn.addEventListener("click", async () => {
  if (selectedFiles.length === 0) return;
  hideError();

  if (selectedFiles.length === 1) {
    await startSingleJob();
  } else {
    await startBatchJob();
  }
});

async function startSingleJob() {
  resetProgressUI();
  showView("progress");
  progressFilename.textContent = selectedFiles[0].name;
  progressStatus.textContent = "In progress";
  progressStatus.className = "status-pill running";

  if (currentMode === "regions") {
    progressRegions.hidden = false;
    progressCascade.hidden = true;
  } else {
    progressRegions.hidden = true;
    progressCascade.hidden = false;
  }

  const fd = new FormData();
  fd.append("file", selectedFiles[0]);
  fd.append("use_gpu", gpuCheckbox.checked ? "true" : "false");
  fd.append("mode", currentMode);
  fd.append("threshold", String(thresholdFraction()));

  let jobId;
  try {
    const res = await fetch("/ocr", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      showError(err.detail || "Upload failed");
      showView("upload");
      return;
    }
    const data = await res.json();
    jobId = data.job_id;
  } catch (e) {
    showError("Network error: " + e.message);
    showView("upload");
    return;
  }

  currentJobId = jobId;
  jobs[jobId] = {
    kind: "single",
    mode: currentMode,
    name: selectedFiles[0].name,
    status: "running",
  };
  renderJobList();

  listenToStream(jobId);
}

async function startBatchJob() {
  resetProgressUI();
  showView("batch");
  renderBatchList(selectedFiles.map((f) => ({ name: f.name, status: "queued" })));
  batchZipBtn.hidden = true;

  const fd = new FormData();
  selectedFiles.forEach((f) => fd.append("files", f));
  fd.append("use_gpu", gpuCheckbox.checked ? "true" : "false");
  fd.append("mode", currentMode);
  fd.append("threshold", String(thresholdFraction()));

  let batchId;
  try {
    const res = await fetch("/batch", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      showError(err.detail || "Upload failed");
      showView("upload");
      return;
    }
    const data = await res.json();
    batchId = data.batch_id;
  } catch (e) {
    showError("Network error: " + e.message);
    showView("upload");
    return;
  }
  currentBatchId = batchId;
  jobs[batchId] = {
    kind: "batch",
    mode: currentMode,
    name: `Batch (${selectedFiles.length})`,
    status: "running",
  };
  renderJobList();
  listenToBatchStream(batchId);
}

// ───────────────────────────────────────────────────────────── SSE: single ──
function listenToStream(jobId) {
  if (currentEventSource) currentEventSource.close();
  currentEventSource = new EventSource(`/stream/${jobId}`);
  currentEventSource.onmessage = (e) => {
    try { handleEvent(JSON.parse(e.data)); } catch {}
  };
  currentEventSource.onerror = () => { /* silently close on normal end */ };
}

const engineSteps = {};
let pageConfs = [];
let regionPageConfs = [];
const regionRows = {};
const redactionPages = {};

function handleEvent(evt) {
  switch (evt.event) {
    case "start":
      currentMode = evt.mode || "cascade";
      if (currentMode === "cascade" && Array.isArray(evt.engines)) buildLadder(evt.engines);
      break;

    case "redactions_marked":
      redactionPages[evt.page] = evt.count;
      renderRedactionBanner(redactionBanner);
      break;

    // ── cascade events ──
    case "engine_start":
      activateStep(evt.engine);
      pageConfs = [];
      setMeter(meterFill, meterPct, 0);
      break;
    case "page_done":
      if (currentMode === "regions") {
        regionPageConfs.push(evt.page_conf);
        const m = regionPageConfs.reduce((a, b) => a + b, 0) / regionPageConfs.length;
        setMeter(regionMeterFill, regionMeterPct, m);
      } else {
        pageConfs.push(evt.confidence);
        const m = pageConfs.reduce((a, b) => a + b, 0) / pageConfs.length;
        setMeter(meterFill, meterPct, m);
      }
      break;
    case "engine_done":
      completeStep(evt.engine, evt.mean_conf, evt.passed);
      break;
    case "escalate":
      showEscalate(evt.from, evt.to, evt.reason);
      break;

    // ── regions events ──
    case "region_detected":
      (evt.regions || []).forEach((r, idx) => ensureRegionRow(evt.page, idx, r.label, r.bbox));
      progressRegionCount.textContent = `${Object.keys(regionRows).length} regions detected`;
      break;
    case "region_engine_start":
      markRegionActive(evt.page, evt.region_index, evt.engine);
      break;
    case "region_done":
      updateRegionRow(evt.page, evt.region_index, evt.engine, evt.ocr_conf, evt.passed);
      break;
    case "region_escalate":
      showEscalate(evt.from, evt.to, evt.reason);
      break;

    case "complete":
      if (currentMode === "regions") setMeter(regionMeterFill, regionMeterPct, evt.mean_conf);
      else setMeter(meterFill, meterPct, evt.mean_conf);
      currentEventSource && currentEventSource.close();
      onSingleJobDone();
      break;

    case "error":
      showError(evt.message);
      progressStatus.textContent = "Error";
      progressStatus.className = "status-pill error";
      currentEventSource && currentEventSource.close();
      jobs[currentJobId].status = "error";
      renderJobList();
      break;
  }
}

async function onSingleJobDone() {
  progressStatus.textContent = "On record";
  progressStatus.className = "status-pill done";
  jobs[currentJobId].status = "done";
  renderJobList();

  // Pull metadata so we can show engine + conf in the viewer header.
  // Use the same poll-on-202 helper as the docx fetch since the worker
  // finalizes after region_cascade_ocr's "complete" event.
  try {
    const res = await _fetchWhenReady(`/result/${currentJobId}`);
    if (res.ok) {
      const data = await res.json();
      jobs[currentJobId].result = data;
    }
  } catch {}

  openViewer(currentJobId);
}

// ─────────────────────────────────────────────────────────── SSE: batch ──
const batchRowEls = [];

function renderBatchList(initial) {
  batchList.innerHTML = "";
  batchRowEls.length = 0;
  initial.forEach((f) => {
    const row = document.createElement("li");
    row.className = "batch-row";
    row.innerHTML = `
      <span class="name" title="${escapeHTML(f.name)}">${escapeHTML(f.name)}</span>
      <span class="badge queued">queued</span>
      <span class="conf">—</span>
      <span class="dl"></span>
    `;
    batchList.appendChild(row);
    batchRowEls.push(row);
  });
}

function setBatchRowStatus(idx, status, conf) {
  const row = batchRowEls[idx];
  if (!row) return;
  row.classList.remove("running", "done", "error");
  row.classList.add(status);
  const badge = row.querySelector(".badge");
  badge.className = "badge " + status;
  badge.textContent = status;
  if (typeof conf === "number") row.querySelector(".conf").textContent = (conf * 100).toFixed(1) + "%";
}

function attachBatchRowDownloads(idx, batchId) {
  const row = batchRowEls[idx];
  if (!row) return;
  row.querySelector(".dl").innerHTML = `
    <a href="/batch/${batchId}/file/${idx}.txt" download>.txt</a>
    <a href="/batch/${batchId}/file/${idx}.docx" download>.docx</a>
  `;
}

function listenToBatchStream(batchId) {
  if (currentEventSource) currentEventSource.close();
  currentEventSource = new EventSource(`/batch/${batchId}/stream`);
  currentEventSource.onmessage = (e) => {
    try { handleBatchEvent(JSON.parse(e.data)); } catch {}
  };
  currentEventSource.onerror = () => {};
}

const batchRedactionPages = {};

function handleBatchEvent(evt) {
  switch (evt.event) {
    case "file_start":
      setBatchRowStatus(evt.file_index, "running");
      break;
    case "file_done":
      setBatchRowStatus(evt.file_index, "done", evt.mean_conf);
      attachBatchRowDownloads(evt.file_index, currentBatchId);
      break;
    case "file_error":
      setBatchRowStatus(evt.file_index, "error");
      break;
    case "redactions_marked":
      batchRedactionPages[`${evt.file_index || 0}-${evt.page}`] = evt.count;
      renderRedactionBanner(batchRedactionBanner, batchRedactionPages);
      break;
    case "batch_done":
      batchZipBtn.hidden = false;
      batchZipBtn.onclick = () => { window.location.href = `/batch/${currentBatchId}/all.zip`; };
      jobs[currentBatchId].status = "done";
      renderJobList();
      currentEventSource && currentEventSource.close();
      break;
  }
}

// ───────────────────────────────────────────────────────── redaction banner ──
function renderRedactionBanner(target, pages) {
  pages = pages || redactionPages;
  const total = Object.values(pages).reduce((a, b) => a + b, 0);
  const lines = Object.entries(pages)
    .map(([k, v]) => `${k.includes("-") ? "filing " + (parseInt(k.split("-")[0], 10) + 1) + ", p" + (parseInt(k.split("-")[1], 10) + 1) : "p" + (parseInt(k, 10) + 1)}: ${v}`)
    .join(" · ");
  // The new HTML pre-renders the .redaction-mark + .redaction-text structure;
  // we only update the breakdown line and the head count via .redaction-head.
  const head = target.querySelector(".redaction-head");
  const breakdown = target.querySelector(".redaction-breakdown");
  if (head) head.textContent = `${total} redaction${total === 1 ? "" : "s"} noted`;
  if (breakdown) breakdown.textContent = lines;
  target.hidden = false;
}

// ───────────────────────────────────────────────────────────── viewer setup ──
async function openViewer(jobId) {
  const info = jobs[jobId];
  // Track which job the viewer is showing so strikeJob can detect it and
  // close out the viewer if that job gets deleted. Without this update,
  // currentJobId stays pointing at whatever was last submitted, which made
  // it possible to strike the job the viewer was actually displaying
  // without the viewer noticing.
  currentJobId = jobId;
  showView("viewer");
  viewerFilename.textContent = info?.name || "Result";

  if (info?.result) {
    viewerConf.textContent = `${(info.result.mean_conf * 100).toFixed(1)}% confidence`;
    if (info.result.engine_use) {
      const list = Object.entries(info.result.engine_use)
        .map(([e, n]) => `${e}×${n}`).join(" · ");
      viewerEngines.textContent = list;
    } else if (info.result.engine) {
      viewerEngines.textContent = info.result.engine;
    }
  } else {
    viewerConf.textContent = "—";
    viewerEngines.textContent = "—";
  }

  viewerDocxBtn.hidden = !(info?.mode === "regions");
  viewerTxtBtn.hidden = false;
  viewerDocxBtn.onclick = () => { window.location.href = `/result/${jobId}.docx`; };
  viewerTxtBtn.onclick = async () => {
    // For single-file we don't have a .txt endpoint; trigger a client-side
    // download using the JSON text payload.
    try {
      const res = await fetch(`/result/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();
      const blob = new Blob([data.text || ""], { type: "text/plain" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = (info?.name || "result").replace(/\.pdf$/i, "") + ".txt";
      a.click();
      URL.revokeObjectURL(a.href);
    } catch {}
  };

  showPaneLoading(pdfBody);
  showPaneLoading(docxBody);

  // Render in parallel.
  Promise.all([renderPDFPane(jobId), renderDocxPane(jobId, info?.mode)]).then(() => {
    setupScrollSync();
  });
}

function showPaneLoading(el) {
  el.innerHTML = `<div class="pane-loading"><div class="spinner"></div>Rendering…</div>`;
}

/**
 * Scale each `section.docx` in the docx pane so the page width matches the
 * pane width. Uses CSS `zoom` (supported in all current browsers) so the
 * layout reflows correctly — `transform: scale()` only repaints, leaving
 * extra empty space below the scaled element.
 */
function _fitDocxPagesToPane() {
  const sections = docxBody.querySelectorAll("section.docx");
  if (!sections.length) return;
  // Subtract horizontal padding so the page sits cleanly inside the pane.
  const paneWidth = docxBody.clientWidth - 44;
  sections.forEach((section) => {
    section.style.zoom = "";  // reset so we measure the natural width
    const naturalWidth = section.offsetWidth;
    if (naturalWidth > paneWidth && naturalWidth > 0) {
      section.style.zoom = (paneWidth / naturalWidth).toFixed(4);
    }
  });
}

// Re-fit when the window resizes (debounced) so the docx page tracks the
// viewer split as it changes.
let _fitTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(_fitTimer);
  _fitTimer = setTimeout(_fitDocxPagesToPane, 120);
});

/**
 * Render a "this filing isn't on the server" notice in both viewer panes
 * and provide a button to drop the stale entry from the in-page docket.
 * Fired when /source/{id} or /result/{id}.docx returns 404 "Job not found".
 */
function _showStaleJobMessage(jobId) {
  const html = `
    <div class="pane-loading" style="text-align:center; max-width:340px;">
      <div style="margin-bottom:10px;">
        <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="9" y1="15" x2="15" y2="15"/>
        </svg>
      </div>
      <div>This filing isn't on the current server.</div>
      <div class="muted small" style="margin-top:6px;">It may be from an earlier session, or it was struck from the record.</div>
      <button class="lex-btn small" style="margin-top:14px;" id="stale-clear-btn">Strike from docket</button>
    </div>`;
  pdfBody.innerHTML = html;
  docxBody.innerHTML = html;
  pdfPageInfo.textContent = "—";
  docxStatus.textContent = "not on server";
  // The button shows up in both panes; either click does the same thing.
  document.querySelectorAll("#stale-clear-btn").forEach((b) => {
    b.addEventListener("click", () => {
      delete jobs[jobId];
      renderJobList();
      if (currentEventSource) { currentEventSource.close(); currentEventSource = null; }
      currentJobId = null;
      showView("upload");
    });
  });
}

/**
 * Poll an endpoint until it's no longer 202 (job still in progress) or
 * we exceed totalMs. Used so the viewer can open immediately on the SSE
 * "complete" event without racing the worker that's still finalising
 * docx/persistence in its `finally` block.
 */
async function _fetchWhenReady(url, totalMs = 30000) {
  const start = Date.now();
  let delay = 250;
  while (true) {
    const res = await fetch(url);
    if (res.status !== 202) return res;
    if (Date.now() - start > totalMs) return res;
    await new Promise((r) => setTimeout(r, delay));
    delay = Math.min(delay * 1.5, 1500);
  }
}

async function renderPDFPane(jobId) {
  try {
    // Probe /source first so a missing-job 404 produces a clear message
    // rather than the generic "InvalidPDFException" pdf.js raises. Use
    // HEAD (now supported on /source) to avoid pulling the whole PDF.
    const probe = await _fetchWhenReady(`/source/${jobId}`);
    if (probe.status === 404) {
      const text = await probe.text().catch(() => "");
      if (text.includes("Job not found")) { _showStaleJobMessage(jobId); return; }
      pdfBody.innerHTML = `<div class="pane-loading">Source PDF unavailable.</div>`;
      return;
    }

    const pdf = await pdfjsLib.getDocument(`/source/${jobId}`).promise;
    pdfBody.innerHTML = "";
    pdfPageInfo.textContent = `${pdf.numPages} page${pdf.numPages === 1 ? "" : "s"}`;
    for (let p = 1; p <= pdf.numPages; p++) {
      const page = await pdf.getPage(p);
      const viewport = page.getViewport({ scale: 1.4 });
      const canvas = document.createElement("canvas");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      pdfBody.appendChild(canvas);
      await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
    }
  } catch (err) {
    pdfBody.innerHTML = `<div class="pane-loading">PDF preview unavailable.</div>`;
  }
}

async function renderDocxPane(jobId, mode) {
  if (mode !== "regions") {
    docxBody.innerHTML = `<div class="pane-loading">Cascade mode does not produce a .docx — use the .txt download.</div>`;
    docxStatus.textContent = "txt only";
    return;
  }
  // Guard against the docx-preview UMD bundle not being on `window`.
  // This used to fail silently because JSZip was loaded *after* docx-preview;
  // surface the cause now so a future regression is debuggable.
  if (!window.docx || typeof window.docx.renderAsync !== "function") {
    docxBody.innerHTML =
      `<div class="pane-loading">Docx renderer didn't load (window.docx missing). Check the script tags in index.html.</div>`;
    docxStatus.textContent = "renderer missing";
    return;
  }
  if (typeof window.JSZip !== "function") {
    docxBody.innerHTML =
      `<div class="pane-loading">JSZip didn't load. docx-preview cannot unzip the .docx.</div>`;
    docxStatus.textContent = "jszip missing";
    return;
  }
  try {
    // The "complete" SSE event fires from inside region_cascade_ocr,
    // before the worker has rendered the docx and set done=True. So a
    // straight fetch here will frequently return 202 ("still in progress")
    // for the first second or two. _fetchWhenReady polls until done.
    const res = await _fetchWhenReady(`/result/${jobId}.docx`);
    if (res.status === 404) {
      const text = await res.text().catch(() => "");
      if (text.includes("Job not found")) {
        _showStaleJobMessage(jobId);
        return;
      }
      throw new Error(`fetch 404: ${text.slice(0, 120)}`);
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`fetch ${res.status}: ${text.slice(0, 120)}`);
    }
    const blob = await res.blob();
    docxBody.innerHTML = "";
    await window.docx.renderAsync(blob, docxBody, null, {
      className: "docx-wrapper",
      // Render as a single continuous flow rather than per-page sections.
      // With breakPages=true the library emits a fixed-height <section>
      // per OOXML page; that interacts badly with our CSS zoom — a wider
      // page gets scaled but the section's own overflow can clip later
      // pages. ignoreHeight=true lets the section grow with content.
      ignoreWidth: false,
      ignoreHeight: true,
      breakPages: false,
      inWrapper: true,
      experimental: true,
    });
    // Each rendered .docx page comes out at the original page width
    // (typically 8.5" = 816 px), which overflows the narrow viewer pane.
    // Scale every rendered page-section down to fit the pane width.
    _fitDocxPagesToPane();
    docxStatus.textContent = "rendered";
  } catch (err) {
    console.error("docx render failed:", err);
    docxBody.innerHTML =
      `<div class="pane-loading">Docx preview failed.<br><span class="muted small">${escapeHTML((err && err.message) || String(err))}</span></div>`;
    docxStatus.textContent = "error";
  }
}

// ───────────────────────────────────────────────────── synced scroll ──
let suppressScroll = false;
function setupScrollSync() {
  function syncFrom(src, dst) {
    if (suppressScroll) return;
    const srcMax = src.scrollHeight - src.clientHeight;
    if (srcMax <= 0) return;
    const frac = src.scrollTop / srcMax;
    const dstMax = dst.scrollHeight - dst.clientHeight;
    suppressScroll = true;
    dst.scrollTop = frac * dstMax;
    requestAnimationFrame(() => { suppressScroll = false; });
  }
  pdfBody.addEventListener("scroll", () => syncFrom(pdfBody, docxBody));
  docxBody.addEventListener("scroll", () => syncFrom(docxBody, pdfBody));
}

// ──────────────────────────────────────────────────── ladder helpers (cascade) ──
function buildLadder(engines) {
  engineLadder.innerHTML = "";
  Object.keys(engineSteps).forEach((k) => delete engineSteps[k]);
  engines.forEach((name) => {
    const li = document.createElement("li");
    li.className = "engine-step";
    li.innerHTML = `
      <span class="step-state">○</span>
      <span class="step-name">${escapeHTML(name)}</span>
      <span class="step-conf"></span>
    `;
    engineLadder.appendChild(li);
    engineSteps[name] = li;
  });
}
function activateStep(name) {
  document.querySelectorAll(".engine-step.active").forEach((el) => el.classList.remove("active"));
  const el = engineSteps[name];
  if (el) { el.classList.add("active"); el.querySelector(".step-state").textContent = "●"; }
}
function completeStep(name, conf, passed) {
  const el = engineSteps[name];
  if (!el) return;
  el.classList.remove("active");
  el.classList.add(passed ? "passed" : "failed");
  el.querySelector(".step-state").textContent = passed ? "✓" : "×";
  el.querySelector(".step-conf").textContent = (conf * 100).toFixed(1) + "%";
}
function showEscalate(from, to, reason) {
  escalateMsg.textContent = `${from} → ${to}: ${reason || ""}`.trim();
  escalateMsg.hidden = false;
}

// ───────────────────────────────────────────── region helpers (regions mode) ──
function regionKey(p, ri) { return `p${p}-r${ri}`; }
function ensureRegionRow(p, ri, label, bbox) {
  const key = regionKey(p, ri);
  if (regionRows[key]) return;
  const row = document.createElement("div");
  row.className = "region-row";
  const tagClass = tagClassFor(label);
  row.innerHTML = `
    <span class="page">p${p + 1}</span>
    <span class="tag ${tagClass}">${escapeHTML(label)}</span>
    <span class="attempts">—</span>
    <span class="conf">—</span>
    <span class="bbox">${bbox.map(Math.round).join(", ")}</span>
  `;
  regionList.appendChild(row);
  // Track attempt history per region so the chain survives across
  // engine_start / region_done events.
  regionRows[key] = row;
  regionAttempts[key] = [];
}

// History per region — entries are {engine, conf, passed, running}.
// `running: true` means the engine is still working (no result yet).
const regionAttempts = {};

function _renderAttempts(row, history) {
  const cell = row.querySelector(".attempts");
  cell.innerHTML = "";
  history.forEach((step, i) => {
    if (i > 0) {
      const sep = document.createElement("span");
      sep.className = "attempt-sep";
      sep.textContent = "▸";
      cell.appendChild(sep);
    }
    const span = document.createElement("span");
    span.className = "attempt"
      + (step.running ? " running" : "")
      + (step.passed ? " passed" : "")
      + (!step.running && !step.passed ? " failed" : "");
    if (step.running) {
      span.innerHTML = `<span class="attempt-name">${escapeHTML(step.engine)}</span><span class="attempt-dots">…</span>`;
    } else {
      span.innerHTML = `<span class="attempt-name">${escapeHTML(step.engine)}</span> <span class="attempt-conf">${(step.conf * 100).toFixed(1)}%</span>`;
    }
    cell.appendChild(span);
  });
}

function markRegionActive(p, ri, engine) {
  const key = regionKey(p, ri);
  const row = regionRows[key];
  if (!row) return;
  row.classList.add("active");
  // If this engine already started (re-fire), don't double-push.
  const hist = regionAttempts[key];
  const last = hist[hist.length - 1];
  if (!last || last.engine !== engine || !last.running) {
    hist.push({ engine, running: true });
  }
  _renderAttempts(row, hist);
}

function updateRegionRow(p, ri, engine, conf, passed) {
  const key = regionKey(p, ri);
  const row = regionRows[key];
  if (!row) return;
  row.classList.remove("active");
  const hist = regionAttempts[key];
  // Resolve the running attempt, or append a new one if region_done
  // arrived without a preceding engine_start.
  const last = hist[hist.length - 1];
  if (last && last.running && last.engine === engine) {
    last.conf = conf;
    last.passed = !!passed;
    last.running = false;
  } else {
    hist.push({ engine, conf, passed: !!passed, running: false });
  }
  _renderAttempts(row, hist);
  // Final-conf column reflects the last completed attempt (the winning
  // engine if passed, the best-effort engine if all failed).
  const c = row.querySelector(".conf");
  c.textContent = (conf * 100).toFixed(1) + "%";
  c.className = "conf " + (passed ? "good" : "below");
}
function tagClassFor(label) {
  if (label === "handwritten_text") return "handwritten";
  if (label === "picture" || label === "chart") return "picture";
  if (label === "table") return "table";
  if (label === "_uncovered") return "uncovered";
  return "";
}

// ────────────────────────────────────────────────────────── meter helper ──
function setMeter(fillEl, pctEl, val) {
  const pct = Math.round(val * 100);
  fillEl.style.width = pct + "%";
  fillEl.style.background = val >= 0.9 ? "var(--green)"
                              : val >= 0.7 ? "var(--accent)"
                              : "var(--yellow)";
  pctEl.textContent = pct + "%";
}

// ────────────────────────────────────────────────────── job list (sidebar) ──
// Per-row "strike" confirmation state — one click arms, second click strikes.
// Resets after 4 seconds of no follow-through.
const strikeArmed = new Map(); // id → timeoutId

function renderJobList() {
  const entries = Object.entries(jobs);
  if (docketCount) {
    docketCount.textContent = entries.length
      ? entries.length + " filing" + (entries.length === 1 ? "" : "s")
      : "—";
  }
  if (entries.length === 0) {
    jobListEl.innerHTML = `<div class="docket-empty">No filings yet.</div>`;
    return;
  }
  jobListEl.innerHTML = "";
  entries.forEach(([id, j]) => {
    const row = document.createElement("div");
    row.className = `docket-item ${j.status || "queued"}`;
    if (id === currentJobId || id === currentBatchId) row.classList.add("active");
    const stateText = strikeArmed.has(id)
      ? "strike?"
      : {
          queued: "queued", running: "live", done: "filed", error: "stricken",
        }[j.status || "queued"];
    row.innerHTML = `
      <span class="docket-id">${escapeHTML(id.slice(0, 4))}</span>
      <span class="docket-name" title="${escapeHTML(j.name)}">${escapeHTML(j.name)}</span>
      <span class="docket-state">${stateText}</span>
      <button type="button" class="docket-strike" title="Strike from the record" aria-label="Strike from the record">×</button>
    `;
    if (strikeArmed.has(id)) row.classList.add("confirming");

    // Row click → open viewer (only for completed single jobs)
    row.addEventListener("click", (e) => {
      // Ignore clicks that originated on the strike button — those have
      // their own handler that stops propagation.
      if (e.target.closest(".docket-strike")) return;
      if (j.kind === "single" && j.status === "done") openViewer(id);
    });

    const strikeBtn = row.querySelector(".docket-strike");
    strikeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (strikeArmed.has(id)) {
        // Second click → actually strike
        clearTimeout(strikeArmed.get(id));
        strikeArmed.delete(id);
        strikeJob(id);
      } else {
        // First click → arm. Re-render to show "strike?" state.
        const t = setTimeout(() => {
          strikeArmed.delete(id);
          renderJobList();
        }, 4000);
        strikeArmed.set(id, t);
        renderJobList();
      }
    });

    jobListEl.appendChild(row);
  });
}

async function strikeJob(id) {
  // Optimistic: remove from sidebar immediately. We consider the user to
  // be "viewing" the struck job if either the running-job pointer matches
  // or the viewer panel is the visible one (since openViewer now keeps
  // currentJobId in sync with what's displayed).
  const viewerOpen = !viewViewer.hidden;
  const wasViewing = (id === currentJobId) || viewerOpen;
  delete jobs[id];
  renderJobList();
  try {
    const res = await fetch(`/jobs/${id}`, { method: "DELETE" });
    if (!res.ok) {
      showError("Couldn't strike that filing.");
    }
  } catch (e) {
    showError("Network error striking filing: " + e.message);
  }
  if (wasViewing) {
    if (currentEventSource) { currentEventSource.close(); currentEventSource = null; }
    currentJobId = null;
    showView("upload");
  }
}

// ───────────────────────────────────────────────────────────── reset helpers ──
function resetProgressUI() {
  for (const k of Object.keys(engineSteps)) delete engineSteps[k];
  for (const k of Object.keys(regionRows)) delete regionRows[k];
  for (const k of Object.keys(regionAttempts)) delete regionAttempts[k];
  for (const k of Object.keys(redactionPages)) delete redactionPages[k];
  for (const k of Object.keys(batchRedactionPages)) delete batchRedactionPages[k];
  pageConfs = [];
  regionPageConfs = [];
  engineLadder.innerHTML = "";
  regionList.innerHTML = "";
  progressRegionCount.textContent = "";
  escalateMsg.hidden = true;
  redactionBanner.hidden = true;
  batchRedactionBanner.hidden = true;
  setMeter(meterFill, meterPct, 0);
  setMeter(regionMeterFill, regionMeterPct, 0);
}

// ──────────────────────────────────────────────────────────────── toast / util ──
let errorTimer = null;
function showError(msg) {
  const body = errorToast.querySelector(".notice-body");
  if (body) body.textContent = msg;
  else errorToast.textContent = msg;
  errorToast.hidden = false;
  clearTimeout(errorTimer);
  errorTimer = setTimeout(() => { errorToast.hidden = true; }, 6000);
}
function hideError() { errorToast.hidden = true; }

function escapeHTML(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// ─────────────────────────────────────────────────── settings (region cascade) ──
// Each settings session holds the latest config in memory; "Save" PUTs the
// whole thing back to the server. We render per-label rows with chips that
// can move up/down/remove, and a dropdown to add unused engines.
let configState = null;          // {labels: [{label, ladder, rationale, citation, is_default}], available_engines: []}
let openAddMenu = null;          // outstanding dropdown element so we can close it on outside-click

settingsBtn.addEventListener("click", async () => {
  showView("settings");
  await loadConfig();
});

settingsResetBtn.addEventListener("click", loadConfig);
settingsSaveBtn.addEventListener("click", saveConfig);

async function loadConfig() {
  settingsList.innerHTML = `<div class="pane-loading"><div class="spinner"></div>Loading…</div>`;
  try {
    const res = await fetch("/config/regions");
    if (!res.ok) throw new Error("config fetch failed");
    configState = await res.json();
    renderSettings();
  } catch (e) {
    settingsList.innerHTML = `<div class="pane-loading">Failed to load config.</div>`;
  }
}

async function saveConfig() {
  if (!configState) return;
  settingsSaveBtn.disabled = true;
  settingsSaveBtn.textContent = "Saving…";
  try {
    const payload = {
      labels: configState.labels.map((l) => ({ label: l.label, ladder: l.ladder })),
    };
    const res = await fetch("/config/regions", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Save failed" }));
      showError(err.detail || "Save failed");
    } else {
      settingsSaveBtn.textContent = "Saved ✓";
      setTimeout(() => { settingsSaveBtn.textContent = "Save"; }, 1400);
    }
  } catch (e) {
    showError("Network error: " + e.message);
    settingsSaveBtn.textContent = "Save";
  } finally {
    settingsSaveBtn.disabled = false;
  }
}

function renderSettings() {
  if (!configState) return;
  settingsList.innerHTML = "";
  configState.labels.forEach((entry, idx) => {
    const row = document.createElement("div");
    row.className = "rules-row";
    row.innerHTML = `
      <div class="rules-body">
        <div class="rules-head">
          <span class="rules-tag ${entry.is_default ? "default" : ""}">${escapeHTML(entry.label)}</span>
          ${entry.is_default ? '<span class="rules-note">catches every label not listed above</span>' : ""}
        </div>
        <div class="ladder-chips" data-idx="${idx}"></div>
        ${entry.rationale ? `<div class="rationale">${escapeHTML(entry.rationale)}</div>` : ""}
      </div>
    `;
    settingsList.appendChild(row);
    renderLadderChips(row.querySelector(".ladder-chips"), idx);
  });
}

function renderLadderChips(container, idx) {
  const entry = configState.labels[idx];
  const ladder = entry.ladder;
  container.innerHTML = "";

  if (ladder.length === 0) {
    const empty = document.createElement("span");
    empty.className = "chip empty";
    empty.textContent = "(no engines — region will be skipped)";
    container.appendChild(empty);
  }

  ladder.forEach((engine, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `
      <span class="pos">${i + 1}.</span>
      <span class="name">${escapeHTML(engine)}</span>
      <button type="button" data-act="up"    title="Move up">↑</button>
      <button type="button" data-act="down"  title="Move down">↓</button>
      <button type="button" data-act="rm" class="danger" title="Remove">×</button>
    `;
    chip.querySelector('[data-act="up"]').disabled = (i === 0);
    chip.querySelector('[data-act="down"]').disabled = (i === ladder.length - 1);
    chip.querySelector('[data-act="up"]').addEventListener("click", () => {
      if (i === 0) return;
      [ladder[i - 1], ladder[i]] = [ladder[i], ladder[i - 1]];
      renderLadderChips(container, idx);
    });
    chip.querySelector('[data-act="down"]').addEventListener("click", () => {
      if (i === ladder.length - 1) return;
      [ladder[i + 1], ladder[i]] = [ladder[i], ladder[i + 1]];
      renderLadderChips(container, idx);
    });
    chip.querySelector('[data-act="rm"]').addEventListener("click", () => {
      ladder.splice(i, 1);
      renderLadderChips(container, idx);
    });
    container.appendChild(chip);
  });

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "chip-add";
  addBtn.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg> add engine`;
  addBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    openAddEngineMenu(addBtn, idx);
  });
  container.appendChild(addBtn);
}

function openAddEngineMenu(anchor, idx) {
  closeAddEngineMenu();
  const ladder = configState.labels[idx].ladder;
  const menu = document.createElement("div");
  menu.className = "chip-add-menu";
  configState.available_engines.forEach((engine) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = engine;
    btn.disabled = ladder.includes(engine);
    btn.addEventListener("click", () => {
      ladder.push(engine);
      closeAddEngineMenu();
      // Re-render the parent's chips
      const container = anchor.parentElement;
      renderLadderChips(container, idx);
    });
    menu.appendChild(btn);
  });
  document.body.appendChild(menu);
  // Position underneath anchor
  const rect = anchor.getBoundingClientRect();
  menu.style.top  = (rect.bottom + 4 + window.scrollY) + "px";
  menu.style.left = (rect.left + window.scrollX) + "px";
  openAddMenu = menu;
  // Close on outside click
  setTimeout(() => {
    document.addEventListener("click", outsideMenuClick, { once: true });
  }, 0);
}

function outsideMenuClick(e) {
  if (openAddMenu && !openAddMenu.contains(e.target)) closeAddEngineMenu();
}

function closeAddEngineMenu() {
  if (openAddMenu) {
    openAddMenu.remove();
    openAddMenu = null;
  }
}

// ───────────────────────────────────────────────────── docket bootstrap ──
// Populate the sidebar from the server on page load so jobs from previous
// sessions are visible. Without this the docket starts empty every reload.
async function bootstrapDocket() {
  try {
    const res = await fetch("/jobs");
    if (!res.ok) return;
    const data = await res.json();
    // Reset to the server's view of truth — drop any in-page entries that
    // are no longer on the server (e.g. struck in a previous tab).
    for (const k of Object.keys(jobs)) {
      // Keep in-flight (no status yet, or running) so we don't clobber the
      // entry we just submitted.
      if (jobs[k] && jobs[k].status === "running") continue;
      delete jobs[k];
    }
    (data.jobs || []).forEach((j) => {
      jobs[j.id] = {
        kind: j.kind || "single",
        mode: j.mode || "regions",
        name: j.name || j.id,
        status: j.status || "done",
        result: {
          mean_conf: j.mean_conf || 0,
          engine: j.mode === "regions" ? "regions" : (j.engine || ""),
        },
      };
    });
    renderJobList();
  } catch {}
}

// initial render
renderJobList();
showView("upload");
bootstrapDocket();

// ──────────────────────────────────────────────────────── demo mode init ──
async function initDemoMode() {
  try {
    const res = await fetch("/api/demo-config");
    if (!res.ok) return;
    const cfg = await res.json();
    if (!cfg.demo_mode) return;
    const banner = document.getElementById("demo-banner");
    if (banner) banner.hidden = false;
    if (gpuCheckbox) {
      gpuCheckbox.checked = false;
      gpuCheckbox.disabled = true;
    }
  } catch {}
}
initDemoMode();
