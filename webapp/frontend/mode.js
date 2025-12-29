// webapp/frontend/mode.js
//
// Frontend controller for PIXEL-T2I web demo.
//
// Responsibilities:
// - Maintain generation mode state (unconditional vs text-conditioned; action view override)
// - Wire UI controls to backend endpoints
// - Render single preview or batch preview grid
// - Provide local export utilities (save PNG, split 2x2 to ZIP, download batch ZIP)
// - Manage cache clearing and UI enable/disable policies
//
// Conventions:
// - Uses backend-served /static URLs as relative paths and converts them to absolute URLs.
// - Text-conditioned generation is treated as single-sample; batch UI is disabled in that mode.
// - Action view override is indicated by window.__po_action_view (set by action_view.js).

document.addEventListener("DOMContentLoaded", () => {
  // ---------------------------------------------------------------------------
  // Elements
  // ---------------------------------------------------------------------------
  const input = document.getElementById("promptInput");
  const mode = document.getElementById("modeIndicator");

  const generateBtn = document.getElementById("generateBtn");
  const previewImage = document.getElementById("previewImage");
  const exampleBadge = document.getElementById("exampleBadge");
  const batchBadge = document.getElementById("batchBadge");
  const generatingOverlay = document.getElementById("generatingOverlay");

  // Batch preview grid (up to 4 images)
  const batchPreviewGrid = document.getElementById("batchPreviewGrid");
  const batchImg1 = document.getElementById("batchImg1");
  const batchImg2 = document.getElementById("batchImg2");
  const batchImg3 = document.getElementById("batchImg3");
  const batchImg4 = document.getElementById("batchImg4");
  const batchPreviewImgs = [batchImg1, batchImg2, batchImg3, batchImg4];

  const diceBtn = document.querySelector(".dice-button");

  const toolsBtn = document.getElementById("toolsBtn");
  const toolsPanel = document.getElementById("toolsPanel");

  const ddimSteps = document.getElementById("ddimSteps");
  const ddimStepsValue = document.getElementById("ddimStepsValue");

  const savePngBtn = document.getElementById("savePngBtn");
  const split4viewBtn = document.getElementById("split4viewBtn");
  const downloadBatchBtn = document.getElementById("downloadBatchBtn");
  const clearCacheBtn = document.getElementById("clearCacheBtn");

  const batchToggle = document.getElementById("batchToggle");
  const batchQty = document.getElementById("batchQty");
  const batchQtyValue = document.getElementById("batchQtyValue");

  // "Batch Generation" row + label (for tooltip placement)
  const batchToggleRow = batchToggle.closest(".tools-row");
  const batchToggleLabel = batchToggleRow ? batchToggleRow.querySelector(".tools-label") : null;

  if (
    !input || !mode || !generateBtn || !previewImage || !exampleBadge || !batchBadge || !generatingOverlay ||
    !toolsBtn || !toolsPanel || !ddimSteps || !ddimStepsValue ||
    !savePngBtn || !split4viewBtn || !downloadBatchBtn || !clearCacheBtn ||
    !batchToggle || !batchQty || !batchQtyValue ||
    !batchToggleRow || !batchToggleLabel ||
    !batchPreviewGrid || batchPreviewImgs.some((x) => !x)
  ) {
    console.warn("Missing DOM elements. Check ids.");
    return;
  }

  // ---------------------------------------------------------------------------
  // Backend config
  // ---------------------------------------------------------------------------
  const API_BASE = "http://127.0.0.1:8000";
  const API_GENERATE = `${API_BASE}/api/generate`;
  const API_CLEAR_CACHE = `${API_BASE}/api/clear_cache`;

  // ---------------------------------------------------------------------------
  // Mode text
  // ---------------------------------------------------------------------------
  const RAND_MODE = "Mode1: Random Generation";
  const TEXT_MODE = "Mode2: Text Prompt Generation";
  const ACTION_MODE = "Mode3: Action Generation";

  const TOOLTIP_RANDOM = "Generate a random character";
  const TOOLTIP_TEXT = "Generate a character based on your description";
  const TOOLTIP_BATCH_DISABLED_TEXT =
    "Batch generation is unavailable in Text Mode to keep generation stable and responsive";

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let lastGeneratedAbsoluteUrl = null;

  // Batch state is tracked by batch id + count.
  let lastBatchId = null;
  let lastBatchCount = 0;

  /**
   * Compute the effective mode and apply UI policy.
   *
   * Rules:
   * - If action view is active, the mode indicator shows ACTION_MODE and the
   *   generate button tooltip reflects unconditional generation intent.
   * - If the prompt input is non-empty, the mode is TEXT_MODE and batch controls
   *   are forcibly disabled in the UI (backend also enforces single-sample).
   * - If the prompt input is empty, the mode is RAND_MODE and batch controls are enabled.
   */
  function updateMode() {
    // Action view overrides mode text.
    if (window.__po_action_view === true) {
      mode.textContent = ACTION_MODE;
      generateBtn.title = TOOLTIP_RANDOM;
      return;
    }

    const hasText = input.value.trim().length > 0;
    mode.textContent = hasText ? TEXT_MODE : RAND_MODE;
    generateBtn.title = hasText ? TOOLTIP_TEXT : TOOLTIP_RANDOM;

    // Text mode: batch is disabled (backend forces n=1 anyway).
    if (hasText) {
      // Enforce UI state.
      batchToggle.checked = false;
      batchToggle.disabled = true;

      // Quantity is always disabled in text mode.
      batchQty.disabled = true;
      batchQtyValue.style.opacity = "0.6";

      // Tooltip: show on the "Batch Generation" label only.
      batchToggleLabel.title = TOOLTIP_BATCH_DISABLED_TEXT;

      // Remove tooltips from switch/slider to avoid mixed placement.
      batchToggle.title = "";
      batchQty.title = "";
      const batchSwitchLabel = batchToggle.closest(".tools-switch");
      if (batchSwitchLabel) batchSwitchLabel.title = "";
    } else {
      // Unconditional mode: batch can be used.
      batchToggle.disabled = false;

      // Restore default switch tooltip.
      const batchSwitchLabel = batchToggle.closest(".tools-switch");
      if (batchSwitchLabel) batchSwitchLabel.title = "Enable batch generation";

      // Clear label tooltip.
      batchToggleLabel.title = "";

      // Apply normal batch UI state (enables/disables qty based on toggle).
      applyBatchUiState();
    }
  }

  /**
   * Auto-resize the prompt input when it is a textarea.
   *
   * This keeps a compact layout for short prompts while allowing the input
   * to expand (up to MAX_H) before enabling scrollbars.
   */
  function autoResizePrompt() {
    if (!input) return;
    if (input.tagName !== "TEXTAREA") return;

    const MAX_H = 110; // Must remain consistent with CSS max-height.

    // Reset to measure correct scrollHeight.
    input.style.height = "auto";

    const needed = input.scrollHeight;

    if (needed <= MAX_H) {
      // Grow freely, hide scrollbar.
      input.style.overflowY = "hidden";
      input.style.height = needed + "px";
    } else {
      // Clamp height, enable scrolling.
      input.style.height = MAX_H + "px";
      input.style.overflowY = "auto";
    }
  }

  /**
   * Set the prompt input value and immediately refresh mode-related UI state.
   *
   * Focus is restored to the input after the click finishes to avoid leaving
   * focus on the triggering button.
   *
   * @param {string} prompt - Prompt text to set.
   */
  function setPromptAndUpdateMode(prompt) {
    input.value = prompt;
    updateMode();
    autoResizePrompt();

    const focusAtStart = () => {
      try {
        input.focus({ preventScroll: true });
        input.setSelectionRange(0, 0);
      } catch (_) {}
    };

    // Next repaint.
    requestAnimationFrame(focusAtStart);
    // After current task.
    setTimeout(focusAtStart, 0);
  }

  // ---------------------------------------------------------------------------
  // UI helpers
  // ---------------------------------------------------------------------------
  function show(el, display = "block") {
    el.style.display = display;
    el.hidden = false;
  }

  function hide(el) {
    el.style.display = "none";
    el.hidden = true;
  }

  /**
   * Enable or disable a button-like element while updating ARIA and tooltip text.
   *
   * @param {HTMLElement} buttonEl - Button element.
   * @param {boolean} enabled - Whether the button should be enabled.
   * @param {string} [titleIfDisabled] - Tooltip text applied only when disabled.
   */
  function setEnabled(buttonEl, enabled, titleIfDisabled = "") {
    buttonEl.disabled = !enabled;
    buttonEl.setAttribute("aria-disabled", String(!enabled));
    if (!enabled && titleIfDisabled) buttonEl.title = titleIfDisabled;
    if (enabled) buttonEl.title = "";
  }

  /**
   * Switch the output area to single-image preview mode.
   */
  function showSinglePreview() {
    hide(batchPreviewGrid);
    show(previewImage, "block");
  }

  /**
   * Switch the output area to batch grid preview mode.
   */
  function showBatchPreview() {
    hide(previewImage);
    show(batchPreviewGrid, "grid");
  }

  /**
   * Render up to four batch preview images.
   *
   * @param {string[]} absUrls - Absolute image URLs; length 0..4.
   */
  function setBatchPreviewImages(absUrls) {
    for (let i = 0; i < batchPreviewImgs.length; i++) {
      const imgEl = batchPreviewImgs[i];
      const url = absUrls[i];

      if (url) {
        imgEl.src = `${url}?t=${Date.now()}`;
        imgEl.hidden = false;
        imgEl.style.display = "block";
      } else {
        imgEl.src = "";
        imgEl.hidden = true;
        imgEl.style.display = "none";
      }
    }
  }

  /**
   * Toggle generating overlay and generation button state.
   *
   * When generating:
   * - preview elements and badges are hidden
   * - overlay is shown
   * - generate button is disabled to prevent concurrent requests
   *
   * @param {boolean} isGenerating - Whether generation is in progress.
   */
  function setGenerating(isGenerating) {
    if (isGenerating) {
      hide(previewImage);
      hide(batchPreviewGrid);
      hide(exampleBadge);
      hide(batchBadge);

      show(generatingOverlay, "flex");

      generateBtn.disabled = true;
      generateBtn.style.opacity = "0.75";
      generateBtn.style.cursor = "not-allowed";
    } else {
      hide(generatingOverlay);

      generateBtn.disabled = false;
      generateBtn.style.opacity = "1";
      generateBtn.style.cursor = "pointer";
    }
  }

  function openTools() {
    show(toolsPanel, "block");
    toolsBtn.setAttribute("aria-expanded", "true");
    toolsBtn.classList.add("is-open");
  }

  function closeTools() {
    hide(toolsPanel);
    toolsBtn.setAttribute("aria-expanded", "false");
    toolsBtn.classList.remove("is-open");
  }

  function toggleTools() {
    const isOpen = toolsBtn.getAttribute("aria-expanded") === "true";
    if (isOpen) closeTools();
    else openTools();
  }

  /**
   * Update the visible label for DDIM steps slider.
   */
  function updateDdimLabel() {
    ddimStepsValue.textContent = String(ddimSteps.value);
  }

  /**
   * Update the visible label for batch quantity slider.
   */
  function updateBatchQtyLabel() {
    batchQtyValue.textContent = String(batchQty.value);
  }

  /**
   * Apply batch UI enable/disable state based on:
   * - prompt content (text mode disables batch)
   * - batch toggle (enables/disables quantity slider)
   */
  function applyBatchUiState() {
    const hasText = input.value.trim().length > 0;

    // Text-conditioned generation does not support batch (backend forces n=1).
    if (hasText) {
      batchToggle.checked = false;
      batchToggle.disabled = true;

      batchQty.disabled = true;
      batchQtyValue.style.opacity = "0.6";
      return;
    }

    // Unconditional: batch allowed.
    batchToggle.disabled = false;

    const on = batchToggle.checked;
    batchQty.disabled = !on;
    batchQtyValue.style.opacity = on ? "1" : "0.6";
  }

  // ---------------------------------------------------------------------------
  // Filename helpers
  // ---------------------------------------------------------------------------
  function pad2(n) {
    return String(n).padStart(2, "0");
  }

  function makeTimestampBase(prefix = "pixel_odyssey") {
    const d = new Date();
    const yyyy = d.getFullYear();
    const mm = pad2(d.getMonth() + 1);
    const dd = pad2(d.getDate());
    const hh = pad2(d.getHours());
    const mi = pad2(d.getMinutes());
    const ss = pad2(d.getSeconds());
    return `${prefix}_${yyyy}${mm}${dd}_${hh}${mi}${ss}`;
  }

  function makeTimestampFilename(prefix = "pixel_odyssey", ext = "png") {
    return `${makeTimestampBase(prefix)}.${ext}`;
  }

  // ---------------------------------------------------------------------------
  // Download / Save helpers
  // ---------------------------------------------------------------------------
  /**
   * Fetch a URL as a Blob without caching.
   *
   * @param {string} url - Resource URL.
   * @returns {Promise<Blob>} Downloaded blob.
   */
  async function fetchAsBlob(url) {
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`Failed to fetch blob (HTTP ${resp.status}).`);
    return await resp.blob();
  }

  /**
   * Save using the File System Access API when available.
   *
   * @param {Blob} blob - Content.
   * @param {string} filename - Suggested filename.
   * @param {string} mime - MIME type.
   */
  async function saveWithPicker(blob, filename, mime) {
    const fileHandle = await window.showSaveFilePicker({
      suggestedName: filename,
      types: [
        { description: "File", accept: { [mime]: [`.${filename.split(".").pop()}`] } },
      ],
    });

    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();
  }

  /**
   * Save by creating a temporary download link.
   *
   * @param {Blob} blob - Content.
   * @param {string} filename - Download filename.
   */
  function saveWithDownloadLink(blob, filename) {
    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = filename;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  }

  /**
   * Save the currently displayed single output PNG.
   */
  async function handleSavePng() {
    if (!lastGeneratedAbsoluteUrl) return;

    try {
      setEnabled(savePngBtn, false);

      const filename = makeTimestampFilename("pixel_odyssey", "png");
      const blob = await fetchAsBlob(lastGeneratedAbsoluteUrl);

      if (window.showSaveFilePicker) {
        await saveWithPicker(blob, filename, "image/png");
      } else {
        saveWithDownloadLink(blob, filename);
      }
    } catch (err) {
      console.error(err);
      alert("Save failed.\n\nYour browser may not support a native Save As picker.");
    } finally {
      refreshOutputActions();
    }
  }

  /**
   * Download the last generated batch as a ZIP using backend /api/download_batch.
   */
  async function handleDownloadBatchZip() {
    if (!lastBatchId || lastBatchCount <= 0) return;

    try {
      setEnabled(downloadBatchBtn, false);

      const zipFilename = `${makeTimestampBase("pixel_odyssey")}_batch_${lastBatchCount}.zip`;
      const url = `${API_BASE}/api/download_batch?batch_id=${encodeURIComponent(lastBatchId)}&t=${Date.now()}`;

      const zipBlob = await fetchAsBlob(url);

      if (window.showSaveFilePicker) {
        await saveWithPicker(zipBlob, zipFilename, "application/zip");
      } else {
        saveWithDownloadLink(zipBlob, zipFilename);
      }
    } catch (err) {
      console.error(err);
      alert("Batch download failed.\n\nCheck backend logs for /api/download_batch.");
    } finally {
      refreshOutputActions();
    }
  }

  /**
   * Clear backend cache via POST /api/clear_cache and reset the UI to the example state.
   */
  async function handleClearCache() {
    const ok = confirm("Clear cache?\n\nThis will delete cached generated images on disk.");
    if (!ok) return;

    const ok2 = confirm("Are you sure?\n\nThis cannot be undone.");
    if (!ok2) return;

    try {
      clearCacheBtn.disabled = true;

      const resp = await fetch(API_CLEAR_CACHE, { method: "POST" });
      const data = await resp.json().catch(() => ({}));

      if (!resp.ok || data.error) {
        throw new Error(data?.error || `Clear cache failed (HTTP ${resp.status}).`);
      }

      lastGeneratedAbsoluteUrl = null;
      lastBatchId = null;
      lastBatchCount = 0;

      previewImage.src = "images/4view_example.png";
      previewImage.classList.add("placeholder-image");

      showSinglePreview();
      show(exampleBadge, "block");
      hide(batchBadge);

      // Clear batch preview images.
      setBatchPreviewImages([]);

      refreshOutputActions();

      alert(`Cache cleared.\nDeleted ${data.deleted ?? 0} file(s).`);
    } catch (err) {
      console.error(err);
      alert("Clear cache failed.\n\nCheck backend is running and /api/clear_cache exists.");
    } finally {
      clearCacheBtn.disabled = false;
    }
  }

  // ---------------------------------------------------------------------------
  // 2x2 split utilities (grid -> four directional PNG blobs)
  // ---------------------------------------------------------------------------
  function imageFromBlob(blob) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to decode image blob."));
      img.src = URL.createObjectURL(blob);
    });
  }

  function canvasToBlob(canvas, type = "image/png", quality) {
    return new Promise((resolve, reject) => {
      canvas.toBlob((b) => {
        if (!b) reject(new Error("Failed to export canvas to blob."));
        else resolve(b);
      }, type, quality);
    });
  }

  /**
   * Split a 2x2 grid PNG into four directional PNG blobs.
   *
   * Assumptions:
   * - Image width and height are even.
   * - Layout: [north | west] on top row, [south | east] on bottom row.
   *
   * @param {Blob} gridBlob - PNG blob representing a 2x2 sprite grid.
   * @returns {Promise<Object<string, Blob>>} Mapping {north, west, south, east} -> PNG blob.
   */
  async function splitGridToPngBlobs(gridBlob) {
    const img = await imageFromBlob(gridBlob);

    const W = img.naturalWidth || img.width;
    const H = img.naturalHeight || img.height;

    if (W <= 1 || H <= 1) throw new Error("Invalid image size for splitting.");
    if (W % 2 !== 0 || H % 2 !== 0) throw new Error(`Grid image size must be even. Got ${W}x${H}.`);

    const tileW = W / 2;
    const tileH = H / 2;

    const canvas = document.createElement("canvas");
    canvas.width = tileW;
    canvas.height = tileH;

    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Failed to create 2D canvas context.");

    const regions = [
      { name: "north", sx: 0,      sy: 0 },
      { name: "west",  sx: tileW,  sy: 0 },
      { name: "south", sx: 0,      sy: tileH },
      { name: "east",  sx: tileW,  sy: tileH },
    ];

    const out = {};
    for (const r of regions) {
      ctx.clearRect(0, 0, tileW, tileH);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, r.sx, r.sy, tileW, tileH, 0, 0, tileW, tileH);
      out[r.name] = await canvasToBlob(canvas, "image/png");
    }

    // Revoke the temporary object URL used by Image().
    try { URL.revokeObjectURL(img.src); } catch (_) {}
    return out;
  }

  // ---------------------------------------------------------------------------
  // ZIP writer (STORE method)
  // ---------------------------------------------------------------------------
  function crc32(bytes) {
    let crc = 0 ^ -1;
    for (let i = 0; i < bytes.length; i++) {
      crc = (crc >>> 8) ^ CRC_TABLE[(crc ^ bytes[i]) & 0xff];
    }
    return (crc ^ -1) >>> 0;
  }

  // Precompute CRC table once.
  const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
      table[i] = c >>> 0;
    }
    return table;
  })();

  function u16le(n) {
    return new Uint8Array([n & 0xff, (n >>> 8) & 0xff]);
  }

  function u32le(n) {
    return new Uint8Array([n & 0xff, (n >>> 8) & 0xff, (n >>> 16) & 0xff, (n >>> 24) & 0xff]);
  }

  function concatBytes(chunks) {
    const total = chunks.reduce((s, a) => s + a.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const a of chunks) { out.set(a, off); off += a.length; }
    return out;
  }

  function textUtf8(str) {
    return new TextEncoder().encode(str);
  }

  function dosDateTime(d) {
    // DOS date/time used by ZIP headers.
    const year = Math.max(1980, d.getFullYear());
    const month = d.getMonth() + 1;
    const day = d.getDate();
    const hour = d.getHours();
    const minute = d.getMinutes();
    const second = Math.floor(d.getSeconds() / 2);
    const dosTime = (hour << 11) | (minute << 5) | second;
    const dosDate = ((year - 1980) << 9) | (month << 5) | day;
    return { dosTime, dosDate };
  }

  /**
   * Create a minimal ZIP archive using the STORE method (no compression).
   *
   * @param {Array<{name: string, data: Uint8Array}>} files - File entries.
   * @returns {Blob} ZIP blob.
   */
  function makeZipStore(files) {
    const now = new Date();
    const { dosTime, dosDate } = dosDateTime(now);

    const localHeaders = [];
    const centralHeaders = [];
    let offset = 0;

    for (const f of files) {
      const nameBytes = textUtf8(f.name);
      const dataBytes = f.data;
      const crc = crc32(dataBytes);
      const size = dataBytes.length;

      // Local file header.
      const local = concatBytes([
        u32le(0x04034b50),
        u16le(20),          // version needed
        u16le(0),           // general purpose
        u16le(0),           // compression = STORE
        u16le(dosTime),
        u16le(dosDate),
        u32le(crc),
        u32le(size),
        u32le(size),
        u16le(nameBytes.length),
        u16le(0),           // extra len
        nameBytes,
        dataBytes,
      ]);
      localHeaders.push(local);

      // Central directory header.
      const central = concatBytes([
        u32le(0x02014b50),
        u16le(20),          // version made by
        u16le(20),          // version needed
        u16le(0),           // general purpose
        u16le(0),           // compression = STORE
        u16le(dosTime),
        u16le(dosDate),
        u32le(crc),
        u32le(size),
        u32le(size),
        u16le(nameBytes.length),
        u16le(0),           // extra len
        u16le(0),           // comment len
        u16le(0),           // disk number
        u16le(0),           // internal attrs
        u32le(0),           // external attrs
        u32le(offset),      // local header offset
        nameBytes,
      ]);
      centralHeaders.push(central);

      offset += local.length;
    }

    const centralStart = offset;
    const centralDir = concatBytes(centralHeaders);

    // End of central directory record.
    const eocd = concatBytes([
      u32le(0x06054b50),
      u16le(0),                        // disk number
      u16le(0),                        // central dir disk
      u16le(files.length),             // entries on disk
      u16le(files.length),             // total entries
      u32le(centralDir.length),        // central dir size
      u32le(centralStart),             // central dir offset
      u16le(0),                        // comment len
    ]);

    const zipBytes = concatBytes([...localHeaders, centralDir, eocd]);
    return new Blob([zipBytes], { type: "application/zip" });
  }

  /**
   * Split the current single 2x2 output into four directional PNGs and write them into a ZIP.
   */
  async function handleSplit4ViewToZip() {
    if (!lastGeneratedAbsoluteUrl) return;

    try {
      setEnabled(split4viewBtn, false);

      const base = makeTimestampBase("pixel_odyssey");
      const zipName = `${base}_4view.zip`;

      const gridBlob = await fetchAsBlob(lastGeneratedAbsoluteUrl);
      const parts = await splitGridToPngBlobs(gridBlob);

      const entries = [];
      for (const key of ["north", "west", "south", "east"]) {
        const blob = parts[key];
        const buf = new Uint8Array(await blob.arrayBuffer());
        entries.push({ name: `${base}_${key}.png`, data: buf });
      }

      const zipBlob = makeZipStore(entries);

      if (window.showSaveFilePicker) {
        await saveWithPicker(zipBlob, zipName, "application/zip");
      } else {
        saveWithDownloadLink(zipBlob, zipName);
      }
    } catch (err) {
      console.error(err);
      alert("Split failed.\n\nMake sure the image is a 2x2 grid and has even dimensions.");
    } finally {
      refreshOutputActions();
    }
  }

  /**
   * Enable or disable output actions according to current state:
   * - Single actions (Save/Split) require a single generated image and batch must be off.
   * - Batch download requires a batch result and batch must be on.
   * - Clear cache is always available.
   * - Batch badge reflects last batch preview state.
   */
  function refreshOutputActions() {
    const isBatch = Boolean(batchToggle.checked);

    // Single mode uses lastGeneratedAbsoluteUrl.
    const hasSingle = Boolean(lastGeneratedAbsoluteUrl);

    // Batch mode uses lastBatchId + lastBatchCount.
    const hasBatch = Boolean(lastBatchId) && Number.isFinite(lastBatchCount) && lastBatchCount > 0;

    // Save / Split: only available in single mode and only after a single image exists.
    if (isBatch) {
      setEnabled(savePngBtn, false, "Switch off Batch Generation to use this action");
      setEnabled(split4viewBtn, false, "Switch off Batch Generation to use this action");
    } else {
      setEnabled(savePngBtn, hasSingle, "Generate an image first");
      setEnabled(split4viewBtn, hasSingle, "Generate an image first");
    }

    // Download batch: only available in batch mode and only after batch exists.
    if (!isBatch) {
      setEnabled(downloadBatchBtn, false, "Switch on Batch Generation to use this action");
    } else {
      setEnabled(downloadBatchBtn, hasBatch, "Generate a batch first");
    }

    // Clear cache: always enabled.
    setEnabled(clearCacheBtn, true);

    // Badge: show batch preview label when a batch result exists.
    if (isBatch && hasBatch) {
      const shown = Math.min(4, lastBatchCount);
      batchBadge.textContent = `Batch Preview: 1-${shown}/${lastBatchCount}`;
      show(batchBadge, "block");
    } else {
      hide(batchBadge);
    }
  }

  // ---------------------------------------------------------------------------
  // Initial state
  // ---------------------------------------------------------------------------
  updateMode();
  autoResizePrompt();
  updateDdimLabel();
  updateBatchQtyLabel();
  applyBatchUiState();

  showSinglePreview();
  show(exampleBadge, "block");
  hide(batchBadge);
  hide(generatingOverlay);
  closeTools();

  // Clear grid images on load.
  setBatchPreviewImages([]);

  refreshOutputActions();

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------
  input.addEventListener("input", () => {
    updateMode();
    autoResizePrompt();
  });

  toolsBtn.addEventListener("click", (e) => {
    e.preventDefault();
    toggleTools();
  });

  ddimSteps.addEventListener("input", updateDdimLabel);

  batchToggle.addEventListener("change", () => {
    // Enforce text-mode policy even if toggled programmatically.
    const hasText = input.value.trim().length > 0;
    if (hasText) {
      batchToggle.checked = false;
    }

    applyBatchUiState();
    refreshOutputActions();
  });

  batchQty.addEventListener("input", updateBatchQtyLabel);

  document.addEventListener("click", (e) => {
    const isOpen = toolsBtn.getAttribute("aria-expanded") === "true";
    if (!isOpen) return;
    const clickedInsideTools = toolsPanel.contains(e.target) || toolsBtn.contains(e.target);
    if (!clickedInsideTools) closeTools();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const isOpen = toolsBtn.getAttribute("aria-expanded") === "true";
    if (isOpen) closeTools();
  });

  savePngBtn.addEventListener("click", handleSavePng);
  split4viewBtn.addEventListener("click", handleSplit4ViewToZip);
  downloadBatchBtn.addEventListener("click", handleDownloadBatchZip);
  clearCacheBtn.addEventListener("click", handleClearCache);

  generateBtn.addEventListener("click", async () => {
    const prompt = input.value.trim();
    const hasText = prompt.length > 0;

    setGenerating(true);

    // Prevent saving stale images during a new run.
    lastGeneratedAbsoluteUrl = null;
    lastBatchId = null;
    lastBatchCount = 0;

    // Clear any previous grid content while generating.
    setBatchPreviewImages([]);

    refreshOutputActions();

    try {
      const steps = parseInt(ddimSteps.value, 10);
      const isBatch = batchToggle.checked && !hasText;

      // Text-conditioned: force single-sample in frontend as well (backend also enforces it).
      const n = isBatch
        ? Math.max(1, Math.min(50, parseInt(batchQty.value, 10)))
        : 1;

      const resp = await fetch(API_GENERATE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          num_samples: n,
          ddim_steps: steps,

          // Kept for compatibility; backend enforces text cfg_scale=1.0 anyway.
          cfg_scale: 1.0,
        }),
      });

      const data = await resp.json().catch(() => null);
      if (!data || !resp.ok || data.error) {
        throw new Error(data?.error || `Generation failed (HTTP ${resp.status}).`);
      }

      hide(exampleBadge);

      // Batch payload.
      if (data.batch && data.batch.enabled) {
        lastBatchId = data.batch.id;
        lastBatchCount = data.batch.count;

        const urls = Array.isArray(data.preview_urls) ? data.preview_urls : [];
        const absUrls = urls.map((u) => `${API_BASE}${u}`);

        showBatchPreview();
        setBatchPreviewImages(absUrls);

        // Batch results are not used for action generation.
        window.__po_last_generated_abs_url = null;

        refreshOutputActions();
      } else {
        // Single payload.
        const firstAbs = `${API_BASE}${data.first_image_url || data.image_url}`;
        const firstDisplay = `${firstAbs}?t=${Date.now()}`;

        showSinglePreview();
        previewImage.src = firstDisplay;
        previewImage.classList.remove("placeholder-image");

        lastGeneratedAbsoluteUrl = firstAbs;
        window.__po_last_generated_abs_url = lastGeneratedAbsoluteUrl;
        hide(batchBadge);

        refreshOutputActions();
      }
    } catch (err) {
      console.error(err);

      alert(
        "Generation failed.\n\n" +
        "Check:\n" +
        "1) Backend running on http://127.0.0.1:8000\n" +
        "2) Frontend is served via http:// (not file://)\n" +
        "3) API endpoint /api/generate is correct"
      );

      showSinglePreview();
      show(exampleBadge, "block");
      hide(batchBadge);

      setBatchPreviewImages([]);

      refreshOutputActions();
    } finally {
      setGenerating(false);
      refreshOutputActions();
    }
  });

  if (diceBtn) {
    diceBtn.addEventListener("click", () => {
      // Release focus from the button.
      diceBtn.blur();

      // Spin animation.
      diceBtn.classList.remove("is-rolling");
      void diceBtn.offsetWidth;
      diceBtn.classList.add("is-rolling");

      // Random prompt from generator.
      if (typeof window.getRandomPrompt !== "function") {
        console.warn("getRandomPrompt() not found. Ensure prompt_generator.js is loaded before mode.js.");
        return;
      }

      const prompt = window.getRandomPrompt();
      setPromptAndUpdateMode(prompt);
    });
  } else {
    console.warn("Dice button not found: .dice-button");
  }

  /**
   * Expose a safe mode refresh hook for other scripts (e.g. action_view.js).
   */
  window.__po_refresh_mode = () => {
    try { updateMode(); } catch (_) {}
  };
});
