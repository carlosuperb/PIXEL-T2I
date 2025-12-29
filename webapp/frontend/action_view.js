// webapp/frontend/action_view.js
//
// Action view controller for PIXEL-T2I web demo.
//
// Responsibilities:
// - Switch between character preview view and action preview view
// - Request an image-conditional actions sheet from the backend (once per character)
// - Preview actions via a lightweight canvas overlay (no additional backend calls)
// - Provide a PNG download for the generated actions sheet
//
// Conventions:
// - Action view state is exposed via window.__po_action_view for integration with mode.js
// - The last generated character absolute URL is read from window.__po_last_generated_abs_url
// - Backend expects character_url as a backend-served path starting with /static/

document.addEventListener("DOMContentLoaded", () => {
  // ---------------------------------------------------------------------------
  // Elements
  // ---------------------------------------------------------------------------
  const generateActionsBtn = document.getElementById("generateActionsBtn");
  const backBtn = document.getElementById("backToCharacterBtn");
  const previewImage = document.getElementById("previewImage");

  const actionTabs = Array.from(document.querySelectorAll(".action-tab"));
  const actionPreviewImg = document.getElementById("actionPreviewImg");

  const actionGeneratingOverlay = document.getElementById("actionGeneratingOverlay");
  const actionDownloadBtn = document.getElementById("actionDownloadBtn");

  if (!generateActionsBtn || !backBtn || !previewImage || !actionPreviewImg) {
    console.warn("[actions] Missing DOM elements.");
    return;
  }

  // ---------------------------------------------------------------------------
  // Config
  // ---------------------------------------------------------------------------
  const API_BASE = "http://127.0.0.1:8000";
  const ACTIONS_API_URL = `${API_BASE}/api/generate_actions`;
  const EXAMPLE_SRC_KEYWORD = "images/4view_example.png";

  // ---------------------------------------------------------------------------
  // Animation Overlay Config
  // ---------------------------------------------------------------------------
  const TILE_SIZE = 64;

  // Full actions sheet layout:
  // - walk   : rows 0..3,  cols=9
  // - thrust : rows 4..7,  cols=8
  // - slash  : rows 8..11, cols=6
  //
  // startRow is expressed in tile rows, not pixels.
  const ACTION_SHEET_LAYOUT = {
    walk: { startRow: 0, cols: 9, label: "WALK" },
    thrust: { startRow: 4, cols: 8, label: "THRUST" },
    slash: { startRow: 8, cols: 6, label: "SLASH" },
  };

  // Row order inside each 4-row block (tile rows).
  const DIR_ROW_OFFSET = {
    W: 0,
    E: 1,
    S: 2,
    N: 3,
  };

  const DEFAULT_DIR = "S";
  const DEFAULT_FPS = 8;

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let isGenerating = false;
  let lastAction = "walk";

  // Cache key = character_url; value = absolute sheet URL (with cache-busting query).
  const cachedSheetUrls = {};

  // Animation overlay state
  let overlayEl = null;
  let overlayCanvas = null;
  let overlayCtx = null;

  let overlayAction = "walk";
  let overlayDir = DEFAULT_DIR;
  let overlayFps = DEFAULT_FPS;
  let overlayPlaying = true;

  let rafId = 0;
  let lastFrameTs = 0;
  let frameIndex = 0;

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  /**
   * Determine whether a generated character is available for action generation.
   *
   * Action generation is only enabled when:
   * - A last generated absolute URL exists
   * - The preview is not the example placeholder
   *
   * @returns {boolean} Whether a valid generated character exists.
   */
  function isGeneratedCharacterReady() {
    const abs = (window.__po_last_generated_abs_url || "").trim();
    if (!abs) return false;
    if (abs.includes(EXAMPLE_SRC_KEYWORD)) return false;
    return true;
  }

  /**
   * Update the state and tooltip of the "Generate Actions" entry point.
   */
  function updateGenerateActionsState() {
    const ready = isGeneratedCharacterReady();
    generateActionsBtn.disabled = !ready || isGenerating;

    if (isGenerating) {
      generateActionsBtn.title = "Generating actions...";
    } else if (!ready) {
      generateActionsBtn.title = "Action generation is only available for a single generated character";
    } else {
      generateActionsBtn.title = "Open Action Generation";
    }
  }

  /**
   * Refresh mode indicator state in mode.js, when available.
   */
  function refreshModeIndicator() {
    if (typeof window.__po_refresh_mode === "function") {
      try { window.__po_refresh_mode(); } catch (_) {}
    }
  }

  /**
   * Show or hide the action-generating overlay within the action view.
   *
   * @param {boolean} visible - Whether the overlay is shown.
   * @param {string} [text] - Optional overlay text content.
   */
  function setOverlayVisible(visible, text) {
    if (!actionGeneratingOverlay) return;
    if (typeof text === "string") actionGeneratingOverlay.textContent = text;
    actionGeneratingOverlay.hidden = !visible;
    actionGeneratingOverlay.style.display = visible ? "flex" : "none";
  }

  /**
   * Enable or disable the action PNG download button.
   *
   * @param {boolean} enabled - Whether downloading is enabled.
   */
  function setDownloadEnabled(enabled) {
    if (!actionDownloadBtn) return;
    actionDownloadBtn.disabled = !enabled;
    actionDownloadBtn.title = enabled ? "Download PNG" : "Generate actions first";
  }

  /**
   * Enter action view.
   *
   * This sets a global flag consumed by mode.js and toggles view-specific DOM elements.
   */
  function enterActionView() {
    window.__po_action_view = true;
    refreshModeIndicator();
    document.body.classList.add("view-actions");

    const controls = document.getElementById("actionControls");
    const footer = document.getElementById("actionFooter");
    if (controls) controls.hidden = false;
    if (footer) footer.hidden = false;
  }

  /**
   * Exit action view and release overlay resources.
   */
  function exitActionView() {
    window.__po_action_view = false;
    refreshModeIndicator();
    document.body.classList.remove("view-actions");

    const controls = document.getElementById("actionControls");
    const footer = document.getElementById("actionFooter");
    if (controls) controls.hidden = true;
    if (footer) footer.hidden = true;

    setOverlayVisible(false);
    closeAnimationOverlay();
  }

  /**
   * Read the currently active action tab.
   *
   * @returns {string} Action key in ACTION_SHEET_LAYOUT.
   */
  function getActiveAction() {
    const active = actionTabs.find((b) => b.classList.contains("is-active"));
    return (active && active.dataset.action) || "walk";
  }

  /**
   * Resolve the last generated character absolute URL to a backend-readable /static path.
   *
   * Backend constraints:
   * - The API expects a path starting with "/static/" which maps to server filesystem.
   *
   * @returns {string|null} Backend /static path or null when unavailable.
   */
  function resolveCharacterUrlForApi() {
    const abs = (window.__po_last_generated_abs_url || "").trim();
    if (!abs) return null;

    try {
      const u = new URL(abs);
      if (u.pathname.startsWith("/static/")) return u.pathname;
    } catch (_) {}

    return null;
  }

  /**
   * Return the action layout for a given action name.
   *
   * @param {string} action - Action name.
   * @returns {{startRow:number, cols:number, label:string}} Layout entry.
   */
  function safeGetLayout(action) {
    return ACTION_SHEET_LAYOUT[action] || ACTION_SHEET_LAYOUT.walk;
  }

  function clamp(n, a, b) {
    return Math.max(a, Math.min(b, n));
  }

  // ---------------------------------------------------------------------------
  // Animation Overlay (Canvas)
  // ---------------------------------------------------------------------------
  function stopAnimLoop() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    lastFrameTs = 0;
  }

  /**
   * Create and attach the animation overlay UI if it does not exist.
   *
   * The overlay is constructed dynamically to avoid requiring extra HTML/CSS files.
   * Inline styles are applied to keep the overlay visually self-contained.
   *
   * @returns {boolean} Whether the overlay was successfully created or already exists.
   */
  function ensureAnimationOverlay() {
    if (overlayEl && overlayCanvas && overlayCtx) return true;

    const panel = actionPreviewImg.closest(".sprite-panel");
    if (!panel) {
      console.warn("[actions] Cannot find action panel container for overlay.");
      return false;
    }

    panel.style.position = panel.style.position || "relative";

    overlayEl = document.createElement("div");
    overlayEl.setAttribute("data-po-action-overlay", "1");
    overlayEl.style.position = "absolute";
    overlayEl.style.inset = "10px";
    overlayEl.style.display = "flex";
    overlayEl.style.flexDirection = "column";
    overlayEl.style.gap = "10px";
    overlayEl.style.padding = "10px";
    overlayEl.style.borderRadius = "14px";
    overlayEl.style.background = "rgba(15, 18, 22, 0.75)";
    overlayEl.style.backdropFilter = "blur(6px)";
    overlayEl.style.border = "1px solid rgba(255,255,255,0.12)";
    overlayEl.style.boxShadow = "0 10px 30px rgba(0,0,0,0.35)";
    overlayEl.style.zIndex = "50";

    // Header
    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.gap = "10px";

    const title = document.createElement("div");
    title.id = "poActionOverlayTitle";
    title.style.color = "rgba(255,255,255,0.95)";
    title.style.fontWeight = "700";
    title.style.letterSpacing = "0.04em";
    title.style.fontSize = "14px";
    title.textContent = "Action: WALK";

    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.setAttribute("aria-label", "Close animation preview");
    closeBtn.textContent = "✕";
    closeBtn.style.cursor = "pointer";
    closeBtn.style.border = "0";
    closeBtn.style.background = "rgba(255,255,255,0.12)";
    closeBtn.style.color = "rgba(255,255,255,0.95)";
    closeBtn.style.borderRadius = "10px";
    closeBtn.style.padding = "6px 10px";
    closeBtn.style.fontWeight = "700";
    closeBtn.addEventListener("click", () => closeAnimationOverlay());

    header.appendChild(title);
    header.appendChild(closeBtn);

    // Canvas wrap
    const canvasWrap = document.createElement("div");
    canvasWrap.style.flex = "1";
    canvasWrap.style.display = "grid";
    canvasWrap.style.placeItems = "center";
    canvasWrap.style.borderRadius = "12px";
    canvasWrap.style.background = "rgba(255,255,255,0.08)";
    canvasWrap.style.border = "1px solid rgba(255,255,255,0.10)";

    overlayCanvas = document.createElement("canvas");
    overlayCanvas.id = "poActionCanvas";
    overlayCanvas.width = 220;
    overlayCanvas.height = 220;
    overlayCanvas.style.width = "220px";
    overlayCanvas.style.height = "220px";
    overlayCanvas.style.imageRendering = "pixelated";
    overlayCanvas.style.display = "block";

    overlayCtx = overlayCanvas.getContext("2d", { alpha: true });
    if (!overlayCtx) {
      console.warn("[actions] Canvas 2D context not available.");
      overlayEl = null;
      overlayCanvas = null;
      overlayCtx = null;
      return false;
    }
    overlayCtx.imageSmoothingEnabled = false;

    canvasWrap.appendChild(overlayCanvas);

    // Controls
    const controls = document.createElement("div");
    controls.style.display = "flex";
    controls.style.alignItems = "center";
    controls.style.justifyContent = "space-between";
    controls.style.gap = "10px";
    controls.style.flexWrap = "wrap";

    // Direction pad
    const dirBox = document.createElement("div");
    dirBox.style.display = "flex";
    dirBox.style.flexDirection = "column";
    dirBox.style.gap = "6px";

    const dirLabel = document.createElement("div");
    dirLabel.textContent = "Direction";
    dirLabel.style.color = "rgba(255,255,255,0.85)";
    dirLabel.style.fontSize = "12px";

    const dirGrid = document.createElement("div");
    dirGrid.style.display = "grid";
    dirGrid.style.gridTemplateColumns = "repeat(2, auto)";
    dirGrid.style.gap = "6px";

    /**
     * Create a direction selection button.
     *
     * @param {string} dir - Direction key (W/E/S/N).
     * @returns {HTMLButtonElement} Button element.
     */
    function mkDirBtn(dir) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = dir;
      b.dataset.dir = dir;
      b.style.cursor = "pointer";
      b.style.border = "0";
      b.style.padding = "6px 10px";
      b.style.borderRadius = "10px";
      b.style.fontWeight = "700";
      b.style.letterSpacing = "0.03em";
      b.style.background = "rgba(255,255,255,0.12)";
      b.style.color = "rgba(255,255,255,0.95)";
      b.addEventListener("click", () => setOverlayDirection(dir));
      return b;
    }

    ["W", "E", "S", "N"].forEach((d) => dirGrid.appendChild(mkDirBtn(d)));
    dirBox.appendChild(dirLabel);
    dirBox.appendChild(dirGrid);

    // Playback
    const playBox = document.createElement("div");
    playBox.style.display = "flex";
    playBox.style.flexDirection = "column";
    playBox.style.gap = "6px";
    playBox.style.minWidth = "220px";

    const playRow = document.createElement("div");
    playRow.style.display = "flex";
    playRow.style.alignItems = "center";
    playRow.style.gap = "10px";

    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.id = "poPlayBtn";
    playBtn.textContent = "⏸ Pause";
    playBtn.style.cursor = "pointer";
    playBtn.style.border = "0";
    playBtn.style.padding = "6px 10px";
    playBtn.style.borderRadius = "10px";
    playBtn.style.fontWeight = "700";
    playBtn.style.background = "rgba(255,255,255,0.12)";
    playBtn.style.color = "rgba(255,255,255,0.95)";
    playBtn.addEventListener("click", () => {
      overlayPlaying = !overlayPlaying;
      updateOverlayControlsUI();
      if (overlayPlaying) startAnimLoop();
    });

    const speedWrap = document.createElement("div");
    speedWrap.style.display = "flex";
    speedWrap.style.alignItems = "center";
    speedWrap.style.gap = "8px";
    speedWrap.style.flex = "1";

    const speedLabel = document.createElement("div");
    speedLabel.textContent = "Speed";
    speedLabel.style.color = "rgba(255,255,255,0.85)";
    speedLabel.style.fontSize = "12px";
    speedLabel.style.minWidth = "44px";

    const speedInput = document.createElement("input");
    speedInput.type = "range";
    speedInput.min = "4";
    speedInput.max = "12";
    speedInput.step = "1";
    speedInput.value = String(DEFAULT_FPS);
    speedInput.style.width = "100%";
    speedInput.addEventListener("input", () => {
      overlayFps = clamp(parseInt(speedInput.value, 10) || DEFAULT_FPS, 1, 24);
      updateOverlayControlsUI();
    });

    const speedValue = document.createElement("div");
    speedValue.id = "poFpsValue";
    speedValue.textContent = `${DEFAULT_FPS} FPS`;
    speedValue.style.color = "rgba(255,255,255,0.85)";
    speedValue.style.fontSize = "12px";
    speedValue.style.minWidth = "56px";
    speedValue.style.textAlign = "right";

    speedWrap.appendChild(speedLabel);
    speedWrap.appendChild(speedInput);
    speedWrap.appendChild(speedValue);

    playRow.appendChild(playBtn);
    playRow.appendChild(speedWrap);
    playBox.appendChild(playRow);

    controls.appendChild(dirBox);
    controls.appendChild(playBox);

    overlayEl.appendChild(header);
    overlayEl.appendChild(canvasWrap);
    overlayEl.appendChild(controls);

    panel.appendChild(overlayEl);

    updateOverlayControlsUI();
    return true;
  }

  /**
   * Sync overlay UI elements with the current overlay state.
   */
  function updateOverlayControlsUI() {
    if (!overlayEl) return;

    const title = overlayEl.querySelector("#poActionOverlayTitle");
    const playBtn = overlayEl.querySelector("#poPlayBtn");
    const fpsValue = overlayEl.querySelector("#poFpsValue");

    const layout = safeGetLayout(overlayAction);
    if (title) title.textContent = `Action: ${layout.label}`;
    if (playBtn) playBtn.textContent = overlayPlaying ? "⏸ Pause" : "▶ Play";
    if (fpsValue) fpsValue.textContent = `${overlayFps} FPS`;

    // Direction highlight.
    const allDirBtns = overlayEl.querySelectorAll("button[data-dir]");
    allDirBtns.forEach((b) => {
      const isActive = (b.dataset.dir || "") === overlayDir;
      b.style.background = isActive ? "rgba(255,255,255,0.22)" : "rgba(255,255,255,0.12)";
    });

    // Sync speed slider.
    const speedInput = overlayEl.querySelector('input[type="range"]');
    if (speedInput) speedInput.value = String(overlayFps);
  }

  /**
   * Open the animation overlay for a specific action.
   *
   * @param {string} action - Action key.
   */
  function openAnimationOverlay(action) {
    overlayAction = action || "walk";
    overlayDir = overlayDir || DEFAULT_DIR;
    overlayFps = overlayFps || DEFAULT_FPS;
    overlayPlaying = true;
    frameIndex = 0;

    if (!ensureAnimationOverlay()) return;

    overlayEl.style.display = "flex";
    updateOverlayControlsUI();
    startAnimLoop();
  }

  /**
   * Close and dispose the animation overlay.
   */
  function closeAnimationOverlay() {
    stopAnimLoop();
    if (overlayEl) overlayEl.remove();
    overlayEl = null;
    overlayCanvas = null;
    overlayCtx = null;
  }

  /**
   * Set the overlay direction and restart playback state.
   *
   * @param {string} dir - Direction key (W/E/S/N).
   */
  function setOverlayDirection(dir) {
    if (!Object.prototype.hasOwnProperty.call(DIR_ROW_OFFSET, dir)) return;
    overlayDir = dir;
    frameIndex = 0;
    updateOverlayControlsUI();
    if (overlayPlaying) startAnimLoop();
  }

  /**
   * Draw the current frame of the selected action/direction into the overlay canvas.
   */
  function drawFrame() {
    if (!overlayCtx || !overlayCanvas) return;

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const img = actionPreviewImg;
    if (!img || !img.complete || !img.naturalWidth) {
      overlayCtx.font = "14px sans-serif";
      overlayCtx.fillStyle = "rgba(255,255,255,0.85)";
      overlayCtx.textAlign = "center";
      overlayCtx.fillText("No sheet loaded", overlayCanvas.width / 2, overlayCanvas.height / 2);
      return;
    }

    const layout = safeGetLayout(overlayAction);
    const dirOff = DIR_ROW_OFFSET[overlayDir] ?? DIR_ROW_OFFSET[DEFAULT_DIR];

    const srcCol = (frameIndex % layout.cols);
    const srcRow = (layout.startRow + dirOff);

    const sx = srcCol * TILE_SIZE;
    const sy = srcRow * TILE_SIZE;

    const scale = 3; // 64*3=192
    const dw = TILE_SIZE * scale;
    const dh = TILE_SIZE * scale;
    const dx = Math.floor((overlayCanvas.width - dw) / 2);
    const dy = Math.floor((overlayCanvas.height - dh) / 2);

    overlayCtx.imageSmoothingEnabled = false;
    overlayCtx.drawImage(img, sx, sy, TILE_SIZE, TILE_SIZE, dx, dy, dw, dh);
  }

  /**
   * Start the requestAnimationFrame loop for the overlay.
   *
   * The loop is single-instance guarded by rafId.
   */
  function startAnimLoop() {
    if (!overlayEl || !overlayCtx || !overlayCanvas) return;
    if (rafId) return;

    lastFrameTs = 0;

    const tick = (ts) => {
      rafId = requestAnimationFrame(tick);

      if (!overlayPlaying) {
        drawFrame();
        return;
      }

      const frameDur = 1000 / clamp(overlayFps, 1, 24);

      if (!lastFrameTs) lastFrameTs = ts;
      const dt = ts - lastFrameTs;

      if (dt >= frameDur) {
        const steps = Math.floor(dt / frameDur);
        frameIndex += Math.max(1, steps);
        lastFrameTs += steps * frameDur;
      }

      drawFrame();
    };

    rafId = requestAnimationFrame(tick);
  }

  // When the full sheet finishes loading, refresh overlay frame immediately.
  actionPreviewImg.addEventListener("load", () => {
    if (overlayEl) {
      frameIndex = 0;
      lastFrameTs = 0;
      drawFrame();
    }
  });

  // ---------------------------------------------------------------------------
  // Core: request one full actions sheet (once per character)
  // ---------------------------------------------------------------------------
  /**
   * Request the actions sheet from the backend only once per character.
   *
   * The returned PNG is cached by character_url to avoid repeated backend calls.
   *
   * @returns {Promise<void>}
   */
  async function requestActionsSheetOnce() {
    if (isGenerating) return;

    const characterUrl = resolveCharacterUrlForApi();
    if (!characterUrl) {
      alert("No generated character found. Please generate a character first.");
      return;
    }

    // Cached for this character: reuse.
    if (cachedSheetUrls[characterUrl]) {
      actionPreviewImg.src = cachedSheetUrls[characterUrl];
      setDownloadEnabled(true);
      return;
    }

    isGenerating = true;
    updateGenerateActionsState();
    setDownloadEnabled(false);
    setOverlayVisible(true, "Generating...");

    try {
      const payload = {
        character_url: characterUrl,
        ddim_steps: 10,
        cfg_scale: 1.0,
        seed: 0,
      };

      const resp = await fetch(ACTIONS_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();
      if (!data || data.ok !== true || !data.image_url) {
        throw new Error("Bad API response");
      }

      const absUrl = `${API_BASE}${data.image_url}`;
      const finalUrl = `${absUrl}?t=${Date.now()}`;

      cachedSheetUrls[characterUrl] = finalUrl;
      actionPreviewImg.src = finalUrl;
      setDownloadEnabled(true);
    } catch (err) {
      console.error("[actions] generation failed:", err);
      alert("Action generation failed. Please try again.");
    } finally {
      isGenerating = false;
      setOverlayVisible(false);
      updateGenerateActionsState();
    }
  }

  // ---------------------------------------------------------------------------
  // Download PNG
  // ---------------------------------------------------------------------------
  actionDownloadBtn.addEventListener("click", async () => {
    if (!actionPreviewImg || !actionPreviewImg.src) return;

    try {
      const cleanUrl = actionPreviewImg.src.split("?")[0];
      const resp = await fetch(cleanUrl, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const blob = await resp.blob();

      const d = new Date();
      const pad = (n) => String(n).padStart(2, "0");
      const filename =
        `pixel_odyssey_actions_` +
        `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
        `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.png`;

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      console.error("[actions] download failed:", err);
      alert("Download failed. Please try again.");
    }
  });

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------
  generateActionsBtn.addEventListener("click", async () => {
    if (generateActionsBtn.disabled) return;

    enterActionView();

    // Ensure the full sheet exists (only once).
    await requestActionsSheetOnce();

    // After the sheet is present, open the overlay using the active tab.
    const a = getActiveAction();
    lastAction = a;
    openAnimationOverlay(a);
  });

  backBtn.addEventListener("click", exitActionView);

  // Tabs do not call the backend. They only switch the overlay animation state.
  actionTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      actionTabs.forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");

      const a = btn.dataset.action || "walk";
      lastAction = a;

      if (document.body.classList.contains("view-actions")) {
        openAnimationOverlay(a);
      }
    });
  });

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  setOverlayVisible(false);
  setDownloadEnabled(false);
  updateGenerateActionsState();

  const obs = new MutationObserver(updateGenerateActionsState);
  obs.observe(previewImage, { attributes: true, attributeFilter: ["src"] });

  window.addEventListener("focus", updateGenerateActionsState);
});
