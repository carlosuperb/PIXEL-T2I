# webapp/backend/api_server.py
#
# FastAPI backend for PIXEL-T2I web demo.
#
# Endpoints:
# - POST /api/generate         : unconditional (empty prompt) or text-conditional (prompt + CLIP)
# - POST /api/generate_actions : image-conditional actions sheet from a generated 4-view PNG
# - GET  /api/download_batch   : zip download for a generated batch_id
# - POST /api/clear_cache      : delete generated PNGs under the cache directory
#
# Models are loaded once on startup. Generated outputs are cached under webapp/web_cache/
# and served via /static (StaticFiles mount).

import sys
import time
import zipfile
import threading
import logging
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Path setup (project root)
#
# Ensures project-root imports work when uvicorn is started from webapp/backend.
# -----------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent                  # .../webapp/backend
PROJECT_ROOT = BACKEND_DIR.parents[1]                          # .../PIXEL-T2I
WEBAPP_DIR = PROJECT_ROOT / "webapp"                           # .../webapp

# UNCONDITIONAL model directory
UNCOND_DIR = PROJECT_ROOT / "models" / "pixel_unconditional"   # .../models/pixel_unconditional
CHECKPOINT_UNCOND_DEFAULT = UNCOND_DIR / "checkpoints" / "model_best.pt"

# TEXT-CONDITIONAL model directory
TEXT_DIR = PROJECT_ROOT / "models" / "pixel_text_conditional"  # .../models/pixel_text_conditional
CHECKPOINT_TEXT_DEFAULT = TEXT_DIR / "checkpoints" / "model_best.pt"

# IMAGE-CONDITIONAL model directory
IMG_DIR = PROJECT_ROOT / "models" / "pixel_image_conditional"  # .../models/pixel_image_conditional
CHECKPOINT_IMG_DIR_DEFAULT = IMG_DIR / "checkpoints"           # directory that contains unet_best.pt + char_encoder_best.pt

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Imports: UNCONDITIONAL inference utilities
# -----------------------------------------------------------------------------
from models.pixel_unconditional.inference import (  # noqa: E402
    load_checkpoint as load_checkpoint_uncond,
    prepare_noise_schedule as prepare_noise_schedule_uncond,
    ddim_sample as ddim_sample_uncond,
    save_images as save_images_uncond,
)


# -----------------------------------------------------------------------------
# Imports: TEXT-CONDITIONAL inference utilities
# -----------------------------------------------------------------------------
from models.pixel_text_conditional.inference import (  # noqa: E402
    load_checkpoint as load_checkpoint_text,
    prepare_noise_schedule as prepare_noise_schedule_text,
    encode_text as encode_text_textcond,
    ddim_sample as ddim_sample_text,
    save_images as save_images_text,
)


# -----------------------------------------------------------------------------
# Imports: IMAGE-CONDITIONAL inference utilities (actions from 4-view)
# -----------------------------------------------------------------------------
from models.pixel_image_conditional.image_inference import (  # noqa: E402
    load_models as load_models_img,
    prepare_noise_schedule as prepare_noise_schedule_img,
    load_fourview_rgba as load_fourview_rgba_img,
    generate_full_actions_sheet as generate_full_actions_sheet_img,
    tensor_to_uint8_rgba as tensor_to_uint8_rgba_img,
    save_rgba_tensor_as_png as save_rgba_tensor_as_png_img,
)


# -----------------------------------------------------------------------------
# Static output (served by backend)
#
# /static -> CACHE_DIR
# Generated PNGs are written under GEN_DIR and become accessible via /static/generated/...
# -----------------------------------------------------------------------------
CACHE_DIR = WEBAPP_DIR / "web_cache"            # .../webapp/web_cache
GEN_DIR = CACHE_DIR / "generated"               # .../webapp/web_cache/generated

# Keep only the latest N images in GEN_DIR (cache policy).
# Cleanup runs after successful generation endpoints.
MAX_CACHE_IMAGES = 100

app = FastAPI()

# Allow frontend (localhost:5500) to call backend (127.0.0.1:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
GEN_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(CACHE_DIR)), name="static")


# -----------------------------------------------------------------------------
# Generation lock (single GPU critical section)
#
# Prevents overlapping requests from concurrently allocating large GPU tensors.
# -----------------------------------------------------------------------------
GEN_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# System-level prompt anchors (TEXT-CONDITIONAL)
#
# Style/layout anchors injected during text-conditioned inference to stabilize
# generation and enforce a consistent four-view sprite layout.
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_PREFIX = "lpc-style pixel art character, "

SYSTEM_PROMPT_SUFFIX = (
    ", 4-view sprite sheet (front/back/left/right), "
    "sharp small pixel art, hard edges, no anti-aliasing"
)


# -----------------------------------------------------------------------------
# UNCONDITIONAL micro-batching (server-side)
#
# Large unconditional batch requests are split into smaller chunks to reduce VRAM load.
# -----------------------------------------------------------------------------
UNCOND_INTERNAL_BATCH = 2  # tune: 2/4/8 depending on GPU stability


# -----------------------------------------------------------------------------
# Request schema
# -----------------------------------------------------------------------------
class GenerateReq(BaseModel):
    """
    /api/generate request payload.

    Routing:
      - prompt empty/whitespace -> unconditional generation
      - prompt non-empty        -> text-conditional generation (CLIP text conditioning)

    Server-side policy:
      - text-conditional generation forces num_samples=1
      - text-conditional generation forces cfg_scale=1.0
    """

    prompt: str | None = ""
    num_samples: int = 1
    ddim_steps: int = 10
    cfg_scale: float = 1.0
    checkpoint: str | None = None  # reserved for future override


class GenerateActionsReq(BaseModel):
    """
    /api/generate_actions request payload.

    character_url:
      URL returned by /api/generate (e.g. /static/generated/generated_xxx_0000.png).
      Must resolve to a backend-served static path under /static/ so the server can read it.
    """

    character_url: str
    ddim_steps: int = 10
    cfg_scale: float = 1.0
    seed: int = 0
    checkpoint: str | None = None  # optional override (dir or file), reserved


# -----------------------------------------------------------------------------
# Global state: load models once at startup
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- UNCONDITIONAL (UNet only) ----------
UNET_UNCOND = None
CONFIG_UNCOND = None
NOISE_SCHEDULE_UNCOND = None

# ---------- TEXT-CONDITIONAL (UNet + CLIP encoder/tokenizer) ----------
UNET_TEXT = None
CONFIG_TEXT = None
NOISE_SCHEDULE_TEXT = None
TEXT_ENCODER = None
TOKENIZER = None

# ---------- IMAGE-CONDITIONAL (UNet + Char Encoder) ----------
UNET_IMG = None
CHAR_ENCODER_IMG = None
CONFIG_IMG = None
NOISE_SCHEDULE_IMG = None


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------
def _cleanup_generated_cache(max_keep: int = MAX_CACHE_IMAGES) -> int:
    """
    Keep only the latest 'max_keep' PNG files in GEN_DIR.
    Returns the number of deleted files.
    """
    if max_keep <= 0:
        return 0

    files = sorted(
        GEN_DIR.glob("*.png"),
        key=lambda p: p.stat().st_mtime,   # sort by modified time (oldest first)
    )

    if len(files) <= max_keep:
        return 0

    to_delete = files[: len(files) - max_keep]
    deleted = 0
    for p in to_delete:
        try:
            p.unlink()
            deleted += 1
        except Exception:
            # Best-effort cleanup; ignore failures (e.g., file in use)
            pass
    return deleted


def _clear_generated_cache() -> int:
    """
    Delete all PNG files under GEN_DIR.
    Returns the number of deleted files.
    """
    deleted = 0
    for p in GEN_DIR.glob("*.png"):
        try:
            p.unlink()
            deleted += 1
        except Exception:
            # Best-effort cleanup; ignore failures (e.g., file in use)
            pass
    return deleted


# -----------------------------------------------------------------------------
# Startup: load models and precompute noise schedules
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _startup():
    """
    Load checkpoints and precompute noise schedules once.

    UNCONDITIONAL:
      - load UNet
      - prepare noise schedule

    TEXT-CONDITIONAL:
      - load UNet + CLIP text_encoder + tokenizer
      - prepare noise schedule

    IMAGE-CONDITIONAL:
      - load UNet + character encoder
      - prepare noise schedule
    """
    global UNET_UNCOND, CONFIG_UNCOND, NOISE_SCHEDULE_UNCOND
    global UNET_TEXT, CONFIG_TEXT, NOISE_SCHEDULE_TEXT
    global TEXT_ENCODER, TOKENIZER
    global UNET_IMG, CHAR_ENCODER_IMG, CONFIG_IMG, NOISE_SCHEDULE_IMG

    # -------------------- UNCONDITIONAL --------------------
    ckpt_u = Path(CHECKPOINT_UNCOND_DEFAULT)
    if not ckpt_u.exists():
        raise FileNotFoundError(f"Unconditional checkpoint not found: {ckpt_u}")

    UNET_UNCOND, CONFIG_UNCOND = load_checkpoint_uncond(ckpt_u, DEVICE)
    NOISE_SCHEDULE_UNCOND = prepare_noise_schedule_uncond(CONFIG_UNCOND, DEVICE)

    # -------------------- TEXT-CONDITIONAL --------------------
    ckpt_t = Path(CHECKPOINT_TEXT_DEFAULT)
    if not ckpt_t.exists():
        raise FileNotFoundError(
            f"Text-conditional checkpoint not found: {ckpt_t}\n"
            "Update TEXT_DIR / CHECKPOINT_TEXT_DEFAULT in api_server.py to match the local folder."
        )

    UNET_TEXT, TEXT_ENCODER, TOKENIZER, CONFIG_TEXT = load_checkpoint_text(ckpt_t, DEVICE)
    NOISE_SCHEDULE_TEXT = prepare_noise_schedule_text(CONFIG_TEXT, DEVICE)

    # -------------------- IMAGE-CONDITIONAL --------------------
    ckpt_img_dir = Path(CHECKPOINT_IMG_DIR_DEFAULT)
    if not ckpt_img_dir.exists():
        raise FileNotFoundError(
            f"Image-conditional checkpoint dir not found: {ckpt_img_dir}\n"
            "Expected split checkpoints inside:\n"
            "  - unet_best.pt\n"
            "  - char_encoder_best.pt"
        )

    # Reuse image_inference.py loader (supports passing a directory for split ckpts)
    class _ArgsImg:
        checkpoint = str(ckpt_img_dir)

    CONFIG_IMG, UNET_IMG, CHAR_ENCODER_IMG = load_models_img(_ArgsImg(), DEVICE)
    NOISE_SCHEDULE_IMG = prepare_noise_schedule_img(CONFIG_IMG, DEVICE)

    # -------------------- Logs --------------------
    logger.info("startup | device=%s", DEVICE)
    logger.info("startup | uncond_ckpt=%s", ckpt_u)
    logger.info("startup | text_ckpt=%s", ckpt_t)
    logger.info("startup | img_cond_ckpt_dir=%s", ckpt_img_dir)
    logger.info("startup | cache_dir=%s", CACHE_DIR)
    logger.info("startup | gen_dir=%s", GEN_DIR)
    logger.info("startup | cache_policy=max_keep=%d", MAX_CACHE_IMAGES)


# -----------------------------------------------------------------------------
# API surface
#
# POST /api/generate
#   - Input : GenerateReq
#   - Output: {first_image_url, preview_urls, time_sec, batch{enabled,id,count}, mode}
#
# POST /api/generate_actions
#   - Input : GenerateActionsReq (character_url under /static/)
#   - Output: {ok, image_url, time_sec}
#
# GET  /api/download_batch?batch_id=...
#   - Output: ZIP archive of {batch_id}_*.png
#
# POST /api/clear_cache
#   - Output: {deleted}
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
@app.post("/api/clear_cache")
def clear_cache():
    """
    Delete all generated PNG files under the cache directory.

    Returns:
      - deleted: number of files removed
    """
    deleted = _clear_generated_cache()
    logger.info("cache | clear_cache deleted=%d", deleted)
    return {"deleted": deleted}


@app.get("/api/download_batch")
def download_batch(batch_id: str):
    """
    Download all generated PNG files associated with a batch_id as a ZIP archive.

    Parameters:
      - batch_id: identifier prefix used when saving generated images

    Returns:
      - application/zip response containing files named {batch_id}_*.png
    """
    batch_id = (batch_id or "").strip()
    if not batch_id:
        return {"error": "Missing batch_id."}

    files = sorted(GEN_DIR.glob(f"{batch_id}_*.png"))
    if not files:
        return {"error": f"No files found for batch_id={batch_id}."}

    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)

    zip_bytes = buf.getvalue()
    headers = {"Content-Disposition": f'attachment; filename="{batch_id}.zip"'}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)


@app.post("/api/generate")
def generate(req: GenerateReq):
    """
    Generate sprites and return URLs under /static/.

    Routing:
      - prompt empty        -> unconditional generation
      - prompt non-empty    -> text-conditional generation

    Output:
      - first_image_url : first image for single preview
      - preview_urls    : up to 4 images for batch preview grid
      - batch           : {enabled, id, count}
    """
    global UNET_UNCOND, CONFIG_UNCOND, NOISE_SCHEDULE_UNCOND
    global UNET_TEXT, CONFIG_TEXT, NOISE_SCHEDULE_TEXT
    global TEXT_ENCODER, TOKENIZER

    # Single GPU critical section
    with GEN_LOCK:
        prompt = (req.prompt or "").strip()
        steps = max(1, min(int(req.ddim_steps), 200))

        n = int(req.num_samples) if req.num_samples is not None else 1
        n = max(1, min(n, 50))

        cfg = float(req.cfg_scale) if req.cfg_scale is not None else 1.0
        cfg = max(1.0, min(cfg, 20.0))

        t0 = time.time()

        samples = None
        context = None

        # Defaults (overridden per-branch)
        batch_id = f"generated_{uuid4().hex[:8]}"
        batch_enabled = False
        batch_count = 1

        try:
            # -----------------------------------------------------------------
            # Branch A: TEXT-CONDITIONAL
            # -----------------------------------------------------------------
            if prompt:
                # Server-side policy: disable batch generation (force single-sample)
                n = 1

                # Server-side policy: enforce fixed CFG scale for stability
                cfg_effective = 1.0

                # Inject system-level style/layout anchors
                full_prompt = (
                    SYSTEM_PROMPT_PREFIX
                    + prompt.strip().rstrip(",")
                    + SYSTEM_PROMPT_SUFFIX
                )

                prompts = [full_prompt]

                shape = (
                    1,
                    CONFIG_TEXT["in_channels"],
                    CONFIG_TEXT["image_size"],
                    CONFIG_TEXT["image_size"],
                )

                # Encode text -> context (B, seq_len, embed_dim)
                context = encode_text_textcond(prompts, TOKENIZER, TEXT_ENCODER, CONFIG_TEXT, DEVICE)

                samples = ddim_sample_text(
                    UNET_TEXT,
                    shape,
                    NOISE_SCHEDULE_TEXT,
                    DEVICE,
                    context=context,
                    cfg_scale=cfg_effective,
                    ddim_steps=steps,
                    eta=0.0,
                    show_progress=False,
                )

                save_images_text(
                    samples,
                    GEN_DIR,
                    prefix=batch_id,
                    save_individual=True,
                    save_grid=False,
                    start_index=0,
                )

                batch_enabled = False
                batch_count = 1

            # -----------------------------------------------------------------
            # Branch B: UNCONDITIONAL
            # -----------------------------------------------------------------
            else:
                batch_enabled = n > 1
                batch_count = n

                generated = 0
                chunk_id = 0

                while generated < n:
                    cur = min(UNCOND_INTERNAL_BATCH, n - generated)
                    chunk_id += 1

                    shape = (
                        cur,
                        CONFIG_UNCOND["in_channels"],
                        CONFIG_UNCOND["image_size"],
                        CONFIG_UNCOND["image_size"],
                    )

                    samples = ddim_sample_uncond(
                        UNET_UNCOND,
                        shape,
                        NOISE_SCHEDULE_UNCOND,
                        DEVICE,
                        ddim_steps=steps,
                        eta=0.0,
                        show_progress=False,
                    )

                    save_images_uncond(
                        samples,
                        GEN_DIR,
                        prefix=batch_id,
                        save_individual=True,
                        save_grid=False,
                        start_index=generated,
                    )

                    # Free GPU memory between micro-batches
                    del samples
                    samples = None
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

                    generated += cur

            # -----------------------------------------------------------------
            # Common response payload
            # -----------------------------------------------------------------
            first_image_url = f"/static/generated/{batch_id}_0000.png"

            preview_k = min(4, batch_count)
            preview_urls = [f"/static/generated/{batch_id}_{i:04d}.png" for i in range(preview_k)]

            deleted = _cleanup_generated_cache(MAX_CACHE_IMAGES)
            if deleted > 0:
                logger.info("cache | cleanup deleted=%d max_keep=%d", deleted, MAX_CACHE_IMAGES)

            mode = "text" if prompt else "unconditional"
            logger.info(
                "generate | mode=%s batch_id=%s count=%d steps=%d time_sec=%.3f",
                mode,
                batch_id,
                batch_count if not prompt else 1,
                steps,
                time.time() - t0,
            )

            return {
                "first_image_url": first_image_url,
                "preview_urls": preview_urls,
                "time_sec": round(time.time() - t0, 3),
                "batch": {
                    "enabled": batch_enabled if not prompt else False,
                    "id": batch_id,
                    "count": batch_count if not prompt else 1,
                },
                "mode": mode,
            }

        finally:
            # Safety cleanup: release tensors after each request
            if context is not None:
                try:
                    del context
                except Exception:
                    pass

            if samples is not None:
                try:
                    del samples
                except Exception:
                    pass

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


@app.post("/api/generate_actions")
def generate_actions(req: GenerateActionsReq):
    """
    Generate an actions spritesheet (12x9 tiles => 768x576) conditioned on a 4-view PNG.

    Input:
      - character_url: backend-served /static/... url (maps to CACHE_DIR on disk)

    Output:
      - image_url: PNG under /static/generated/
    """
    global UNET_IMG, CHAR_ENCODER_IMG, CONFIG_IMG, NOISE_SCHEDULE_IMG

    with GEN_LOCK:
        char_url = (req.character_url or "").strip()
        if not char_url:
            return {"ok": False, "error": "Missing character_url."}

        # Must be a file served by the StaticFiles mount: /static -> CACHE_DIR
        if not char_url.startswith("/static/"):
            return {"ok": False, "error": "character_url must start with /static/."}

        # Map URL -> filesystem path
        rel = char_url[len("/static/"):]  # e.g. generated/xxx.png
        char_path = CACHE_DIR / rel

        if not char_path.exists():
            return {"ok": False, "error": f"character file not found: {char_url}"}

        steps = max(1, min(int(req.ddim_steps), 200))
        cfg = float(req.cfg_scale) if req.cfg_scale is not None else 1.0
        cfg = max(1.0, min(cfg, 20.0))

        seed = int(req.seed) if req.seed is not None else 0

        t0 = time.time()

        # Seed for reproducibility
        torch.manual_seed(seed)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        fourview = None
        sheet = None

        try:
            fourview = load_fourview_rgba_img(char_path, size=int(CONFIG_IMG["fourview_size"])).to(DEVICE)

            sheet = generate_full_actions_sheet_img(
                unet=UNET_IMG,
                char_encoder=CHAR_ENCODER_IMG,
                fourview=fourview,
                noise_schedule=NOISE_SCHEDULE_IMG,
                device=DEVICE,
                cfg_scale=cfg,
                ddim_steps=steps,
                eta=0.0,
                config=CONFIG_IMG,
                show_progress=False,
            )[0]  # (4,768,576) in [-1,1]

            sheet_uint8 = tensor_to_uint8_rgba_img(sheet)

            batch_id = f"actions_{uuid4().hex[:8]}"
            out_path = GEN_DIR / f"{batch_id}_0000.png"
            save_rgba_tensor_as_png_img(sheet_uint8, out_path)

            deleted = _cleanup_generated_cache(MAX_CACHE_IMAGES)
            if deleted > 0:
                logger.info("cache | cleanup deleted=%d max_keep=%d", deleted, MAX_CACHE_IMAGES)

            logger.info(
                "generate_actions | batch_id=%s steps=%d cfg=%.3f seed=%d time_sec=%.3f",
                batch_id,
                steps,
                cfg,
                seed,
                time.time() - t0,
            )

            return {
                "ok": True,
                "image_url": f"/static/generated/{batch_id}_0000.png",
                "time_sec": round(time.time() - t0, 3),
            }

        finally:
            # Safety cleanup
            if fourview is not None:
                try:
                    del fourview
                except Exception:
                    pass
            if sheet is not None:
                try:
                    del sheet
                except Exception:
                    pass

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
