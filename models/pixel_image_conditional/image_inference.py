"""
image_inference.py - Image-Conditional Sprite Actions Generation (Tile-Conditional Diffusion)

This script generates LPC-style action spritesheets (RGBA) conditioned on a 4-view character image.
The model predicts ONE 64x64 tile at a time, conditioned by:
  - character features from a 4-view encoder (char_encoder)
  - a frame_id (0..107) injected into UNet time embedding (frame embedding)

It then assembles tiles into a full actions sheet with layout:
  - walk:   4 rows x 9 cols
  - thrust: 4 rows x 8 cols
  - slash:  4 rows x 6 cols
Total sheet layout: 12 rows x 9 cols => 768 x 576 (HxW) for 64px tiles.

Features:
- Single-image inference and batch inference (folder or filelist)
- Optional tile-level batch generation for faster inference on high-memory GPUs
- Adjustable CFG scale and DDIM steps
- Select checkpoint (from models/pixel_image_conditional/checkpoints)
- Saves outputs to models/pixel_image_conditional/temp_outputs by default
- Outputs are named sequentially: image_conditional_0001.png, image_conditional_0002.png, ...

Usage:
    # Single image (loads best checkpoint automatically)
    python image_inference.py --input path/to/4view.png

    # Batch: input directory
    python image_inference.py --input_dir path/to/4view_images

    # Batch: filelist (one path per line)
    python image_inference.py --filelist inputs.txt

    # Use a specific checkpoint_epoch_*.pt (expects both unet + char_encoder inside)
    python image_inference.py --checkpoint models/pixel_image_conditional/checkpoints/checkpoint_epoch_10.pt

    # Control sampling
    python image_inference.py --input 4view.png --cfg_scale 2.0 --ddim_steps 50 --seed 123

    # Enable fast tile-level batch inference (requires more GPU memory)
    python image_inference.py --input 4view.png --tile_batch_size 64

Outputs:
    models/pixel_image_conditional/temp_outputs/
        image_conditional_0001.png
        image_conditional_0002.png
        ...
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_dir: Path) -> Path:
    """
    Robustly locate project root by walking upwards until a directory
    containing 'models' is found.

    This avoids fragile assumptions like SCRIPT_DIR.parents[2], which
    can break if the project is nested (e.g., .../COMP30040 Third Year Project/PIXEL-T2I/...).
    """
    cur = start_dir
    for _ in range(12):  # safety limit
        if (cur / "models").exists() and (cur / "models").is_dir():
            return cur
        cur = cur.parent
    raise RuntimeError(
        f"Could not find project root from: {start_dir}\n"
        "Expected a parent directory that contains a 'models/' folder."
    )


# If this file is located at: PIXEL-T2I/models/pixel_image_conditional/image_inference.py
PROJECT_ROOT = find_project_root(SCRIPT_DIR)  # .../PIXEL-T2I

MODELS_DIR = PROJECT_ROOT / "models" / "pixel_image_conditional"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
DEFAULT_OUTPUT_DIR = MODELS_DIR / "temp_outputs"


# ============================================================
# Section 1: Model Architecture (same blocks as training)
# ============================================================
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for timestep t."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for image conditioning.
    Q from actions features, K/V from character features (spatial).
    """
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(32, channels)
        self.context_norm = nn.GroupNorm(32, context_dim)

        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv2d(context_dim, channels, kernel_size=1)
        self.to_v = nn.Conv2d(context_dim, channels, kernel_size=1)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        context = self.context_norm(context)

        q = self.to_q(x)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        k = self.to_k(context)
        v = self.to_v(context)
        _, _, h, w = k.shape
        k = k.reshape(B, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.to_out(out)
        return out + residual


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding and optional cross-attention.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_cross_attn: bool = False,
        context_dim: int | None = None,
    ):
        super().__init__()

        self.use_cross_attn = use_cross_attn

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

        if use_cross_attn:
            assert context_dim is not None, "context_dim must be provided for cross-attention"
            self.cross_attn = CrossAttentionBlock(out_channels, context_dim)
        else:
            self.cross_attn = None

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_emb = self.act(time_emb)
        time_emb = self.time_emb_proj(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)

        return h + residual


class AttentionBlock(nn.Module):
    """Self-attention block for spatial features."""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.proj(out)
        return out + residual


class Downsample(nn.Module):
    """Spatial downsampling by factor of 2."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling by factor of 2."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SpriteCharacterEncoder(nn.Module):
    """
    Character encoder for 4-view sprite images.
    Input:  [B, 4, 128, 128]
    Output: [B, 512, 16, 16]
    """
    def __init__(self, in_channels: int = 4, channel_progression=(64, 128, 256, 512), output_dim: int = 512):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev_ch = in_channels

        for i, out_ch in enumerate(channel_progression):
            stride = 2 if i < 3 else 1  # 128->64->32->16->16
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                    nn.GroupNorm(num_groups=min(32, max(1, out_ch // 4)), num_channels=out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=min(32, max(1, out_ch // 4)), num_channels=out_ch),
                    nn.SiLU(),
                )
            )
            prev_ch = out_ch

        self.final_proj = nn.Conv2d(prev_ch, output_dim, 1) if prev_ch != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.final_proj(x)


class UNet(nn.Module):
    """
    UNet for image-conditional diffusion models (tile-level).

    Generates ONE tile conditioned on:
      - character feature map (spatial)
      - frame_id (cond_ids) injected into time embedding
    """
    def __init__(self, config: dict):
        super().__init__()

        in_channels = config["in_channels"]
        out_channels = config.get("out_channels", in_channels)
        model_channels = config["unet_channels"]
        channel_mult = config["channel_mult"]
        num_res_blocks = config["num_res_blocks"]
        attention_resolutions = config["attention_resolutions"]
        dropout = config["dropout"]
        num_heads = config["num_heads"]

        use_cross_attn = config.get("use_cross_attention", True)
        context_dim = config.get("char_feature_dim", 512)

        time_emb_dim = model_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        num_frames = int(config.get("num_frames", 108))  # +1 null id for CFG
        self.frame_emb = nn.Embedding(num_frames + 1, time_emb_dim)

        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        tile_size = int(config["tile_size"])
        current_h, current_w = tile_size, tile_size

        self.encoder_blocks = nn.ModuleList()
        current_channels = model_channels
        self.encoder_out_channels = [model_channels]

        for level, mult in enumerate(channel_mult):
            out_ch_level = model_channels * mult

            for _ in range(num_res_blocks):
                block_layers = nn.ModuleList()
                block_layers.append(
                    ResidualBlock(
                        current_channels, out_ch_level, time_emb_dim,
                        dropout, use_cross_attn=use_cross_attn, context_dim=context_dim
                    )
                )
                current_channels = out_ch_level

                avg_resolution = (current_h + current_w) // 2
                if avg_resolution in attention_resolutions:
                    block_layers.append(AttentionBlock(current_channels, num_heads))

                self.encoder_blocks.append(block_layers)
                self.encoder_out_channels.append(current_channels)

            if level != len(channel_mult) - 1:
                self.encoder_blocks.append(nn.ModuleList([Downsample(current_channels)]))
                current_h //= 2
                current_w //= 2
                self.encoder_out_channels.append(current_channels)

        self.mid_block1 = ResidualBlock(
            current_channels, current_channels, time_emb_dim,
            dropout, use_cross_attn=use_cross_attn, context_dim=context_dim
        )
        self.mid_attn = AttentionBlock(current_channels, num_heads)
        self.mid_block2 = ResidualBlock(
            current_channels, current_channels, time_emb_dim,
            dropout, use_cross_attn=use_cross_attn, context_dim=context_dim
        )

        self.decoder_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch_level = model_channels * mult

            for i in range(num_res_blocks + 1):
                skip_channels = self.encoder_out_channels.pop()
                block_layers = nn.ModuleList()
                block_layers.append(
                    ResidualBlock(
                        current_channels + skip_channels, out_ch_level, time_emb_dim,
                        dropout, use_cross_attn=use_cross_attn, context_dim=context_dim
                    )
                )
                current_channels = out_ch_level

                avg_resolution = (current_h + current_w) // 2
                if avg_resolution in attention_resolutions:
                    block_layers.append(AttentionBlock(current_channels, num_heads))

                self.decoder_blocks.append(block_layers)

                if level != 0 and i == num_res_blocks:
                    self.decoder_blocks.append(nn.ModuleList([Upsample(current_channels)]))
                    current_h *= 2
                    current_w *= 2

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None = None, cond_ids: torch.Tensor | None = None) -> torch.Tensor:
        time_emb = self.time_mlp(t)

        if cond_ids is not None:
            time_emb = time_emb + self.frame_emb(cond_ids)

        h = self.conv_in(x)

        encoder_features = [h]
        for block_layers in self.encoder_blocks:
            for layer in block_layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb, context)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Downsample):
                    h = layer(h)
            encoder_features.append(h)

        h = self.mid_block1(h, time_emb, context)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb, context)

        for block_layers in self.decoder_blocks:
            if any(isinstance(layer, ResidualBlock) for layer in block_layers):
                skip = encoder_features.pop()
                h = torch.cat([h, skip], dim=1)

            for layer in block_layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb, context)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Upsample):
                    h = layer(h)

        return self.conv_out(h)


# ============================================================
# Section 2: Diffusion schedule utilities
# ============================================================
def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    device: str = "cpu"
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


def prepare_noise_schedule(config: dict, device: torch.device) -> dict:
    timesteps = int(config.get("timesteps", 1000))
    beta_start = float(config.get("beta_start", 0.0001))
    beta_end = float(config.get("beta_end", 0.02))

    betas = linear_beta_schedule(timesteps, beta_start, beta_end, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        "timesteps": timesteps,
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
    }


# ============================================================
# Section 3: Image IO + transforms
# ============================================================
def load_fourview_rgba(path: Path, size: int = 128) -> torch.Tensor:
    """
    Load a 4-view image and convert to model input tensor in [-1, 1].
    Returns: (1, 4, size, size)
    """
    img = Image.open(path).convert("RGBA")
    img = img.resize((size, size), resample=Image.NEAREST)

    arr = np.array(img, dtype=np.uint8)        # (H, W, 4)
    x = torch.from_numpy(arr).permute(2, 0, 1) # (4, H, W) uint8
    x = x.float() / 255.0                      # [0,1]
    x = x * 2.0 - 1.0                          # [-1,1]
    return x.unsqueeze(0)                      # (1,4,H,W)


def tensor_to_uint8_rgba(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor in [-1,1] CHW to uint8 [0,255] CHW.
    """
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    x = (x * 255.0).round().to(torch.uint8)
    return x


def save_rgba_tensor_as_png(x_chw_uint8: torch.Tensor, out_path: Path):
    """
    Save CHW uint8 RGBA tensor to PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_hwc = x_chw_uint8.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(x_hwc, mode="RGBA").save(out_path)


# ============================================================
# Section 4: DDIM sampling (tile-level + assembly)
# ============================================================
@torch.no_grad()
def ddim_sample_tile(
    unet: UNet,
    char_encoder: SpriteCharacterEncoder,
    fourview: torch.Tensor,       # (B,4,128,128) in [-1,1]
    frame_id: torch.Tensor,       # (B,) long
    noise_schedule: dict,
    device: torch.device,
    cfg_scale: float = 1.0,
    ddim_steps: int = 50,
    eta: float = 0.0,
    tile_size: int = 64,
) -> torch.Tensor:
    """
    DDIM sampling for ONE tile.
    Returns: (B,4,tile_size,tile_size) in [-1,1]
    """
    B = fourview.shape[0]
    total_timesteps = int(noise_schedule["timesteps"])
    alphas_cumprod = noise_schedule["alphas_cumprod"]

    c = max(1, total_timesteps // max(1, int(ddim_steps)))
    ddim_timesteps = torch.arange(0, total_timesteps, c, device=device)
    if ddim_timesteps[-1].item() != total_timesteps - 1:
        ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([total_timesteps - 1], device=device)])

    x = torch.randn((B, 4, tile_size, tile_size), device=device)

    # NOTE: char_features must be computed once per tile-sampling call (per design)
    char_features = char_encoder(fourview)  # (B,512,16,16)

    use_cfg = float(cfg_scale) > 1.0
    if use_cfg:
        uncond_char = torch.zeros_like(char_features)

    for i in reversed(range(len(ddim_timesteps))):
        t = ddim_timesteps[i]
        t_prev = ddim_timesteps[i - 1] if i > 0 else torch.tensor(-1, device=device)
        t_batch = t.repeat(B)

        alpha_t = alphas_cumprod[t.item()].view(1, 1, 1, 1) if t.item() >= 0 else torch.tensor(1.0, device=device).view(1, 1, 1, 1)
        alpha_prev = alphas_cumprod[t_prev.item()].view(1, 1, 1, 1) if t_prev.item() >= 0 else torch.tensor(1.0, device=device).view(1, 1, 1, 1)

        if use_cfg:
            noise_cond = unet(x, t_batch, context=char_features, cond_ids=frame_id)
            noise_uncond = unet(x, t_batch, context=uncond_char, cond_ids=frame_id)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = unet(x, t_batch, context=char_features, cond_ids=frame_id)

        x0_pred = (x - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        noise = torch.randn_like(x) if eta > 0 else 0

        x = (
            torch.sqrt(alpha_prev) * x0_pred
            + torch.sqrt(torch.clamp(1 - alpha_prev - sigma_t ** 2, min=0.0)) * noise_pred
            + sigma_t * noise
        )

    return torch.clamp(x, -1.0, 1.0)


def assemble_action_sheet_from_tiles(
    tiles_by_frame: dict,
    tile_size: int = 64,
    sheet_rows: int = 12,
    sheet_cols: int = 9
) -> torch.Tensor:
    """
    tiles_by_frame: dict frame_id(int) -> tile (4,64,64) in [-1,1]
    returns: sheet (4, 768, 576) in [-1,1]
    """
    sheet_h = sheet_rows * tile_size
    sheet_w = sheet_cols * tile_size
    device = next(iter(tiles_by_frame.values())).device

    # Use -1.0 for empty pixels so that after [-1,1] -> [0,255],
    # RGBA becomes (0,0,0,0): fully transparent.
    sheet = torch.full((4, sheet_h, sheet_w), -1.0, device=device)

    for frame_id, tile in tiles_by_frame.items():
        row = frame_id // sheet_cols
        col = frame_id % sheet_cols
        y0 = row * tile_size
        x0 = col * tile_size
        sheet[:, y0:y0 + tile_size, x0:x0 + tile_size] = tile

    return sheet


@torch.no_grad()
def generate_full_actions_sheet(
    unet: UNet,
    char_encoder: SpriteCharacterEncoder,
    fourview: torch.Tensor,       # (1,4,128,128)
    noise_schedule: dict,
    device: torch.device,
    cfg_scale: float,
    ddim_steps: int,
    eta: float,
    config: dict,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Generate full sheet (1,4,768,576) in [-1,1]
    """
    tile_size = int(config["tile_size"])
    sheet_cols = int(config["sheet_cols"])
    sheet_rows = int(config["sheet_rows"])
    actions = config["actions"]
    dirs = config["dirs"]
    action_cols = config["action_cols"]

    action_to_id = {a: i for i, a in enumerate(actions)}
    tiles_map = {}

    iterable = actions
    if show_progress:
        iterable = tqdm(actions, desc="Actions", leave=False)

    for action_name in iterable:
        a_id = action_to_id[action_name]
        max_t = int(action_cols[action_name])

        for d_id in range(len(dirs)):
            for t_id in range(max_t):
                frame_id_int = a_id * (4 * sheet_cols) + d_id * sheet_cols + t_id
                frame_id = torch.tensor([frame_id_int], device=device, dtype=torch.long)

                tile = ddim_sample_tile(
                    unet=unet,
                    char_encoder=char_encoder,
                    fourview=fourview,
                    frame_id=frame_id,
                    noise_schedule=noise_schedule,
                    device=device,
                    cfg_scale=float(cfg_scale),
                    ddim_steps=int(ddim_steps),
                    eta=float(eta),
                    tile_size=tile_size,
                )[0]  # (4,64,64)

                tiles_map[frame_id_int] = tile

    sheet = assemble_action_sheet_from_tiles(
        tiles_map, tile_size=tile_size, sheet_rows=sheet_rows, sheet_cols=sheet_cols
    )
    return sheet.unsqueeze(0)


@torch.no_grad()
def generate_full_actions_sheet_batch(
    unet: UNet,
    char_encoder: SpriteCharacterEncoder,
    fourview: torch.Tensor,
    noise_schedule: dict,
    device: torch.device,
    cfg_scale: float,
    ddim_steps: int,
    eta: float,
    config: dict,
    tile_batch_size: int = 64,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Batch-parallel tile generation (optimized for high-end GPUs).

    This function generates multiple tiles in parallel to significantly
    accelerate inference, especially on GPUs with large memory (e.g., A100/H100).

    IMPORTANT:
    - Does NOT change output
    - Only improves speed
    - Safe replacement for generate_full_actions_sheet
    """

    tile_size = int(config["tile_size"])
    sheet_cols = int(config["sheet_cols"])
    sheet_rows = int(config["sheet_rows"])
    actions = config["actions"]
    dirs = config["dirs"]
    action_cols = config["action_cols"]

    action_to_id = {a: i for i, a in enumerate(actions)}

    # ===== collect all frame ids =====
    all_frame_ids = []

    for action_name in actions:
        a_id = action_to_id[action_name]
        max_t = int(action_cols[action_name])

        for d_id in range(len(dirs)):
            for t_id in range(max_t):
                fid = a_id * (4 * sheet_cols) + d_id * sheet_cols + t_id
                all_frame_ids.append(fid)

    tiles_map = {}

    iterator = range(0, len(all_frame_ids), tile_batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Batch Tiles", leave=False)

    # ===== precompute char features ONCE =====
    char_features_single = char_encoder(fourview)  # (1,512,16,16)

    for i in iterator:
        batch_ids = all_frame_ids[i:i + tile_batch_size]
        B = len(batch_ids)

        frame_id = torch.tensor(batch_ids, device=device, dtype=torch.long)

        # repeat input only for shape consistency
        fourview_batch = fourview.repeat(B, 1, 1, 1)

        # reuse encoded features instead of recomputing
        char_features = char_features_single.repeat(B, 1, 1, 1)

        # ===== sampling init =====
        total_timesteps = int(noise_schedule["timesteps"])
        alphas_cumprod = noise_schedule["alphas_cumprod"]

        c = max(1, total_timesteps // max(1, int(ddim_steps)))
        ddim_timesteps = torch.arange(0, total_timesteps, c, device=device)
        if ddim_timesteps[-1].item() != total_timesteps - 1:
            ddim_timesteps = torch.cat([
                ddim_timesteps,
                torch.tensor([total_timesteps - 1], device=device)
            ])

        x = torch.randn((B, 4, tile_size, tile_size), device=device)

        use_cfg = float(cfg_scale) > 1.0
        if use_cfg:
            uncond_char = torch.zeros_like(char_features)

        for t_idx in reversed(range(len(ddim_timesteps))):
            t = ddim_timesteps[t_idx]
            t_prev = ddim_timesteps[t_idx - 1] if t_idx > 0 else torch.tensor(-1, device=device)

            t_batch = t.repeat(B)

            alpha_t = (
                alphas_cumprod[t.item()].view(1, 1, 1, 1)
                if t.item() >= 0 else torch.tensor(1.0, device=device).view(1, 1, 1, 1)
            )
            alpha_prev = (
                alphas_cumprod[t_prev.item()].view(1, 1, 1, 1)
                if t_prev.item() >= 0 else torch.tensor(1.0, device=device).view(1, 1, 1, 1)
            )

            if use_cfg:
                noise_cond = unet(x, t_batch, context=char_features, cond_ids=frame_id)
                noise_uncond = unet(x, t_batch, context=uncond_char, cond_ids=frame_id)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = unet(x, t_batch, context=char_features, cond_ids=frame_id)

            x0_pred = (x - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
            noise = torch.randn_like(x) if eta > 0 else 0

            x = (
                torch.sqrt(alpha_prev) * x0_pred
                + torch.sqrt(torch.clamp(1 - alpha_prev - sigma_t ** 2, min=0.0)) * noise_pred
                + sigma_t * noise
            )

        tiles = torch.clamp(x, -1.0, 1.0)

        for j, fid in enumerate(batch_ids):
            tiles_map[fid] = tiles[j]

    sheet = assemble_action_sheet_from_tiles(
        tiles_map,
        tile_size=tile_size,
        sheet_rows=sheet_rows,
        sheet_cols=sheet_cols,
    )

    return sheet.unsqueeze(0)


# ============================================================
# Section 5: Model loading
# ============================================================
def build_default_config() -> dict:
    """
    Keep config consistent with training.
    """
    return {
        "in_channels": 4,
        "out_channels": 4,

        "fourview_size": 128,
        "tile_size": 64,
        "sheet_cols": 9,
        "sheet_rows": 12,

        "actions": ["walk", "thrust", "slash"],
        "action_cols": {"walk": 9, "thrust": 8, "slash": 6},
        "dirs": ["west", "east", "south", "north"],

        "timesteps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "linear",

        "num_frames": 108,
        "char_encoder_channels": (64, 128, 256, 512),
        "char_feature_dim": 512,

        "unet_channels": 128,
        "channel_mult": (1, 2, 3, 4),
        "num_res_blocks": 2,
        "attention_resolutions": (16, 8),
        "dropout": 0.1,
        "num_heads": 8,
        "use_cross_attention": True,
    }


def _find_default_checkpoint(checkpoint_dir: Path):
    """
    Default checkpoint policy:
      A) Prefer combined checkpoint:
         - model_best.pt
         - checkpoint_latest.pt
         - checkpoint_epoch_*.pt (highest epoch)
      B) Else fallback to split checkpoints:
         - unet_best.pt + char_encoder_best.pt
         - unet_latest.pt + char_encoder_latest.pt (optional)

    Returns:
      - Path (combined ckpt) OR (unet_path, char_path) tuple for split ckpts
    """
    # ---- A) Combined checkpoints ----
    c1 = checkpoint_dir / "model_best.pt"
    if c1.exists():
        return c1

    c2 = checkpoint_dir / "checkpoint_latest.pt"
    if c2.exists():
        return c2

    candidates = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if candidates:
        def epoch_num(p: Path) -> int:
            s = p.stem
            try:
                return int(s.split("_")[-1])
            except Exception:
                return -1
        candidates = sorted(candidates, key=epoch_num)
        return candidates[-1]

    # ---- B) Split checkpoints fallback ----
    unet_best = checkpoint_dir / "unet_best.pt"
    char_best = checkpoint_dir / "char_encoder_best.pt"
    if unet_best.exists() and char_best.exists():
        return (unet_best, char_best)

    unet_latest = checkpoint_dir / "unet_latest.pt"
    char_latest = checkpoint_dir / "char_encoder_latest.pt"
    if unet_latest.exists() and char_latest.exists():
        return (unet_latest, char_latest)

    raise FileNotFoundError(
        f"No checkpoints found in: {checkpoint_dir}\n"
        "Expected one of:\n"
        "  - model_best.pt / checkpoint_latest.pt / checkpoint_epoch_*.pt\n"
        "  - OR split: unet_best.pt + char_encoder_best.pt"
    )


def _unwrap_state_dict(obj):
    """
    Support both:
      - torch.save(model.state_dict(), path)
      - torch.save({"state_dict": model.state_dict(), ...}, path)
      - torch.save({"unet_state_dict": ..., "char_encoder_state_dict": ...}, path)
    """
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def load_models(args, device: torch.device):
    """
    Loading priority:
      1) If --checkpoint provided:
         - If it's a FILE: load combined ckpt (expects both unet+char_encoder in dict).
         - If it's a DIR : load split ckpts inside (unet_best.pt + char_encoder_best.pt).
      2) Else: auto-pick from CHECKPOINT_DIR:
         - Prefer combined: model_best.pt / checkpoint_latest.pt / checkpoint_epoch_*.pt
         - Else fallback to split: unet_best.pt + char_encoder_best.pt
    """
    config = build_default_config()

    char_encoder = SpriteCharacterEncoder(
        in_channels=4,
        channel_progression=config["char_encoder_channels"],
        output_dim=config["char_feature_dim"],
    )
    unet = UNet(config)

    # ------------------------------------------------------------
    # Resolve checkpoint target
    # ------------------------------------------------------------
    ckpt_target = None  # Path (combined) OR tuple(Path, Path) (split)

    if args.checkpoint is not None:
        user_path = Path(args.checkpoint)

        if not user_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {user_path}")

        if user_path.is_dir():
            # Directory: assume split best by default
            unet_path = user_path / "unet_best.pt"
            char_path = user_path / "char_encoder_best.pt"
            if not (unet_path.exists() and char_path.exists()):
                raise FileNotFoundError(
                    f"--checkpoint is a directory, but split checkpoints not found inside:\n"
                    f"  Expected: {unet_path.name} and {char_path.name}\n"
                    f"  In dir: {user_path}"
                )
            ckpt_target = (unet_path, char_path)
        else:
            # File: assume combined training checkpoint
            ckpt_target = user_path
    else:
        ckpt_target = _find_default_checkpoint(CHECKPOINT_DIR)

    # ------------------------------------------------------------
    # Load: combined checkpoint
    # ------------------------------------------------------------
    if isinstance(ckpt_target, Path):
        ckpt_path = ckpt_target
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if not (isinstance(ckpt, dict) and "unet_state_dict" in ckpt and "char_encoder_state_dict" in ckpt):
            raise ValueError(
                "Combined checkpoint must contain 'unet_state_dict' and 'char_encoder_state_dict'.\n"
                "If you saved split checkpoints (unet_best.pt + char_encoder_best.pt), either:\n"
                "  (a) omit --checkpoint and let it auto-find split best, OR\n"
                "  (b) pass --checkpoint as the checkpoints DIRECTORY.\n"
                f"Got: {ckpt_path}"
            )

        unet.load_state_dict(_unwrap_state_dict(ckpt["unet_state_dict"]), strict=True)
        char_encoder.load_state_dict(_unwrap_state_dict(ckpt["char_encoder_state_dict"]), strict=True)

        print(f"[Load] Combined checkpoint: {ckpt_path}")

    # ------------------------------------------------------------
    # Load: split checkpoints
    # ------------------------------------------------------------
    else:
        unet_path, char_path = ckpt_target

        unet_ckpt = torch.load(unet_path, map_location="cpu", weights_only=False)
        char_ckpt = torch.load(char_path, map_location="cpu", weights_only=False)

        unet_state = _unwrap_state_dict(unet_ckpt)
        char_state = _unwrap_state_dict(char_ckpt)

        # Some people save as {"model": state_dict}; support that too (best-effort)
        if isinstance(unet_state, dict) and "model" in unet_state and isinstance(unet_state["model"], dict):
            unet_state = unet_state["model"]
        if isinstance(char_state, dict) and "model" in char_state and isinstance(char_state["model"], dict):
            char_state = char_state["model"]

        unet.load_state_dict(unet_state, strict=True)
        char_encoder.load_state_dict(char_state, strict=True)

        print("[Load] Split checkpoints:")
        print(f"       UNet        : {unet_path}")
        print(f"       Char Encoder: {char_path}")

    unet.to(device).eval()
    char_encoder.to(device).eval()
    return config, unet, char_encoder


# ============================================================
# Section 6: Main
# ============================================================
def collect_inputs(args) -> list[Path]:
    paths: list[Path] = []

    if args.input is not None:
        paths.append(Path(args.input))

    if args.input_dir is not None:
        d = Path(args.input_dir)
        if not d.exists():
            raise FileNotFoundError(f"input_dir not found: {d}")
        paths.extend(sorted([p for p in d.glob("*.png")]))

    if args.filelist is not None:
        fl = Path(args.filelist)
        if not fl.exists():
            raise FileNotFoundError(f"filelist not found: {fl}")
        with open(fl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.append(Path(line))

    # de-dup but keep order
    seen = set()
    final = []
    for p in paths:
        rp = str(p.expanduser().resolve())
        if rp not in seen:
            seen.add(rp)
            final.append(Path(rp))

    if not final:
        raise ValueError("No inputs provided. Use --input, --input_dir, or --filelist.")
    return final


def find_next_index(out_dir: Path, prefix: str = "image_conditional_", ext: str = ".png") -> int:
    """
    Find next sequential index based on existing files:
      image_conditional_0001.png, image_conditional_0002.png, ...
    Returns next index (int).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{prefix}*{ext}"))
    best = 0
    for p in existing:
        s = p.stem  # image_conditional_0001
        tail = s.replace(prefix, "")
        if tail.isdigit():
            best = max(best, int(tail))
    return best + 1


def main():
    parser = argparse.ArgumentParser()

    # Inputs (choose one)
    parser.add_argument("--input", type=str, default=None, help="Single 4-view PNG path")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of 4-view PNGs")
    parser.add_argument("--filelist", type=str, default=None, help="Text file with one PNG path per line")

    # Checkpoint selection (optional)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional training checkpoint_epoch_*.pt containing both unet+char_encoder states. "
             "If omitted, script auto-picks a default checkpoint from models/pixel_image_conditional/checkpoints. "
             "If you saved split checkpoints, pass the checkpoints DIRECTORY to load unet_best.pt + char_encoder_best.pt.",
    )

    # Sampling controls
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale (>=1.0)")
    parser.add_argument("--ddim_steps", type=int, default=10, help="DDIM steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0.0 deterministic)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    # Tile-level batch inference (optional acceleration)
    parser.add_argument(
        "--tile_batch_size",
        type=int,
        default=1,
        help="Tile-level batch size (>1 enables faster inference but uses more GPU memory)"
    )

    # Output
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # load models
    config, unet, char_encoder = load_models(args, device)

    # noise schedule
    noise_schedule = prepare_noise_schedule(config, device)

    # collect inputs
    input_paths = collect_inputs(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Inputs] {len(input_paths)} image(s)")
    print(f"[Output] {out_dir}")

    next_idx = find_next_index(out_dir)

    # Process one-by-one (fullsheet is heavy; keep it simple and deterministic)
    for p in input_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

        fourview = load_fourview_rgba(p, size=int(config["fourview_size"])).to(device)  # (1,4,128,128)

        t0 = time.time()

        # Select generation mode based on tile_batch_size
        if int(args.tile_batch_size) > 1:
            sheet = generate_full_actions_sheet_batch(
                unet=unet,
                char_encoder=char_encoder,
                fourview=fourview,
                noise_schedule=noise_schedule,
                device=device,
                cfg_scale=float(args.cfg_scale),
                ddim_steps=int(args.ddim_steps),
                eta=float(args.eta),
                config=config,
                tile_batch_size=int(args.tile_batch_size),
                show_progress=True,
            )[0]
        else:
            sheet = generate_full_actions_sheet(
                unet=unet,
                char_encoder=char_encoder,
                fourview=fourview,
                noise_schedule=noise_schedule,
                device=device,
                cfg_scale=float(args.cfg_scale),
                ddim_steps=int(args.ddim_steps),
                eta=float(args.eta),
                config=config,
                show_progress=True,
            )[0]  # (4,768,576) in [-1,1]

        dt = time.time() - t0

        sheet_uint8 = tensor_to_uint8_rgba(sheet)

        out_name = f"image_conditional_{next_idx:04d}.png"
        out_path = out_dir / out_name
        save_rgba_tensor_as_png(sheet_uint8, out_path)

        print(f"[Saved] {out_path.name}  ({dt:.1f}s)  <- {p.name}")
        next_idx += 1

    print("[Done]")


if __name__ == "__main__":
    main()
