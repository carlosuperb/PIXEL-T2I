"""
inference.py - Unconditional Sprite Generation (All-in-One)

All-in-one inference script for generating 128x128 RGBA pixel art sprites.
Includes model definition, DDIM sampling, and visualization tools.

Features:
- Generate sprites with DDIM sampling
- Visualize denoising process step-by-step

Usage:
    # Generate sprites
    python inference.py --num_samples 16 --ddim_steps 50
    
    # Visualize denoising process
    python inference.py --mode visualize --ddim_steps 10
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent


# Section 1: Model Architecture
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for timestep t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        
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
        
        self.act = nn.SiLU()
    
    def forward(self, x, time_emb):
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
        
        return h + residual


class AttentionBlock(nn.Module):
    """Self-attention block for spatial features."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
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
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling by factor of 2."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet for unconditional diffusion models.
    Optimized for 128x128 RGBA sprite generation.
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        in_channels = config["in_channels"]
        model_channels = config["unet_channels"]
        channel_mult = config["channel_mult"]
        num_res_blocks = config["num_res_blocks"]
        attention_resolutions = config["attention_resolutions"]
        dropout = config["dropout"]
        num_heads = config["num_heads"]
        
        time_emb_dim = model_channels * 4
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        current_channels = model_channels
        current_resolution = config["image_size"]
        self.encoder_out_channels = [model_channels]
        
        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            
            for i in range(num_res_blocks):
                block_layers = nn.ModuleList()
                block_layers.append(ResidualBlock(current_channels, out_channels, time_emb_dim, dropout))
                current_channels = out_channels
                
                if current_resolution in attention_resolutions:
                    block_layers.append(AttentionBlock(current_channels, num_heads))
                
                self.encoder_blocks.append(block_layers)
                self.encoder_out_channels.append(current_channels)
            
            if level != len(channel_mult) - 1:
                self.encoder_blocks.append(nn.ModuleList([Downsample(current_channels)]))
                current_resolution //= 2
                self.encoder_out_channels.append(current_channels)
        
        # Bottleneck
        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(current_channels, num_heads)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_emb_dim, dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_channels = self.encoder_out_channels.pop()
                block_layers = nn.ModuleList()
                block_layers.append(ResidualBlock(
                    current_channels + skip_channels, 
                    out_channels, 
                    time_emb_dim, 
                    dropout
                ))
                current_channels = out_channels
                
                if current_resolution in attention_resolutions:
                    block_layers.append(AttentionBlock(current_channels, num_heads))
                
                self.decoder_blocks.append(block_layers)
                
                if level != 0 and i == num_res_blocks:
                    self.decoder_blocks.append(nn.ModuleList([Upsample(current_channels)]))
                    current_resolution *= 2
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x, t):
        """
        Args:
            x: Noisy images (B, 4, 128, 128)
            t: Timesteps (B,)
        Returns:
            Predicted noise (B, 4, 128, 128)
        """
        time_emb = self.time_mlp(t)
        h = self.conv_in(x)
        
        encoder_features = [h]
        for block_layers in self.encoder_blocks:
            for layer in block_layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Downsample):
                    h = layer(h)
            encoder_features.append(h)
        
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        for block_layers in self.decoder_blocks:
            if any(isinstance(layer, ResidualBlock) for layer in block_layers):
                skip = encoder_features.pop()
                h = torch.cat([h, skip], dim=1)
            
            for layer in block_layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Upsample):
                    h = layer(h)
        
        return self.conv_out(h)


# Section 2: Noise Schedule
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear schedule from DDPM paper."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from Improved DDPM paper."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def prepare_noise_schedule(config, device):
    """Precompute all noise schedule parameters."""
    timesteps = config["timesteps"]
    
    if config["beta_schedule"] == "linear":
        betas = linear_beta_schedule(timesteps, config["beta_start"], config["beta_end"])
    elif config["beta_schedule"] == "cosine":
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown beta schedule: {config['beta_schedule']}")
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    noise_schedule = {
        "betas": betas.to(device),
        "alphas": alphas.to(device),
        "alphas_cumprod": alphas_cumprod.to(device),
        "alphas_cumprod_prev": alphas_cumprod_prev.to(device),
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod.to(device),
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod.to(device),
        "sqrt_recip_alphas": sqrt_recip_alphas.to(device),
        "posterior_variance": posterior_variance.to(device),
        'timesteps': timesteps,
    }
    
    return noise_schedule


# Section 3: DDIM Sampling
@torch.no_grad()
def ddim_sample_step(unet, x, t, t_prev, noise_schedule, eta=0.0):
    """Single DDIM sampling step."""
    device = x.device
    alphas_cumprod = noise_schedule['alphas_cumprod']
    
    t_index = t[0].item()
    alpha_t = alphas_cumprod[t_index]
    
    if isinstance(t_prev, torch.Tensor):
        if t_prev.item() >= 0:
            alpha_prev = alphas_cumprod[t_prev.item()]
        else:
            alpha_prev = torch.tensor(1.0, device=device)
    else:
        if t_prev >= 0:
            alpha_prev = alphas_cumprod[t_prev]
        else:
            alpha_prev = torch.tensor(1.0, device=device)
    
    alpha_t = alpha_t.view(1, 1, 1, 1)
    alpha_prev = alpha_prev.view(1, 1, 1, 1)
    
    predicted_noise = unet(x, t)
    
    x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
    
    sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
    
    noise = torch.randn_like(x) if eta > 0 else 0
    x_prev = torch.sqrt(alpha_prev) * x0_pred + \
             torch.sqrt(1 - alpha_prev - sigma_t**2) * predicted_noise + \
             sigma_t * noise
    
    return x_prev


@torch.no_grad()
def ddim_sample(unet, shape, noise_schedule, device, ddim_steps=50, eta=0.0, show_progress=True):
    """DDIM sampling - fast and deterministic."""
    batch_size = shape[0]
    total_timesteps = noise_schedule['timesteps']
    
    c = total_timesteps // ddim_steps
    ddim_timesteps = torch.arange(0, total_timesteps, c, device=device)
    ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([total_timesteps - 1], device=device)])
    
    x = torch.randn(shape, device=device)
    
    iterator = reversed(range(len(ddim_timesteps)))
    if show_progress:
        iterator = tqdm(list(iterator), desc="DDIM Sampling")
    
    for i in iterator:
        t = ddim_timesteps[i]
        t_prev = ddim_timesteps[i - 1] if i > 0 else torch.tensor(-1, device=device)
        t_batch = t.repeat(batch_size)
        
        x = ddim_sample_step(unet, x, t_batch, t_prev, noise_schedule, eta)
    
    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def ddim_sample_with_trajectory(unet, shape, noise_schedule, device, ddim_steps=50, eta=0.0):
    """
    DDIM sampling that returns the entire denoising trajectory.
    Used for visualization.
    
    Returns:
        final_image: Final generated image
        trajectory: List of intermediate images at each step
    """
    batch_size = shape[0]
    total_timesteps = noise_schedule['timesteps']
    
    c = total_timesteps // ddim_steps
    ddim_timesteps = torch.arange(0, total_timesteps, c, device=device)
    ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([total_timesteps - 1], device=device)])
    
    x = torch.randn(shape, device=device)
    trajectory = [x.clone()]
    
    for i in tqdm(reversed(range(len(ddim_timesteps))), desc="Generating trajectory"):
        t = ddim_timesteps[i]
        t_prev = ddim_timesteps[i - 1] if i > 0 else torch.tensor(-1, device=device)
        t_batch = t.repeat(batch_size)
        
        x = ddim_sample_step(unet, x, t_batch, t_prev, noise_schedule, eta)
        trajectory.append(x.clone())
    
    return x, trajectory


# Section 4: Checkpoint Loading
def load_checkpoint(checkpoint_path, device):
    """
    Load model checkpoint.
    Supports both full checkpoints and model-only files.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        unet: Loaded UNet model
        config: Model configuration
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if it's a full checkpoint or model-only
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint
        config = checkpoint['config']
        model_state_dict = checkpoint['model_state_dict']
        print(f"  Type: Full checkpoint")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Loss: {checkpoint.get('loss', 'N/A'):.6f}" if 'loss' in checkpoint else "")
    else:
        # Model-only checkpoint
        model_state_dict = checkpoint
        # Use default config
        config = get_default_config()
        print(f"  Type: Model-only checkpoint")
    
    # Create and load model
    unet = UNet(config)
    unet.load_state_dict(model_state_dict)
    unet = unet.to(device)
    unet.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"  Parameters: {total_params / 1e6:.2f}M")
    print(f"  Device: {device}")
    
    return unet, config


def get_default_config():
    """Default configuration for model."""
    return {
        "image_size": 128,
        "in_channels": 4,
        "unet_channels": 128,
        "channel_mult": (1, 2, 4, 4),
        "num_res_blocks": 2,
        "attention_resolutions": (8, 16),
        "dropout": 0.1,
        "num_heads": 8,
        "timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
    }


# Section 5: Image Utilities
def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    # tensor: (C, H, W) in range [-1, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Keep all 4 channels for RGBA
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor, mode='RGBA')


def save_images(
    images,
    save_dir,
    prefix="sample",
    save_individual=True,
    save_grid=True,
    start_index=0,
):
    """
    Save generated images.

    Args:
        images: Tensor of shape (B, 4, 128, 128) in range [-1, 1]
        save_dir: Directory to save images
        prefix: Filename prefix
        save_individual: Whether to save individual images
        save_grid: Whether to save a grid
        start_index: Global starting index for image numbering
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert to [0, 1]
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0.0, 1.0)

    # Save individual images with global indexing
    if save_individual:
        for i, img in enumerate(images):
            global_idx = start_index + i
            filepath = save_dir / f"{prefix}_{global_idx:04d}.png"
            save_image(img, filepath)
        print(f"Saved {len(images)} individual images to {save_dir}")

    # Save grid (unchanged)
    if save_grid and len(images) > 1:
        nrow = min(4, len(images))
        grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
        grid_path = save_dir / f"{prefix}_grid.png"
        save_image(grid, grid_path)
        print(f"Saved grid to {grid_path}")

    return save_dir


# Section 6: Generation Functions
def generate_sprites(args):
    """Main generation function with internal auto-batching to avoid OOM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    unet, config = load_checkpoint(checkpoint_path, device)

    # Prepare noise schedule
    noise_schedule = prepare_noise_schedule(config, device)

    # Output directory (default: temp_outputs)
    DEFAULT_OUTDIR = SCRIPT_DIR / "temp_outputs"
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Internal batching (hidden from user)
    INTERNAL_BATCH = 8  # fixed internal batch size to reduce VRAM usage

    total = args.num_samples
    generated = 0
    batch_id = 0

    print(f"\nGenerating {total} sprites with {args.ddim_steps} DDIM steps...")
    print(f"Using output directory: {output_dir}")
    print(f"Internal batching enabled: up to {INTERNAL_BATCH} samples per batch")

    start_time = time.time()

    while generated < total:
        cur = min(INTERNAL_BATCH, total - generated)
        batch_id += 1

        print(f"\nBatch {batch_id}: generating {cur} sprites ({generated}/{total} done)...")

        shape = (cur, config["in_channels"], config["image_size"], config["image_size"])
        samples = ddim_sample(
            unet,
            shape,
            noise_schedule,
            device,
            ddim_steps=args.ddim_steps,
            eta=0.0,
            show_progress=True,
        )

        # Save individual images only (no grids)
        save_images(
            samples,
            output_dir,
            prefix="generated",
            save_individual=True,
            save_grid=False,
            start_index=generated,
        )

        # Free GPU memory between batches
        del samples
        torch.cuda.empty_cache()

        generated += cur

    generation_time = time.time() - start_time
    print(
        f"\nGeneration completed in {generation_time:.2f}s "
        f"({generation_time / total:.2f}s per image)"
    )
    print(f"\nDone! Images saved to {output_dir}")


# Section 7: Visualization Functions
def visualize_denoising(args):
    """Visualize the denoising process step by step."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    unet, config = load_checkpoint(Path(args.checkpoint), device)
    noise_schedule = prepare_noise_schedule(config, device)
    
    # Generate with trajectory
    print(f"\nGenerating sprite with trajectory ({args.ddim_steps} steps)...")
    shape = (1, config["in_channels"], config["image_size"], config["image_size"])
    final_image, trajectory = ddim_sample_with_trajectory(
        unet, shape, noise_schedule, device, ddim_steps=args.ddim_steps, eta=0.0
    )
    
    # Select frames to display (evenly spaced)
    num_frames = min(args.vis_frames, len(trajectory))
    frame_indices = np.linspace(0, len(trajectory) - 1, num_frames, dtype=int)
    
    # Create visualization
    fig, axes = plt.subplots(2, num_frames // 2, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(frame_indices):
        img = trajectory[frame_idx][0]  # First image in batch
        img = (img + 1.0) / 2.0
        img = torch.clamp(img[:3], 0.0, 1.0)  # RGB only for display
        img = img.cpu().permute(1, 2, 0).numpy()
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        if frame_idx == 0:
            axes[idx].set_title(f'Start (noise)', fontsize=10)
        elif frame_idx == len(trajectory) - 1:
            axes[idx].set_title(f'End (clean)', fontsize=10)
        else:
            step_num = len(trajectory) - 1 - frame_idx
            axes[idx].set_title(f'Step {step_num}', fontsize=10)
    
    plt.suptitle(f'Denoising Process ({args.ddim_steps} DDIM steps)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # ---- Save (default to temp_outputs) ----
    DEFAULT_OUTDIR = SCRIPT_DIR / "temp_outputs"
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTDIR
    
    # args.output is treated as a FILENAME (not a full path)
    filename = args.output if args.output else "visualization.png"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to {output_path}")

    plt.show()


# Section 8: Command Line Interface
def main():
    parser = argparse.ArgumentParser(
        description="Unconditional Sprite Generation with Diffusion Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 4 sprites with default settings
  python inference.py

  # Generate 16 sprites
  python inference.py --num_samples 16

  # Generate with more DDIM steps (higher quality)
  python inference.py --num_samples 16 --ddim_steps 50

  # Save generated sprites to a custom directory
  python inference.py --num_samples 16 --output_dir my_sprites

  # Visualize denoising process (saved to temp_outputs/visualization.png)
  python inference.py --mode visualize

  # Visualize with more DDIM steps (saved to temp_outputs/visualization.png)
  python inference.py --mode visualize --ddim_steps 20

  # Visualize and save to a custom directory
  python inference.py --mode visualize --output_dir reports/figures/denoising

  # Visualize with a custom filename (saved to temp_outputs/run1.png)
  python inference.py --mode visualize --output run1.png

  # Visualize with custom directory and filename
  python inference.py --mode visualize --output_dir reports/figures/denoising --output run1.png
        """
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="generate",
                       choices=["generate", "visualize"],
                       help="Operation mode: generate sprites or visualize denoising (default: generate)")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, 
                   default=str(Path(__file__).resolve().parent / "checkpoints" / "model_best.pt"),
                   help="Path to model checkpoint")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=4,
                       help="Number of sprites to generate (default: 4)")
    parser.add_argument("--ddim_steps", type=int, default=10,
                       help="Number of DDIM sampling steps (default: 10)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for generated images and visualizations (default: temp_outputs)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename for visualization (default: visualization.png, visualize mode only)")
    
    # Visualization parameters
    parser.add_argument("--vis_frames", type=int, default=8,
                       help="Number of frames to show in visualization (default: 8)")
    
    args = parser.parse_args()

    if args.output and (Path(args.output).parent != Path(".")):
        raise ValueError("--output must be a filename only. Use --output_dir to set the directory.")
    
    # Route to appropriate function
    if args.mode == "generate":
        generate_sprites(args)
    elif args.mode == "visualize":
        visualize_denoising(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()