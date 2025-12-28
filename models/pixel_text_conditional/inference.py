"""
inference.py - Text-Conditional Sprite Generation

Text-conditional inference script for generating 128x128 RGBA pixel art sprites
from text descriptions using a trained diffusion model with CLIP text encoding.

Features:
- Generate sprites from single or multiple text prompts
- Adjustable CFG scale and sampling steps
- Visualize the denoising process step-by-step
- Batch processing with automatic memory management

Usage:
    # Generate from single prompt (uses default checkpoint)
    python inference.py --prompt "male human with red cape, 4-view sprite"
    
    # Generate from multiple prompts file
    python inference.py --prompts prompts.txt --ddim_steps 100
    
    # Visualize denoising process
    python inference.py --mode visualize --prompt "female elf with gold armor"
    
    # Custom checkpoint and output directory
    python inference.py --checkpoint models/checkpoint_epoch_50.pt --prompt "warrior" --output_dir my_outputs
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from transformers import CLIPTokenizer, CLIPTextModel
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


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for text conditioning.
    Q from image features, K/V from text embeddings.
    """
    def __init__(self, channels, context_dim, num_heads=8):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(32, channels)
        self.context_norm = nn.LayerNorm(context_dim)

        # Q from image, K/V from text
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, context):
        """
        Args:
            x: Image features (B, C, H, W)
            context: Text embeddings (B, seq_len, context_dim)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x

        # Normalize
        x = self.norm(x)
        context = self.context_norm(context)

        # Q from image
        q = self.to_q(x)  # (B, C, H, W)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        q = q.permute(0, 1, 3, 2)  # (B, num_heads, H*W, head_dim)

        # K, V from text
        k = self.to_k(context)  # (B, seq_len, C)
        v = self.to_v(context)  # (B, seq_len, C)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = torch.matmul(attn, v)  # (B, num_heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # Project and residual
        out = self.to_out(out)
        return out + residual


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding and optional cross-attention.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim,
                 dropout=0.0, use_cross_attn=False, context_dim=None):
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

        # Cross-attention
        if use_cross_attn:
            assert context_dim is not None, "context_dim must be provided for cross-attention"
            self.cross_attn = CrossAttentionBlock(out_channels, context_dim)
        else:
            self.cross_attn = None

        self.act = nn.SiLU()

    def forward(self, x, time_emb, context=None):
        """
        Args:
            x: Image features (B, C, H, W)
            time_emb: Time embeddings (B, time_emb_dim)
            context: Text embeddings (B, seq_len, context_dim) or None
        Returns:
            out: (B, C, H, W)
        """
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

        # Cross-attention with text
        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)

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
    UNet for text-conditional diffusion models.
    Optimized for 128x128 RGBA sprite generation with CLIP text conditioning.
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

        # Text conditioning
        use_cross_attn = config.get("use_cross_attention", True)
        context_dim = config.get("text_embed_dim", 512)

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

                # ResBlock with cross-attention
                block_layers.append(
                    ResidualBlock(
                        current_channels,
                        out_channels,
                        time_emb_dim,
                        dropout,
                        use_cross_attn=use_cross_attn,
                        context_dim=context_dim
                    )
                )
                current_channels = out_channels

                # Self-attention
                if current_resolution in attention_resolutions:
                    block_layers.append(AttentionBlock(current_channels, num_heads))

                self.encoder_blocks.append(block_layers)
                self.encoder_out_channels.append(current_channels)

            # Downsample
            if level != len(channel_mult) - 1:
                self.encoder_blocks.append(nn.ModuleList([Downsample(current_channels)]))
                current_resolution //= 2
                self.encoder_out_channels.append(current_channels)

        # Bottleneck
        self.mid_block1 = ResidualBlock(
            current_channels,
            current_channels,
            time_emb_dim,
            dropout,
            use_cross_attn=use_cross_attn,
            context_dim=context_dim
        )
        self.mid_attn = AttentionBlock(current_channels, num_heads)
        self.mid_block2 = ResidualBlock(
            current_channels,
            current_channels,
            time_emb_dim,
            dropout,
            use_cross_attn=use_cross_attn,
            context_dim=context_dim
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult

            for i in range(num_res_blocks + 1):
                skip_channels = self.encoder_out_channels.pop()
                block_layers = nn.ModuleList()

                # ResBlock with cross-attention
                block_layers.append(
                    ResidualBlock(
                        current_channels + skip_channels,
                        out_channels,
                        time_emb_dim,
                        dropout,
                        use_cross_attn=use_cross_attn,
                        context_dim=context_dim
                    )
                )
                current_channels = out_channels

                # Self-attention
                if current_resolution in attention_resolutions:
                    block_layers.append(AttentionBlock(current_channels, num_heads))

                self.decoder_blocks.append(block_layers)

                # Upsample
                if level != 0 and i == num_res_blocks:
                    self.decoder_blocks.append(nn.ModuleList([Upsample(current_channels)]))
                    current_resolution *= 2

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t, context=None):
        """
        Args:
            x: Noisy images (B, 4, 128, 128)
            t: Timesteps (B,)
            context: Text embeddings (B, seq_len, text_embed_dim) or None
        Returns:
            Predicted noise (B, 4, 128, 128)
        """
        # Time embedding
        time_emb = self.time_mlp(t)

        # Initial conv
        h = self.conv_in(x)

        # Encoder
        encoder_features = [h]
        for block_layers in self.encoder_blocks:
            for layer in block_layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb, context)  # Pass context
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Downsample):
                    h = layer(h)
            encoder_features.append(h)

        # Bottleneck
        h = self.mid_block1(h, time_emb, context)  # Pass context
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb, context)  # Pass context

        # Decoder
        for block_layers in self.decoder_blocks:
            if any(isinstance(layer, ResidualBlock) for layer in block_layers):
                skip = encoder_features.pop()
                h = torch.cat([h, skip], dim=1)

            for layer in block_layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb, context)  # Pass context
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Upsample):
                    h = layer(h)

        # Output
        return self.conv_out(h)


# Section 2: Noise Schedule
def prepare_noise_schedule(config, device):
    """Prepare noise schedule for diffusion process"""
    timesteps = config.get("timesteps", 1000)
    beta_start = config.get("beta_start", 0.0001)
    beta_end = config.get("beta_end", 0.02)
    beta_schedule = config.get("beta_schedule", "linear")
    
    if beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    else:
        raise ValueError(f"Unknown beta schedule: {beta_schedule}")
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        'timesteps': timesteps,
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
    }


# Section 3: DDIM Sampling
@torch.no_grad()
def ddim_sample_step(unet, x, t, t_prev, noise_schedule, context=None, eta=0.0):
    """Single DDIM sampling step"""
    device = x.device
    alphas_cumprod = noise_schedule['alphas_cumprod']
    
    if t >= 0:
        alpha_t = alphas_cumprod[t]
    else:
        alpha_t = torch.tensor(1.0, device=device)
    
    if t_prev >= 0:
        alpha_t_prev = alphas_cumprod[t_prev]
    else:
        alpha_t_prev = torch.tensor(1.0, device=device)
    
    predicted_noise = unet(x, t.expand(x.shape[0]), context=context)
    
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    pred_x0 = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    
    sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
    dir_xt = torch.sqrt(1.0 - alpha_t_prev - eta**2 * (1.0 - alpha_t_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_t_prev)) * predicted_noise
    
    if eta > 0:
        noise = torch.randn_like(x)
        sigma_t = eta * torch.sqrt((1.0 - alpha_t_prev) / (1.0 - alpha_t)) * torch.sqrt(1.0 - alpha_t / alpha_t_prev)
    else:
        noise = 0
        sigma_t = 0
    
    x_prev = sqrt_alpha_t_prev * pred_x0 + dir_xt + sigma_t * noise
    
    return x_prev


@torch.no_grad()
def ddim_sample(unet, shape, noise_schedule, device, 
                context=None, cfg_scale=1.0, 
                ddim_steps=50, eta=0.0, show_progress=True):
    """
    DDIM sampling with Classifier-Free Guidance
    """
    batch_size = shape[0]
    total_timesteps = noise_schedule['timesteps']
    
    c = total_timesteps // ddim_steps
    ddim_timesteps = torch.arange(0, total_timesteps, c, device=device)
    ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([total_timesteps - 1], device=device)])
    
    x = torch.randn(shape, device=device)
    
    use_cfg = (context is not None) and (cfg_scale > 1.0)
    
    if use_cfg:
        uncond_context = torch.zeros_like(context)
    
    iterator = reversed(range(len(ddim_timesteps)))
    if show_progress:
        iterator = tqdm(list(iterator), desc="DDIM Sampling")
    
    for i in iterator:
        t = ddim_timesteps[i]
        t_prev = ddim_timesteps[i - 1] if i > 0 else torch.tensor(-1, device=device)
        t_batch = t.repeat(batch_size)
        
        if use_cfg:
            noise_pred_cond = unet(x, t_batch, context=context)
            noise_pred_uncond = unet(x, t_batch, context=uncond_context)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            alphas_cumprod = noise_schedule['alphas_cumprod']
            
            if t >= 0:
                alpha_t = alphas_cumprod[t]
            else:
                alpha_t = torch.tensor(1.0, device=device)
            
            if t_prev >= 0:
                alpha_t_prev = alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            pred_x0 = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            dir_xt = torch.sqrt(1.0 - alpha_t_prev) * noise_pred
            x = sqrt_alpha_t_prev * pred_x0 + dir_xt
            
        else:
            x = ddim_sample_step(unet, x, t_batch, t_prev, noise_schedule, context, eta)
    
    return x


@torch.no_grad()
def ddim_sample_with_trajectory(unet, shape, noise_schedule, device, 
                                context=None, cfg_scale=1.0,
                                ddim_steps=50, eta=0.0):
    """
    DDIM sampling that returns the entire denoising trajectory.
    Used for visualization.
    """
    batch_size = shape[0]
    total_timesteps = noise_schedule['timesteps']
    
    c = total_timesteps // ddim_steps
    ddim_timesteps = torch.arange(0, total_timesteps, c, device=device)
    ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([total_timesteps - 1], device=device)])
    
    x = torch.randn(shape, device=device)
    trajectory = [x.clone()]
    
    use_cfg = (context is not None) and (cfg_scale > 1.0)
    
    if use_cfg:
        uncond_context = torch.zeros_like(context)
    
    for i in tqdm(reversed(range(len(ddim_timesteps))), desc="Generating trajectory"):
        t = ddim_timesteps[i]
        t_prev = ddim_timesteps[i - 1] if i > 0 else torch.tensor(-1, device=device)
        t_batch = t.repeat(batch_size)
        
        if use_cfg:
            noise_pred_cond = unet(x, t_batch, context=context)
            noise_pred_uncond = unet(x, t_batch, context=uncond_context)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            alphas_cumprod = noise_schedule['alphas_cumprod']
            
            if t >= 0:
                alpha_t = alphas_cumprod[t]
            else:
                alpha_t = torch.tensor(1.0, device=device)
            
            if t_prev >= 0:
                alpha_t_prev = alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            pred_x0 = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            dir_xt = torch.sqrt(1.0 - alpha_t_prev) * noise_pred
            x = sqrt_alpha_t_prev * pred_x0 + dir_xt
        else:
            x = ddim_sample_step(unet, x, t_batch, t_prev, noise_schedule, context, eta)
        
        trajectory.append(x.clone())
    
    return x, trajectory


# Section 4: Checkpoint Loading
def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint and initialize CLIP"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint with model_state_dict
        model_state_dict = checkpoint['model_state_dict']
        
        # Try to get config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  Type: Full checkpoint with config")
        else:
            # No config in checkpoint, use default
            config = get_default_config()
            print(f"  Type: Checkpoint without config (using default config)")
        
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A') + 1 if 'epoch' in checkpoint else 'N/A'}")
        print(f"  Loss: {checkpoint.get('loss', 'N/A'):.6f}" if 'loss' in checkpoint else "")
        
    elif isinstance(checkpoint, dict) and any(k.startswith('conv_in') for k in checkpoint.keys()):
        # Direct state_dict (no wrapper)
        model_state_dict = checkpoint
        config = get_default_config()
        print(f"  Type: Direct state_dict (using default config)")
        
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    # Initialize UNet
    unet = UNet(config)
    unet.load_state_dict(model_state_dict)
    unet = unet.to(device)
    unet.eval()
    
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"  Parameters: {total_params / 1e6:.2f}M")
    
    # Initialize CLIP
    clip_model_name = config.get("clip_model_name", "openai/clip-vit-base-patch32")
    print(f"Loading CLIP: {clip_model_name}")
    
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    text_encoder.eval()
    
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    print(f"  Device: {device}")
    
    return unet, text_encoder, tokenizer, config


def get_default_config():
    """Default configuration matching training config"""
    return {
        "image_size": 128,
        "in_channels": 4,
        "unet_channels": 128,
        "channel_mult": (1, 2, 4, 4),
        "num_res_blocks": 2,
        "attention_resolutions": (8, 16),
        "dropout": 0.1,
        "num_heads": 8,
        "use_cross_attention": True,
        "text_embed_dim": 512,
        "clip_model_name": "openai/clip-vit-base-patch32",
        "max_text_length": 77,
        "timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
    }


# Section 5: Text Encoding
def encode_text(prompts, tokenizer, text_encoder, config, device):
    """Encode text prompts using CLIP"""
    max_length = config.get("max_text_length", 77)
    
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        text_embeddings = text_encoder(**text_inputs).last_hidden_state
    
    return text_embeddings


# Section 6: Image Utilities
def save_images(images, save_dir, prompts=None, prefix="generated", save_individual=True, save_grid=True, start_index=0):
    """Save generated images"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0.0, 1.0)
    
    if save_individual:
        for i, img in enumerate(images):
            global_idx = start_index + i
            # Simple naming: generated_0000.png
            filename = f"{prefix}_{global_idx:04d}.png"
            
            filepath = save_dir / filename
            save_image(img, filepath)
        print(f"Saved {len(images)} individual images to {save_dir}")
    
    if save_grid and len(images) > 1:
        nrow = min(4, len(images))
        grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
        grid_path = save_dir / f"{prefix}_grid.png"
        save_image(grid, grid_path)
        print(f"Saved grid to {grid_path}")
    
    # Save prompts file with index mapping
    if prompts is not None:
        prompts_file = save_dir / f"{prefix}_prompts.txt"
        
        # If file exists, append; otherwise create new
        mode = 'a' if prompts_file.exists() else 'w'
        
        with open(prompts_file, mode, encoding='utf-8') as f:
            for i, prompt in enumerate(prompts):
                global_idx = start_index + i
                f.write(f"{global_idx:04d}: {prompt}\n")
        
        print(f"Prompts saved: {prompts_file.name}")
    
    return save_dir


# Section 7: Generation Functions
def generate_sprites(args):
    """Main generation function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model and CLIP
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    unet, text_encoder, tokenizer, config = load_checkpoint(checkpoint_path, device)
    noise_schedule = prepare_noise_schedule(config, device)
    
    # Get prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts:
        with open(args.prompts, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must provide either --prompt or --prompts")
    
    # Output directory
    DEFAULT_OUTDIR = SCRIPT_DIR / "temp_outputs"
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Internal batching
    INTERNAL_BATCH = 8
    total = len(prompts)
    generated = 0
    batch_id = 0
    
    print(f"\nGenerating {total} sprites with {args.ddim_steps} DDIM steps (CFG={args.cfg_scale})...")
    print(f"Using output directory: {output_dir}")
    print(f"Internal batching enabled: up to {INTERNAL_BATCH} samples per batch")
    
    start_time = time.time()
    all_prompts_for_saving = []
    
    while generated < total:
        cur = min(INTERNAL_BATCH, total - generated)
        batch_prompts = prompts[generated:generated + cur]
        batch_id += 1
        
        print(f"\nBatch {batch_id}: generating {cur} sprites ({generated}/{total} done)...")
        
        # Encode text
        text_embeddings = encode_text(batch_prompts, tokenizer, text_encoder, config, device)
        
        # Generate
        shape = (cur, config["in_channels"], config["image_size"], config["image_size"])
        samples = ddim_sample(
            unet,
            shape,
            noise_schedule,
            device,
            context=text_embeddings,
            cfg_scale=args.cfg_scale,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
            show_progress=True,
        )
        
        # Save
        save_images(
            samples,
            output_dir,
            prompts=batch_prompts,
            prefix="generated",
            save_individual=True,
            save_grid=False,
            start_index=generated,
        )
        
        all_prompts_for_saving.extend(batch_prompts)
        
        del samples, text_embeddings
        torch.cuda.empty_cache()
        
        generated += cur
    
    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f}s ({generation_time / total:.2f}s per image)")
    print(f"\nDone! Images saved to {output_dir}")


def visualize_denoising(args):
    """Visualize the denoising process step by step"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    unet, text_encoder, tokenizer, config = load_checkpoint(Path(args.checkpoint), device)
    noise_schedule = prepare_noise_schedule(config, device)
    
    # Get prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "lpc-style pixel art character, 4-view sprite sheet"
        print(f"No prompt provided, using default: {prompt}")
    
    # Encode text
    print(f"\nPrompt: {prompt}")
    text_embeddings = encode_text([prompt], tokenizer, text_encoder, config, device)
    
    # Generate with trajectory
    print(f"\nGenerating sprite with trajectory ({args.ddim_steps} steps, CFG={args.cfg_scale})...")
    shape = (1, config["in_channels"], config["image_size"], config["image_size"])
    final_image, trajectory = ddim_sample_with_trajectory(
        unet, shape, noise_schedule, device,
        context=text_embeddings,
        cfg_scale=args.cfg_scale,
        ddim_steps=args.ddim_steps,
        eta=args.eta
    )
    
    # Select frames
    num_frames = min(args.vis_frames, len(trajectory))
    frame_indices = np.linspace(0, len(trajectory) - 1, num_frames, dtype=int)
    
    # Create visualization
    fig, axes = plt.subplots(2, num_frames // 2, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(frame_indices):
        img = trajectory[frame_idx][0]
        img = (img + 1.0) / 2.0
        img = torch.clamp(img[:3], 0.0, 1.0)
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
    
    plt.suptitle(f'Denoising Process ({args.ddim_steps} DDIM steps, CFG={args.cfg_scale})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    DEFAULT_OUTDIR = SCRIPT_DIR / "temp_outputs"
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTDIR
    filename = args.output if args.output else "visualization.png"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to {output_path}")
    
    plt.show()


# Section 8: Command Line Interface
def main():
    parser = argparse.ArgumentParser(
        description="Text-Conditional Sprite Generation with Diffusion Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from single prompt (uses default checkpoint)
  python inference.py --prompt "male human with red cape, 4-view sprite"
  
  # Generate from multiple prompts file
  python inference.py --prompts prompts.txt
  
  # Generate with custom settings
  python inference.py --prompt "female elf with gold armor" --cfg_scale 1.0 --ddim_steps 100
  
  # Save to custom directory
  python inference.py --prompts prompts.txt --output_dir my_sprites
  
  # Visualize denoising process
  python inference.py --mode visualize --prompt "warrior with blue cape"
  
  # Visualize with more steps
  python inference.py --mode visualize --prompt "mage" --ddim_steps 20
  
  # Visualize and save to custom location
  python inference.py --mode visualize --prompt "archer" --output_dir reports/figures --output denoising.png
  
  # Use custom checkpoint
  python inference.py --checkpoint models/checkpoint_epoch_50.pt --prompt "knight"
        """
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="generate",
                       choices=["generate", "visualize"],
                       help="Operation mode: generate sprites or visualize denoising (default: generate)")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str,
                       default=str(SCRIPT_DIR / "checkpoints" / "model_best.pt"),
                       help="Path to model checkpoint (default: checkpoints/model_best.pt)")
    
    # Prompt arguments
    parser.add_argument("--prompt", type=str,
                       help="Single text prompt for sprite generation")
    parser.add_argument("--prompts", type=str,
                       help="Path to file with multiple prompts (one per line)")
    
    # Generation parameters
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                       help="Classifier-free guidance scale (default: 1.0, recommended for stability)")
    parser.add_argument("--ddim_steps", type=int, default=10,
                       help="Number of DDIM sampling steps (default: 10)")
    parser.add_argument("--eta", type=float, default=0.0,
                       help="DDIM eta parameter for stochasticity (default: 0.0 for deterministic)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for generated images (default: temp_outputs)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename for visualization (visualize mode only, default: visualization.png)")
    
    # Visualization parameters
    parser.add_argument("--vis_frames", type=int, default=8,
                       help="Number of frames to show in visualization (default: 8)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.output and (Path(args.output).parent != Path(".")):
        raise ValueError("--output must be a filename only. Use --output_dir to set the directory.")
    
    if args.mode == "generate":
        if args.prompt is None and args.prompts is None:
            parser.error("generate mode requires either --prompt or --prompts")
        generate_sprites(args)
    elif args.mode == "visualize":
        visualize_denoising(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()