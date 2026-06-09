#!/usr/bin/env python
"""Train V-JEPA on LegoGears videos and fine-tune a YOLO-style head."""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

DEFAULT_CLASS_NAMES = ["red light", "pin", "center", "small gear", "medium gear"]
VIDEO_SUFFIXES = {".avi", ".mov", ".mp4", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ANSI_RESET = "\033[0m"
ANSI_COLORS = {
    "green": "\033[38;5;46m",
    "blue": "\033[38;5;33m",
    "cerulean": "\033[38;5;45m",
    "yellow": "\033[38;5;226m",
    "orange": "\033[38;5;208m",
    "light_red": "\033[38;5;203m",
    "dark_red": "\033[38;5;124m",
}


def load_yaml_config(path: str | None) -> dict[str, Any]:
    """Load a nested YAML config."""
    if path is None:
        return {}

    with Path(path).open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}

    if not isinstance(loaded, dict):
        msg = f"Expected top-level YAML mapping in {path}."
        raise TypeError(msg)

    return loaded


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """Apply matching YAML config keys to parsed CLI arguments."""
    config = load_yaml_config(args.config)
    configured_command = config.get("command")
    if configured_command is not None and configured_command != args.command:
        msg = f"Config command {configured_command!r} does not match CLI command {args.command!r}."
        raise ValueError(msg)

    valid_args = set(vars(args))
    for key, value in config.items():
        if key == "command":
            continue

        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if nested_key in valid_args:
                    setattr(args, nested_key, nested_value)
            continue

        if key in valid_args:
            setattr(args, key, value)

    return args


def require_arg(value: Any, name: str) -> Any:
    """Require a config or CLI argument."""
    if value is None:
        msg = f"Missing required argument or config value: {name}"
        raise ValueError(msg)
    return value


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve an explicit or automatic device name."""
    if device_name != "auto":
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint across PyTorch versions."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        msg = f"Expected checkpoint dict at {path}, got {type(checkpoint)!r}."
        raise TypeError(msg)

    return checkpoint


def grade_lower_better(value: float, thresholds: Sequence[float]) -> str:
    """Grade a metric where lower values are better."""
    for grade, threshold in zip(ANSI_COLORS, thresholds, strict=True):
        if value <= threshold:
            return grade
    return "dark_red"


def grade_higher_better(value: float, thresholds: Sequence[float]) -> str:
    """Grade a metric where higher values are better."""
    for grade, threshold in zip(ANSI_COLORS, thresholds, strict=True):
        if value >= threshold:
            return grade
    return "dark_red"


def metric_grade(key: str, value: float) -> str | None:
    """Choose a CLI color grade for a metric value."""
    if key.endswith("loss") or key in {"box", "obj"}:
        return grade_lower_better(value, [0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00])
    if key == "grad":
        return grade_lower_better(value, [0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00])
    if key in {"rank", "t_rank", "p_rank"}:
        return grade_higher_better(value, [0.75, 0.60, 0.45, 0.30, 0.20, 0.10, 0.05])
    if key in {"std", "t_std", "p_std"}:
        return grade_higher_better(value, [0.25, 0.18, 0.12, 0.08, 0.04, 0.02, 0.01])
    if key in {"iou", "precision", "recall", "f1", "map50"}:
        return grade_higher_better(value, [0.85, 0.70, 0.55, 0.40, 0.25, 0.10, 0.03])
    if key == "ema":
        return grade_lower_better(value, [0.01, 0.03, 0.07, 0.15, 0.30, 0.50, 1.00])
    return None


def format_metric(key: str, value: float, *, color: bool) -> str:
    """Format and optionally color one compact metric value."""
    value_text = f"{value:.4f}"
    if not color:
        return value_text

    grade = metric_grade(key, value)
    if grade is None:
        return value_text

    return f"{ANSI_COLORS[grade]}{value_text}{ANSI_RESET}"


def resolve_image_dimensions(
    *,
    image_size: int | None,
    image_height: int | None,
    image_width: int | None,
) -> tuple[int, int]:
    """Resolve rectangular image dimensions, falling back to square image_size."""
    if image_height is None and image_width is None:
        if image_size is None:
            msg = "Either image_size or both image_height and image_width must be set."
            raise ValueError(msg)
        return image_size, image_size

    if image_height is None or image_width is None:
        msg = "image_height and image_width must be set together."
        raise ValueError(msg)

    return image_height, image_width


def load_image_tensor(path: Path, *, image_height: int, image_width: int) -> torch.Tensor:
    """Load an image as a normalized CHW float tensor."""
    with Image.open(path) as raw_image:
        image = raw_image.convert("RGB").resize((image_width, image_height), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(image)
    return (tensor - 0.5) / 0.5


def find_videos(dataset_dir: Path) -> list[Path]:
    """Find source videos in a dataset directory."""
    return sorted(path for path in dataset_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def find_images(dataset_dir: Path) -> list[Path]:
    """Find labeled images recursively."""
    images = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and ":Zone.Identifier" not in path.name
    ]
    return sorted(images)


def normalize_frame_sample_fps(value: Any) -> float | None:
    """Normalize frame sampling configuration; None means all frames."""
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "all", "none", "null"}:
            return None
        value = normalized

    frame_sample_fps = float(value)
    if frame_sample_fps <= 0:
        return None

    return frame_sample_fps


def format_frame_sample_fps(value: float | None) -> str:
    """Format frame sampling configuration for logs and metadata."""
    if value is None:
        return "all"
    return f"{value:g}"


@dataclass(frozen=True)
class VideoFrameRecord:
    """Cached frames for one source video."""

    source_video: Path
    frames: list[Path]


def expected_frame_cache_manifest(
    video: Path,
    *,
    image_height: int,
    image_width: int,
    frame_sample_fps: float | None,
) -> dict[str, Any]:
    """Build metadata that identifies a reusable frame cache."""
    stat = video.stat()
    return {
        "source_video": str(video.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "image_height": image_height,
        "image_width": image_width,
        "frame_sample_fps": format_frame_sample_fps(frame_sample_fps),
    }


def frame_cache_matches(manifest_path: Path, expected_manifest: dict[str, Any]) -> bool:
    """Check whether a cached frame directory matches the current extraction settings."""
    if not manifest_path.exists():
        return False

    try:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return False

    return loaded == expected_manifest


def cache_video_frames(
    *,
    videos: Sequence[Path],
    cache_dir: Path,
    image_height: int,
    image_width: int,
    frame_sample_fps: float | None,
    rebuild: bool,
) -> list[VideoFrameRecord]:
    """Extract resized video frames with ffmpeg and return cache records."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        msg = "ffmpeg is required for LegoGears V-JEPA video frame extraction."
        raise RuntimeError(msg)

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[VideoFrameRecord] = []

    for video in videos:
        video_cache_dir = cache_dir / video.stem
        manifest_path = video_cache_dir / "manifest.yaml"
        expected_manifest = expected_frame_cache_manifest(
            video,
            image_height=image_height,
            image_width=image_width,
            frame_sample_fps=frame_sample_fps,
        )
        if rebuild and video_cache_dir.exists():
            shutil.rmtree(video_cache_dir)
        video_cache_dir.mkdir(parents=True, exist_ok=True)

        existing_frames = sorted(video_cache_dir.glob("frame_*.jpg"))
        if existing_frames and not frame_cache_matches(manifest_path, expected_manifest):
            shutil.rmtree(video_cache_dir)
            video_cache_dir.mkdir(parents=True, exist_ok=True)
            existing_frames = []

        if not existing_frames:
            output_pattern = video_cache_dir / "frame_%06d.jpg"
            video_filter = f"scale={image_width}:{image_height}"
            if frame_sample_fps is not None:
                video_filter = f"fps={frame_sample_fps:g},{video_filter}"

            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video),
                    "-vf",
                    video_filter,
                    str(output_pattern),
                ],
                check=True,
            )
            existing_frames = sorted(video_cache_dir.glob("frame_*.jpg"))
            manifest_path.write_text(yaml.safe_dump(expected_manifest, sort_keys=True), encoding="utf-8")

        if not existing_frames:
            msg = f"No frames were extracted for {video}."
            raise RuntimeError(msg)

        records.append(VideoFrameRecord(source_video=video, frames=existing_frames))

    return records


class VideoClipDataset(Dataset[torch.Tensor]):
    """Sample fixed-length clips from cached video frames."""

    def __init__(
        self,
        records: Sequence[VideoFrameRecord],
        *,
        image_height: int,
        image_width: int,
        num_frames: int,
        temporal_stride: int,
    ) -> None:
        self.records = list(records)
        self.image_height = image_height
        self.image_width = image_width
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.clip_span = (num_frames - 1) * temporal_stride + 1
        self.samples: list[tuple[int, int]] = []

        for record_index, record in enumerate(self.records):
            max_start = len(record.frames) - self.clip_span
            if max_start < 0:
                continue
            step = max(1, num_frames)
            self.samples.extend((record_index, start) for start in range(0, max_start + 1, step))

        if not self.samples:
            msg = "No video clips are available after frame extraction."
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        record_index, start = self.samples[index]
        frames = self.records[record_index].frames
        clip = [
            load_image_tensor(
                frames[start + frame_index * self.temporal_stride],
                image_height=self.image_height,
                image_width=self.image_width,
            )
            for frame_index in range(self.num_frames)
        ]
        return torch.stack(clip, dim=0)


def sincos_1d(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """Build one-dimensional sinusoidal embeddings for scalar coordinates."""
    if dim <= 0:
        return coords.new_zeros((coords.numel(), 0))

    half_dim = max(1, dim // 2)
    exponent = torch.arange(half_dim, dtype=coords.dtype, device=coords.device)
    exponent /= max(1, half_dim - 1)
    frequencies = torch.exp(-math.log(10000.0) * exponent)
    angles = coords.unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    if embedding.size(1) < dim:
        padding = coords.new_zeros((coords.numel(), dim - embedding.size(1)))
        embedding = torch.cat([embedding, padding], dim=1)
    return embedding[:, :dim]


def build_3d_sincos_position_embedding(grid_t: int, grid_h: int, grid_w: int, embed_dim: int) -> torch.Tensor:
    """Build a fixed 3D sin-cos positional embedding."""
    coords_t, coords_y, coords_x = torch.meshgrid(
        torch.linspace(0.0, 2.0 * math.pi, grid_t),
        torch.linspace(0.0, 2.0 * math.pi, grid_h),
        torch.linspace(0.0, 2.0 * math.pi, grid_w),
        indexing="ij",
    )
    coords = [coords_t.flatten(), coords_y.flatten(), coords_x.flatten()]
    base_dim = embed_dim // 3
    dims = [base_dim, base_dim, embed_dim - (2 * base_dim)]
    embeddings = []
    for axis_coords, dim in zip(coords, dims, strict=True):
        embeddings.append(sincos_1d(axis_coords, dim))
    embedding = torch.cat(embeddings, dim=1)
    return embedding.unsqueeze(0)


class VideoPatchEmbed(nn.Module):
    """Convert a video clip into tubelet tokens."""

    def __init__(
        self,
        *,
        image_size: int | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
        num_frames: int,
        patch_size: int,
        tubelet_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        resolved_height, resolved_width = resolve_image_dimensions(
            image_size=image_size,
            image_height=image_height,
            image_width=image_width,
        )
        if resolved_height % patch_size != 0:
            msg = f"image_height={resolved_height} must be divisible by patch_size={patch_size}."
            raise ValueError(msg)
        if resolved_width % patch_size != 0:
            msg = f"image_width={resolved_width} must be divisible by patch_size={patch_size}."
            raise ValueError(msg)
        if num_frames % tubelet_size != 0:
            msg = f"num_frames={num_frames} must be divisible by tubelet_size={tubelet_size}."
            raise ValueError(msg)

        self.grid_t = num_frames // tubelet_size
        self.grid_h = resolved_height // patch_size
        self.grid_w = resolved_width // patch_size
        self.num_patches = self.grid_t * self.grid_h * self.grid_w
        self.proj = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        clips = clips.permute(0, 2, 1, 3, 4)
        patches = self.proj(clips)
        return patches.flatten(2).transpose(1, 2)


class VideoTransformerEncoder(nn.Module):
    """Small ViT-style encoder over video tubelet tokens."""

    def __init__(
        self,
        *,
        image_size: int | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
        num_frames: int,
        patch_size: int,
        tubelet_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        resolved_height, resolved_width = resolve_image_dimensions(
            image_size=image_size,
            image_height=image_height,
            image_width=image_width,
        )
        self.image_height = resolved_height
        self.image_width = resolved_width
        self.num_frames = num_frames
        self.patch_embed = VideoPatchEmbed(
            image_height=resolved_height,
            image_width=resolved_width,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
        )
        self.embed_dim = embed_dim
        self.grid_t = self.patch_embed.grid_t
        self.grid_h = self.patch_embed.grid_h
        self.grid_w = self.patch_embed.grid_w
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "pos_embed",
            build_3d_sincos_position_embedding(self.grid_t, self.grid_h, self.grid_w, embed_dim),
            persistent=False,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, clips: torch.Tensor, keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.patch_embed(clips)

        if keep_mask is not None:
            mask_token = self.mask_token.expand(tokens.size(0), tokens.size(1), -1)
            tokens = torch.where(keep_mask.unsqueeze(-1), tokens, mask_token)

        tokens += self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        return self.norm(self.encoder(tokens))


class VideoJEPA(nn.Module):
    """V-JEPA model with online and EMA target video encoders."""

    def __init__(self, encoder_config: dict[str, Any]) -> None:
        super().__init__()
        self.online_encoder = VideoTransformerEncoder(**encoder_config)
        self.target_encoder = VideoTransformerEncoder(**encoder_config)
        embed_dim = int(encoder_config["embed_dim"])
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sync_target()
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def sync_target(self) -> None:
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())

    def forward(self, clips: torch.Tensor, keep_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        online_tokens = self.online_encoder(clips, keep_mask=keep_mask)
        with torch.no_grad():
            target_tokens = self.target_encoder(clips)
        return self.predictor(online_tokens), target_tokens.detach()


def trainable_jepa_parameters(model: VideoJEPA) -> Iterable[nn.Parameter]:
    """Return online encoder and predictor parameters."""
    yield from model.online_encoder.parameters()
    yield from model.predictor.parameters()


@torch.no_grad()
def update_ema(model: VideoJEPA, tau: float) -> None:
    """Update target encoder weights from online encoder weights."""
    for online_param, target_param in zip(model.online_encoder.parameters(), model.target_encoder.parameters(), strict=True):
        target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)


@torch.no_grad()
def ema_relative_l2(model: VideoJEPA) -> float:
    """Measure relative online/target parameter distance."""
    device = next(model.online_encoder.parameters()).device
    distance_sq = torch.zeros((), device=device)
    target_sq = torch.zeros_like(distance_sq)
    for online_param, target_param in zip(model.online_encoder.parameters(), model.target_encoder.parameters(), strict=True):
        distance_sq += torch.sum((online_param - target_param) ** 2)
        target_sq += torch.sum(target_param**2)
    return float((distance_sq.sqrt() / target_sq.sqrt().clamp_min(1e-12)).detach().cpu().item())


def sample_spatial_tube_group(
    *,
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    target_ratio: float,
    block_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample a spatial tube mask that spans the temporal grid."""
    target_spatial_count = max(1, min((grid_h * grid_w) - 1, round(grid_h * grid_w * target_ratio)))
    block_h = max(1, min(grid_h, round(grid_h * block_scale)))
    block_w = max(1, min(grid_w, round(grid_w * block_scale)))
    spatial_mask = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.bool, device=device)

    for batch_index in range(batch_size):
        attempts = 0
        while int(spatial_mask[batch_index].sum().item()) < target_spatial_count and attempts < 256:
            top = torch.randint(0, grid_h - block_h + 1, (), device=device).item()
            left = torch.randint(0, grid_w - block_w + 1, (), device=device).item()
            spatial_mask[batch_index, top : top + block_h, left : left + block_w] = True
            attempts += 1

        extra = int(spatial_mask[batch_index].sum().item()) - target_spatial_count
        if extra > 0:
            indices = torch.nonzero(spatial_mask[batch_index].flatten(), as_tuple=False).flatten()
            spatial_mask[batch_index].flatten()[indices[:extra]] = False

    return spatial_mask.flatten(1).unsqueeze(1).expand(-1, grid_t, -1).reshape(batch_size, grid_t * grid_h * grid_w)


def sample_vjepa_masks(
    *,
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    short_mask_ratio: float,
    long_mask_ratio: float,
    short_block_scale: float,
    long_block_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample short and long target tube masks plus the remaining context mask."""
    short_mask = sample_spatial_tube_group(
        batch_size=batch_size,
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        target_ratio=short_mask_ratio,
        block_scale=short_block_scale,
        device=device,
    )
    long_mask = sample_spatial_tube_group(
        batch_size=batch_size,
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        target_ratio=long_mask_ratio,
        block_scale=long_block_scale,
        device=device,
    )
    long_mask &= ~short_mask

    target_union = short_mask | long_mask
    for batch_index in range(batch_size):
        if bool((~target_union[batch_index]).any().item()):
            continue
        short_mask[batch_index, -1] = False
        long_mask[batch_index, -1] = False
        target_union[batch_index, -1] = False

    keep_mask = ~target_union
    return keep_mask, short_mask, long_mask


def normalized_l1_loss(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute L1 loss on layer-normalized latent tokens."""
    if not bool(mask.any().item()):
        return pred_tokens.new_zeros(())

    embed_dim = pred_tokens.size(-1)
    pred_values = F.layer_norm(pred_tokens[mask], (embed_dim,))
    target_values = F.layer_norm(target_tokens[mask], (embed_dim,))
    return F.l1_loss(pred_values, target_values)


def covariance_effective_rank(values: torch.Tensor) -> float:
    """Compute normalized effective rank for flattened token features."""
    if values.size(0) < 2:
        return 0.0

    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (values.size(0) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0

    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    rank = entropy.exp() / values.size(1)
    return float(rank.detach().cpu().item())


def tensor_scalar(value: torch.Tensor) -> float:
    """Convert a scalar tensor to a Python float."""
    return float(value.detach().float().cpu().item())


def ema_tau_for_step(step: int, total_steps: int, *, start: float, end: float) -> float:
    """Linearly move EMA tau from start toward end."""
    if total_steps <= 1:
        return end
    progress = min(1.0, step / (total_steps - 1))
    return start + ((end - start) * progress)


def run_pretrain(args: argparse.Namespace) -> None:
    """Run V-JEPA pretraining on LegoGears videos."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_dir = Path(require_arg(args.dataset_dir, "dataset_dir"))
    output_dir = Path(args.output_dir)
    run_name = require_arg(args.run_name, "run_name")
    videos = find_videos(dataset_dir)
    if not videos:
        msg = f"No videos found in {dataset_dir}."
        raise ValueError(msg)
    image_height, image_width = resolve_image_dimensions(
        image_size=args.image_size,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    frame_sample_fps = normalize_frame_sample_fps(args.frame_sample_fps)

    if args.dry_run:
        print(
            "vjepa-pretrain "
            f"videos={len(videos)} dataset_dir={dataset_dir} output_dir={output_dir} run_name={run_name} "
            f"input={image_width}x{image_height} frame_sample_fps={format_frame_sample_fps(frame_sample_fps)} "
            f"num_frames={args.num_frames} patch_size={args.patch_size} "
            f"tubelet_size={args.tubelet_size} batch_size={args.batch_size} epochs={args.epochs} device={device}",
        )
        return

    records = cache_video_frames(
        videos=videos,
        cache_dir=Path(args.frame_cache_dir),
        image_height=image_height,
        image_width=image_width,
        frame_sample_fps=frame_sample_fps,
        rebuild=args.rebuild_frame_cache,
    )
    dataset = VideoClipDataset(
        records,
        image_height=image_height,
        image_width=image_width,
        num_frames=args.num_frames,
        temporal_stride=args.temporal_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    encoder_config = {
        "image_height": image_height,
        "image_width": image_width,
        "num_frames": args.num_frames,
        "patch_size": args.patch_size,
        "tubelet_size": args.tubelet_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
    }
    model = VideoJEPA(encoder_config).to(device)
    trainable_params = list(trainable_jepa_parameters(model))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    run_dir = output_dir / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_steps = max(1, args.epochs * len(loader))
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(loader, desc=f"vjepa lego epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for step, clip_batch in enumerate(progress):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

            clips = clip_batch.to(device, non_blocking=True)
            keep_mask, short_mask, long_mask = sample_vjepa_masks(
                batch_size=clips.size(0),
                grid_t=model.online_encoder.grid_t,
                grid_h=model.online_encoder.grid_h,
                grid_w=model.online_encoder.grid_w,
                short_mask_ratio=args.short_mask_ratio,
                long_mask_ratio=args.long_mask_ratio,
                short_block_scale=args.short_block_scale,
                long_block_scale=args.long_block_scale,
                device=device,
            )
            pred_tokens, target_tokens = model(clips, keep_mask)
            short_loss = normalized_l1_loss(pred_tokens, target_tokens, short_mask)
            long_loss = normalized_l1_loss(pred_tokens, target_tokens, long_mask)
            loss = short_loss + long_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
            optimizer.step()
            tau = ema_tau_for_step(global_step, total_steps, start=args.ema_tau_start, end=args.ema_tau_end)
            update_ema(model, tau)

            metrics = {
                "loss": tensor_scalar(loss),
                "short_loss": tensor_scalar(short_loss),
                "long_loss": tensor_scalar(long_loss),
                "grad": tensor_scalar(grad_norm),
                "ema": ema_relative_l2(model),
                "short_ratio": tensor_scalar(short_mask.float().mean()),
                "long_ratio": tensor_scalar(long_mask.float().mean()),
                "ctx": tensor_scalar(keep_mask.float().mean()),
            }
            if args.cli_log_every_steps > 0 and (global_step == 0 or (global_step + 1) % args.cli_log_every_steps == 0):
                target_union = short_mask | long_mask
                metrics["t_std"] = tensor_scalar(target_tokens[target_union].detach().float().std(unbiased=False))
                metrics["p_std"] = tensor_scalar(pred_tokens[target_union].detach().float().std(unbiased=False))
                metrics["t_rank"] = covariance_effective_rank(target_tokens[target_union].detach())
                metrics["p_rank"] = covariance_effective_rank(pred_tokens[target_union].detach())
                progress.write(
                    " | ".join(
                        [
                            f"global_step={global_step + 1}",
                            f"loss={format_metric('loss', metrics['loss'], color=args.color_cli)}",
                            f"short={format_metric('loss', metrics['short_loss'], color=args.color_cli)}",
                            f"long={format_metric('loss', metrics['long_loss'], color=args.color_cli)}",
                            f"grad={format_metric('grad', metrics['grad'], color=args.color_cli)}",
                            f"t_std={format_metric('t_std', metrics['t_std'], color=args.color_cli)}",
                            f"p_std={format_metric('p_std', metrics['p_std'], color=args.color_cli)}",
                            f"t_rank={format_metric('t_rank', metrics['t_rank'], color=args.color_cli)}",
                            f"p_rank={format_metric('p_rank', metrics['p_rank'], color=args.color_cli)}",
                            f"ema={format_metric('ema', metrics['ema'], color=args.color_cli)}",
                        ],
                    ),
                )

            progress.set_postfix(
                loss=format_metric("loss", metrics["loss"], color=args.color_cli),
                short=format_metric("loss", metrics["short_loss"], color=args.color_cli),
                long=format_metric("loss", metrics["long_loss"], color=args.color_cli),
                grad=format_metric("grad", metrics["grad"], color=args.color_cli),
                ctx=f"{metrics['ctx']:.3f}",
            )
            global_step += 1

    checkpoint_path = checkpoint_dir / "vjepa_pretrain_last.pt"
    torch.save(
        {
            "dataset": "LegoGears",
            "dataset_dir": str(dataset_dir),
            "model_config": encoder_config,
            "online_encoder": model.online_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved V-JEPA checkpoint: {checkpoint_path}")


@dataclass(frozen=True)
class DetectionTarget:
    """YOLO boxes and labels for one image."""

    boxes: torch.Tensor
    labels: torch.Tensor
    path: Path


class LegoDetectionDataset(Dataset[tuple[torch.Tensor, DetectionTarget]]):
    """Image dataset with YOLO-format LegoGears labels."""

    def __init__(self, *, dataset_dir: Path, image_height: int, image_width: int) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.images = find_images(dataset_dir)
        if not self.images:
            msg = f"No labeled images found in {dataset_dir}."
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectionTarget]:
        image_path = self.images[index]
        boxes, labels = read_yolo_labels(image_path.with_suffix(".txt"))
        return (
            load_image_tensor(image_path, image_height=self.image_height, image_width=self.image_width),
            DetectionTarget(boxes=boxes, labels=labels, path=image_path),
        )


def read_yolo_labels(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read normalized YOLO labels as class ids and xywh boxes."""
    if not path.exists() or path.stat().st_size == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

    boxes: list[list[float]] = []
    labels: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            msg = f"Expected 5 YOLO fields in {path}:{line_number}, got {len(parts)}."
            raise ValueError(msg)
        class_id = int(float(parts[0]))
        cx, cy, width, height = (float(part) for part in parts[1:])
        labels.append(class_id)
        boxes.append(
            [
                min(1.0, max(0.0, cx)),
                min(1.0, max(0.0, cy)),
                min(1.0, max(0.0, width)),
                min(1.0, max(0.0, height)),
            ],
        )

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def collate_detection_batch(
    batch: Sequence[tuple[torch.Tensor, DetectionTarget]],
) -> tuple[torch.Tensor, list[DetectionTarget]]:
    """Collate images while keeping variable-length targets as a list."""
    images, targets = zip(*batch, strict=True)
    return torch.stack(list(images), dim=0), list(targets)


class LegoYoloDetector(nn.Module):
    """Small class-specific YOLO-style head on top of a V-JEPA encoder."""

    def __init__(self, encoder: VideoTransformerEncoder, *, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Conv2d(encoder.embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, num_classes * 5, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        clips = images.unsqueeze(1).expand(-1, self.encoder.num_frames, -1, -1, -1).contiguous()
        tokens = self.encoder(clips)
        features = tokens.view(
            images.size(0),
            self.encoder.grid_t,
            self.encoder.grid_h,
            self.encoder.grid_w,
            self.encoder.embed_dim,
        )
        features = features.mean(dim=1).permute(0, 3, 1, 2).contiguous()
        output = self.head(features)
        return output.view(images.size(0), self.num_classes, 5, self.encoder.grid_h, self.encoder.grid_w)


def build_detection_targets(
    targets: Sequence[DetectionTarget],
    *,
    grid_h: int,
    grid_w: int,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project normalized YOLO boxes onto a class-specific prediction grid."""
    batch_size = len(targets)
    objectness = torch.zeros(batch_size, num_classes, grid_h, grid_w, device=device)
    boxes = torch.zeros(batch_size, num_classes, 4, grid_h, grid_w, device=device)
    positives = torch.zeros(batch_size, num_classes, grid_h, grid_w, dtype=torch.bool, device=device)

    for batch_index, target in enumerate(targets):
        target_boxes = target.boxes.to(device)
        target_labels = target.labels.to(device)
        for box, label in zip(target_boxes, target_labels, strict=True):
            class_id = int(label.item())
            if class_id < 0 or class_id >= num_classes:
                continue
            cell_x = min(grid_w - 1, max(0, int(float(box[0].item()) * grid_w)))
            cell_y = min(grid_h - 1, max(0, int(float(box[1].item()) * grid_h)))
            objectness[batch_index, class_id, cell_y, cell_x] = 1.0
            boxes[batch_index, class_id, :, cell_y, cell_x] = box
            positives[batch_index, class_id, cell_y, cell_x] = True

    return objectness, boxes, positives


def detection_loss(
    predictions: torch.Tensor,
    targets: Sequence[DetectionTarget],
    *,
    positive_weight: float,
    box_loss_weight: float,
    obj_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute objectness and box loss for the lightweight YOLO head."""
    device = predictions.device
    num_classes = predictions.size(1)
    objectness_target, box_target, positives = build_detection_targets(
        targets,
        grid_h=predictions.size(-2),
        grid_w=predictions.size(-1),
        num_classes=num_classes,
        device=device,
    )
    objectness_logits = predictions[:, :, 0]
    raw_boxes = predictions[:, :, 1:5]
    obj_loss_raw = F.binary_cross_entropy_with_logits(objectness_logits, objectness_target, reduction="none")
    weights = torch.where(objectness_target.bool(), positive_weight, 1.0)
    obj_loss = (obj_loss_raw * weights).mean()

    if bool(positives.any().item()):
        pred_boxes = raw_boxes.sigmoid().permute(0, 1, 3, 4, 2)[positives]
        true_boxes = box_target.permute(0, 1, 3, 4, 2)[positives]
        box_loss = F.smooth_l1_loss(pred_boxes, true_boxes)
    else:
        box_loss = predictions.new_zeros(())

    loss = (obj_loss_weight * obj_loss) + (box_loss_weight * box_loss)
    return loss, {"obj": tensor_scalar(obj_loss), "box": tensor_scalar(box_loss), "loss": tensor_scalar(loss)}


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized xywh boxes to xyxy boxes."""
    cx, cy, width, height = boxes.unbind(dim=-1)
    half_w = width / 2
    half_h = height / 2
    return torch.stack(
        [
            (cx - half_w).clamp(0.0, 1.0),
            (cy - half_h).clamp(0.0, 1.0),
            (cx + half_w).clamp(0.0, 1.0),
            (cy + half_h).clamp(0.0, 1.0),
        ],
        dim=-1,
    )


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU between one xywh box and many xywh boxes."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))

    box_xyxy = xywh_to_xyxy(box.unsqueeze(0))[0]
    boxes_xyxy = xywh_to_xyxy(boxes)
    top_left = torch.maximum(box_xyxy[:2], boxes_xyxy[:, :2])
    bottom_right = torch.minimum(box_xyxy[2:], boxes_xyxy[:, 2:])
    wh = (bottom_right - top_left).clamp_min(0.0)
    intersection = wh[:, 0] * wh[:, 1]
    box_area = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
    boxes_area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    return intersection / (box_area + boxes_area - intersection).clamp_min(1e-12)


@dataclass(frozen=True)
class DecodedPrediction:
    """One decoded detector prediction."""

    score: float
    label: int
    box: torch.Tensor


def decode_predictions(
    predictions: torch.Tensor,
    *,
    score_threshold: float,
    max_predictions_per_image: int,
) -> list[list[DecodedPrediction]]:
    """Decode class-grid predictions into normalized boxes."""
    object_scores = predictions[:, :, 0].sigmoid()
    boxes = predictions[:, :, 1:5].sigmoid()
    decoded: list[list[DecodedPrediction]] = []

    for batch_index in range(predictions.size(0)):
        image_predictions: list[DecodedPrediction] = []
        for class_id in range(predictions.size(1)):
            class_scores = object_scores[batch_index, class_id]
            ys, xs = torch.nonzero(class_scores >= score_threshold, as_tuple=True)
            for y, x in zip(ys, xs, strict=True):
                image_predictions.append(
                    DecodedPrediction(
                        score=float(class_scores[y, x].detach().cpu().item()),
                        label=class_id,
                        box=boxes[batch_index, class_id, :, y, x].detach().cpu(),
                    ),
                )

        image_predictions.sort(key=lambda prediction: prediction.score, reverse=True)
        decoded.append(image_predictions[:max_predictions_per_image])

    return decoded


def match_predictions(
    predictions: Sequence[DecodedPrediction],
    target: DetectionTarget,
    *,
    iou_threshold: float,
) -> tuple[int, int, int, float]:
    """Greedily match predictions to targets by class and IoU."""
    matched_target_indices: set[int] = set()
    true_positive = 0
    false_positive = 0
    matched_iou_total = 0.0
    target_boxes = target.boxes.cpu()
    target_labels = target.labels.cpu()

    for prediction in predictions:
        candidate_indices = torch.nonzero(target_labels == prediction.label, as_tuple=False).flatten()
        candidate_indices = torch.tensor(
            [index for index in candidate_indices.tolist() if index not in matched_target_indices],
            dtype=torch.long,
        )
        if candidate_indices.numel() == 0:
            false_positive += 1
            continue

        ious = box_iou(prediction.box, target_boxes[candidate_indices])
        best_iou, best_offset = ious.max(dim=0)
        if float(best_iou.item()) >= iou_threshold:
            true_positive += 1
            matched_iou_total += float(best_iou.item())
            matched_target_indices.add(int(candidate_indices[int(best_offset.item())].item()))
        else:
            false_positive += 1

    false_negative = len(target_boxes) - len(matched_target_indices)
    return true_positive, false_positive, false_negative, matched_iou_total


def evaluate_detector(
    model: LegoYoloDetector,
    loader: DataLoader[tuple[torch.Tensor, list[DetectionTarget]]],
    *,
    device: torch.device,
    positive_weight: float,
    box_loss_weight: float,
    obj_loss_weight: float,
    score_threshold: float,
    iou_threshold: float,
    max_predictions_per_image: int,
    max_batches: int | None,
) -> dict[str, float]:
    """Evaluate detector loss and mAP50-style matching metrics."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    matched_iou_total = 0.0

    with torch.no_grad():
        for step, (image_batch, targets) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break

            images = image_batch.to(device, non_blocking=True)
            predictions = model(images)
            loss, _ = detection_loss(
                predictions,
                targets,
                positive_weight=positive_weight,
                box_loss_weight=box_loss_weight,
                obj_loss_weight=obj_loss_weight,
            )
            total_loss += tensor_scalar(loss)
            total_batches += 1

            decoded = decode_predictions(
                predictions,
                score_threshold=score_threshold,
                max_predictions_per_image=max_predictions_per_image,
            )
            for image_predictions, target in zip(decoded, targets, strict=True):
                tp, fp, fn, matched_iou = match_predictions(
                    image_predictions,
                    target,
                    iou_threshold=iou_threshold,
                )
                true_positive += tp
                false_positive += fp
                false_negative += fn
                matched_iou_total += matched_iou

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)
    mean_iou = matched_iou_total / max(1, true_positive)
    return {
        "loss": total_loss / max(1, total_batches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": mean_iou,
        "map50": recall,
    }


def run_detect(args: argparse.Namespace) -> None:
    """Fine-tune a YOLO-style head from a V-JEPA checkpoint."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_dir = Path(require_arg(args.dataset_dir, "dataset_dir"))
    output_dir = Path(args.output_dir)
    run_name = require_arg(args.run_name, "run_name")
    checkpoint_path = Path(require_arg(args.checkpoint, "checkpoint"))
    class_names = args.class_names or DEFAULT_CLASS_NAMES
    image_count = len(find_images(dataset_dir))

    if args.dry_run:
        print(
            "vjepa-yolo-finetune "
            f"images={image_count} dataset_dir={dataset_dir} output_dir={output_dir} run_name={run_name} "
            f"checkpoint={checkpoint_path} classes={len(class_names)} epochs={args.epochs} "
            f"batch_size={args.batch_size} device={device}",
        )
        return

    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    encoder_config = dict(checkpoint["model_config"])
    image_height, image_width = resolve_image_dimensions(
        image_size=encoder_config.get("image_size"),
        image_height=encoder_config.get("image_height"),
        image_width=encoder_config.get("image_width"),
    )
    dataset = LegoDetectionDataset(dataset_dir=dataset_dir, image_height=image_height, image_width=image_width)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection_batch,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection_batch,
    )

    encoder = VideoTransformerEncoder(**encoder_config)
    encoder.load_state_dict(checkpoint["online_encoder"])
    model = LegoYoloDetector(encoder, num_classes=len(class_names), hidden_dim=args.detector_hidden_dim).to(device)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_dir = output_dir / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"vjepa-yolo lego epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for step, (image_batch, targets) in enumerate(progress):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

            images = image_batch.to(device, non_blocking=True)
            predictions = model(images)
            loss, metrics = detection_loss(
                predictions,
                targets,
                positive_weight=args.positive_weight,
                box_loss_weight=args.box_loss_weight,
                obj_loss_weight=args.obj_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            progress.set_postfix(
                loss=format_metric("loss", metrics["loss"], color=args.color_cli),
                obj=format_metric("obj", metrics["obj"], color=args.color_cli),
                box=format_metric("box", metrics["box"], color=args.color_cli),
                grad=format_metric("grad", tensor_scalar(grad_norm), color=args.color_cli),
            )

        val_metrics = evaluate_detector(
            model,
            val_loader,
            device=device,
            positive_weight=args.positive_weight,
            box_loss_weight=args.box_loss_weight,
            obj_loss_weight=args.obj_loss_weight,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            max_predictions_per_image=args.max_predictions_per_image,
            max_batches=args.max_val_batches,
        )
        print(
            " | ".join(
                [
                    f"epoch={epoch + 1}",
                    f"val_loss={format_metric('loss', val_metrics['loss'], color=args.color_cli)}",
                    f"precision={format_metric('precision', val_metrics['precision'], color=args.color_cli)}",
                    f"recall={format_metric('recall', val_metrics['recall'], color=args.color_cli)}",
                    f"f1={format_metric('f1', val_metrics['f1'], color=args.color_cli)}",
                    f"iou={format_metric('iou', val_metrics['iou'], color=args.color_cli)}",
                    f"map50_like={format_metric('map50', val_metrics['map50'], color=args.color_cli)}",
                ],
            ),
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "dataset": "LegoGears",
                    "dataset_dir": str(dataset_dir),
                    "source_checkpoint": str(checkpoint_path),
                    "class_names": class_names,
                    "encoder_config": encoder_config,
                    "model": model.state_dict(),
                    "val_metrics": val_metrics,
                },
                checkpoint_dir / "vjepa_yolo_best.pt",
            )

    checkpoint_last = checkpoint_dir / "vjepa_yolo_last.pt"
    torch.save(
        {
            "dataset": "LegoGears",
            "dataset_dir": str(dataset_dir),
            "source_checkpoint": str(checkpoint_path),
            "class_names": class_names,
            "encoder_config": encoder_config,
            "model": model.state_dict(),
            "best_f1": best_f1,
        },
        checkpoint_last,
    )
    print(f"Saved V-JEPA YOLO checkpoint: {checkpoint_last}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared by both commands."""
    parser.add_argument("--config")
    parser.add_argument("--dataset-dir", default="datasets/LegoGears")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--color-cli", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain", help="Pretrain V-JEPA on LegoGears videos.")
    add_common_args(pretrain)
    pretrain.add_argument("--frame-cache-dir", default="runs/cache/lego_gears_frames")
    pretrain.add_argument("--frame-sample-fps", type=normalize_frame_sample_fps)
    pretrain.add_argument("--rebuild-frame-cache", action="store_true")
    pretrain.add_argument("--image-size", type=int, default=128)
    pretrain.add_argument("--image-height", type=int)
    pretrain.add_argument("--image-width", type=int)
    pretrain.add_argument("--num-frames", type=int, default=8)
    pretrain.add_argument("--temporal-stride", type=int, default=1)
    pretrain.add_argument("--patch-size", type=int, default=20)
    pretrain.add_argument("--tubelet-size", type=int, default=2)
    pretrain.add_argument("--embed-dim", type=int, default=128)
    pretrain.add_argument("--depth", type=int, default=4)
    pretrain.add_argument("--num-heads", type=int, default=4)
    pretrain.add_argument("--mlp-ratio", type=float, default=4.0)
    pretrain.add_argument("--short-mask-ratio", type=float, default=0.15)
    pretrain.add_argument("--long-mask-ratio", type=float, default=0.35)
    pretrain.add_argument("--short-block-scale", type=float, default=0.25)
    pretrain.add_argument("--long-block-scale", type=float, default=0.50)
    pretrain.add_argument("--ema-tau-start", type=float, default=0.998)
    pretrain.add_argument("--ema-tau-end", type=float, default=1.0)
    pretrain.add_argument("--cli-log-every-steps", type=int, default=10)
    pretrain.set_defaults(func=run_pretrain)

    detect = subparsers.add_parser("detect", help="Fine-tune a YOLO-style head from V-JEPA features.")
    add_common_args(detect)
    detect.add_argument("--checkpoint")
    detect.add_argument("--class-names", nargs="+")
    detect.add_argument("--detector-hidden-dim", type=int, default=128)
    detect.add_argument("--positive-weight", type=float, default=8.0)
    detect.add_argument("--box-loss-weight", type=float, default=5.0)
    detect.add_argument("--obj-loss-weight", type=float, default=1.0)
    detect.add_argument("--score-threshold", type=float, default=0.35)
    detect.add_argument("--iou-threshold", type=float, default=0.50)
    detect.add_argument("--max-predictions-per-image", type=int, default=25)
    detect.add_argument("--max-val-batches", type=int)
    detect.add_argument("--freeze-encoder", action="store_true")
    detect.set_defaults(func=run_detect)

    return parser.parse_args()


def main() -> None:
    """Run the selected command."""
    start_time = time.perf_counter()
    args = apply_config(parse_args())
    args.func(args)
    elapsed = time.perf_counter() - start_time
    print(f"elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
