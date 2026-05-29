#!/usr/bin/env python
"""Small runnable I-JEPA pretraining and supervised fine-tuning driver."""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, STL10
from tqdm.auto import tqdm

DATASET_IMAGE_SIZE = {
    "cifar10": 32,
    "stl10": 96,
}
MIN_COVARIANCE_SAMPLES = 2
DEAD_DIM_STD_THRESHOLD = 1e-4
ANSI_RESET = "\033[0m"
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
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
    """Load a YAML config file."""
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
    """Apply a nested YAML config to parsed arguments."""
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
    """Require a config or CLI argument before using it."""
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


def image_transforms(image_size: int, *, train: bool) -> transforms.Compose:
    """Build image transforms for pretraining and fine-tuning."""
    steps: list[nn.Module] = []

    if train:
        steps.extend(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
            ],
        )
    else:
        steps.append(transforms.Resize((image_size, image_size)))

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ],
    )
    return transforms.Compose(steps)


def build_pretrain_dataset(dataset: str, data_dir: Path, image_size: int) -> CIFAR10 | STL10:
    """Build a self-supervised image dataset and download it if needed."""
    transform = image_transforms(image_size, train=True)

    if dataset == "cifar10":
        return CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    if dataset == "stl10":
        return STL10(root=data_dir, split="unlabeled", download=True, transform=transform)

    msg = f"Unsupported dataset: {dataset}"
    raise ValueError(msg)


def build_finetune_datasets(dataset: str, data_dir: Path, image_size: int) -> tuple[CIFAR10 | STL10, CIFAR10 | STL10]:
    """Build supervised train and validation datasets and download them if needed."""
    train_transform = image_transforms(image_size, train=True)
    val_transform = image_transforms(image_size, train=False)

    if dataset == "cifar10":
        train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
        return train_dataset, val_dataset

    if dataset == "stl10":
        train_dataset = STL10(root=data_dir, split="train", download=True, transform=train_transform)
        val_dataset = STL10(root=data_dir, split="test", download=True, transform=val_transform)
        return train_dataset, val_dataset

    msg = f"Unsupported dataset: {dataset}"
    raise ValueError(msg)


class PatchEmbed(nn.Module):
    """Convert an image into patch tokens."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            msg = f"image_size={image_size} must be divisible by patch_size={patch_size}."
            raise ValueError(msg)

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.proj(images)
        return patches.flatten(2).transpose(1, 2)


class TinyViTEncoder(nn.Module):
    """A compact ViT encoder suitable for local CIFAR-10 and STL-10 runs."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, images: torch.Tensor, keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.patch_embed(images)

        if keep_mask is not None:
            mask_token = self.mask_token.expand(tokens.size(0), tokens.size(1), -1)
            tokens = torch.where(keep_mask.unsqueeze(-1), tokens, mask_token)

        tokens += self.pos_embed
        return self.norm(self.encoder(tokens))


class IJEPAForImages(nn.Module):
    """Minimal image JEPA model with online and EMA target encoders."""

    def __init__(self, encoder_config: dict[str, Any]) -> None:
        super().__init__()
        self.online_encoder = TinyViTEncoder(**encoder_config)
        self.target_encoder = TinyViTEncoder(**encoder_config)
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

    def forward(self, images: torch.Tensor, keep_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        online_tokens = self.online_encoder(images, keep_mask=keep_mask)
        with torch.no_grad():
            target_tokens = self.target_encoder(images)
        return self.predictor(online_tokens), target_tokens.detach()


class FineTuneClassifier(nn.Module):
    """Supervised classifier built on a pretrained encoder."""

    def __init__(self, encoder: TinyViTEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(images)
        return self.head(tokens.mean(dim=1))


def trainable_jepa_parameters(model: IJEPAForImages) -> Iterable[nn.Parameter]:
    """Return only online encoder and predictor parameters."""
    yield from model.online_encoder.parameters()
    yield from model.predictor.parameters()


@torch.no_grad()
def update_ema(model: IJEPAForImages, tau: float) -> None:
    """Update the target encoder from the online encoder."""
    for online_param, target_param in zip(model.online_encoder.parameters(), model.target_encoder.parameters(), strict=True):
        target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)


def tensor_scalar(value: torch.Tensor) -> float:
    """Convert a scalar tensor to a Python float."""
    return float(value.detach().float().cpu().item())


def covariance_diagnostics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute effective rank and off-diagonal covariance magnitude."""
    if values.size(0) < MIN_COVARIANCE_SAMPLES:
        zero = values.new_tensor(0.0)
        return zero, zero

    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (values.size(0) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if total <= 0:
        effective_rank = values.new_tensor(0.0)
    else:
        probabilities = eigenvalues / total
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        effective_rank = entropy.exp()

    offdiag = covariance - torch.diag(torch.diag(covariance))
    offdiag_abs_mean = offdiag.abs().mean()
    return effective_rank, offdiag_abs_mean


def representation_diagnostics(prefix: str, values: torch.Tensor) -> dict[str, float]:
    """Compute representation health metrics for flattened token features."""
    values = values.detach().float()
    feature_std = values.std(dim=0, unbiased=False)
    feature_norm = values.norm(dim=-1)
    effective_rank, cov_offdiag_abs_mean = covariance_diagnostics(values)
    effective_rank_fraction = effective_rank / values.size(1)

    return {
        f"{prefix}/norm_mean": tensor_scalar(feature_norm.mean()),
        f"{prefix}/norm_std": tensor_scalar(feature_norm.std(unbiased=False)),
        f"{prefix}/std_mean": tensor_scalar(feature_std.mean()),
        f"{prefix}/std_min": tensor_scalar(feature_std.min()),
        f"{prefix}/dead_dim_fraction": tensor_scalar((feature_std < DEAD_DIM_STD_THRESHOLD).float().mean()),
        f"{prefix}/effective_rank": tensor_scalar(effective_rank),
        f"{prefix}/effective_rank_fraction": tensor_scalar(effective_rank_fraction),
        f"{prefix}/cov_offdiag_abs_mean": tensor_scalar(cov_offdiag_abs_mean),
    }


def mask_diagnostics(keep_mask: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
    """Compute context, target, and overlap mask diagnostics."""
    total_patches = keep_mask.size(1)
    context_count = keep_mask.sum(dim=1).float().mean()
    target_count = target_mask.sum(dim=1).float().mean()
    overlap_fraction = (keep_mask & target_mask).float().mean()

    return {
        "mask/context_count": tensor_scalar(context_count),
        "mask/target_count": tensor_scalar(target_count),
        "mask/context_ratio": tensor_scalar(context_count / total_patches),
        "mask/target_ratio": tensor_scalar(target_count / total_patches),
        "mask/overlap_fraction": tensor_scalar(overlap_fraction),
    }


@torch.no_grad()
def ema_diagnostics(model: IJEPAForImages) -> dict[str, float]:
    """Compute EMA parameter-distance diagnostics."""
    distance_sq = torch.zeros((), device=next(model.online_encoder.parameters()).device)
    target_sq = torch.zeros_like(distance_sq)

    for online_param, target_param in zip(model.online_encoder.parameters(), model.target_encoder.parameters(), strict=True):
        distance_sq += torch.sum((online_param - target_param) ** 2)
        target_sq += torch.sum(target_param**2)

    param_l2 = distance_sq.sqrt()
    relative_param_l2 = param_l2 / target_sq.sqrt().clamp_min(1e-12)
    return {
        "ema/param_l2": tensor_scalar(param_l2),
        "ema/relative_param_l2": tensor_scalar(relative_param_l2),
    }


def grade_lower_better(value: float, thresholds: list[float]) -> str:
    """Grade a metric where lower values are better."""
    for grade, threshold in zip(ANSI_COLORS, thresholds, strict=True):
        if value <= threshold:
            return grade
    return "dark_red"


def grade_higher_better(value: float, thresholds: list[float]) -> str:
    """Grade a metric where higher values are better."""
    for grade, threshold in zip(ANSI_COLORS, thresholds, strict=True):
        if value >= threshold:
            return grade
    return "dark_red"


def grade_band(value: float) -> str:
    """Grade metrics that should be nonzero but not explosive."""
    if 0.30 <= value <= 2.0:
        return "green"
    if 0.15 <= value < 0.30 or 2.0 < value <= 3.0:
        return "blue"
    if 0.08 <= value < 0.15 or 3.0 < value <= 4.0:
        return "cerulean"
    if 0.04 <= value < 0.08 or 4.0 < value <= 6.0:
        return "yellow"
    if 0.02 <= value < 0.04 or 6.0 < value <= 8.0:
        return "orange"
    if DEAD_DIM_STD_THRESHOLD <= value < 0.02 or 8.0 < value <= 12.0:
        return "light_red"
    return "dark_red"


def metric_grade(key: str, value: float, metrics: dict[str, float]) -> str | None:
    """Choose a CLI color grade for a metric value."""
    if key in {"loss", "pred_target/mse"}:
        return grade_lower_better(value, [0.05, 0.10, 0.20, 0.40, 0.80, 1.50, 2.50])
    if key == "pred_target/mae":
        return grade_lower_better(value, [0.05, 0.10, 0.20, 0.40, 0.70, 1.00, 1.50])
    if key == "grad_norm":
        return grade_lower_better(value, [0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00])
    if key == "pred_target/cosine":
        return grade_higher_better(value, [0.85, 0.65, 0.45, 0.20, 0.00, -0.20, -0.50])
    if key.endswith("/std_mean") or key.endswith("/std_min") or key.endswith("/norm_mean"):
        return grade_band(value)
    if key.endswith("/dead_dim_fraction") or key == "mask/overlap_fraction":
        return grade_lower_better(value, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75])
    if key.endswith("/effective_rank"):
        prefix = key.removesuffix("/effective_rank")
        rank_fraction = metrics.get(f"{prefix}/effective_rank_fraction", value)
        return grade_higher_better(rank_fraction, [0.75, 0.60, 0.45, 0.30, 0.20, 0.10, 0.05])
    if key in {"mask/context_ratio", "mask/target_ratio"}:
        return grade_band(value)
    if key == "ema_tau":
        return grade_higher_better(value, [0.99, 0.97, 0.95, 0.90, 0.80, 0.60, 0.40])
    if key == "ema/relative_param_l2":
        return grade_lower_better(value, [0.01, 0.03, 0.07, 0.15, 0.30, 0.50, 1.00])
    return None


def color_metric_text(key: str, value_text: str, value: float, metrics: dict[str, float], *, color: bool) -> str:
    """Apply ANSI color to a metric value."""
    if not color:
        return value_text

    grade = metric_grade(key, value, metrics)
    if grade is None:
        return value_text

    return f"{ANSI_COLORS[grade]}{value_text}{ANSI_RESET}"


def format_metric_value(key: str, value: float, metrics: dict[str, float], *, color: bool) -> str:
    """Format and color one metric value."""
    if key in {"epoch", "step"}:
        return str(int(value))

    value_text = f"{value:.4f}"
    return color_metric_text(key, value_text, value, metrics, color=color)


def format_live_metric_value(key: str, value: float, metrics: dict[str, float], *, color: bool) -> str:
    """Format one compact metric value for the live two-line display."""
    if abs(value) < 0.0005:
        value_text = "0"
    else:
        precision = 2 if key.endswith("/effective_rank") else 3
        value_text = f"{value:.{precision}f}"
        if value_text.startswith("0."):
            value_text = value_text[1:]
        elif value_text.startswith("-0."):
            value_text = "-" + value_text[2:]

    return color_metric_text(key, value_text, value, metrics, color=color)


def format_cli_metrics(step: int, metrics: dict[str, float], *, color: bool) -> str:
    """Format a full CLI diagnostics row."""
    keys = [
        "epoch",
        "step",
        "loss",
        "lr",
        "grad_norm",
        "pred_target/mse",
        "pred_target/mae",
        "pred_target/cosine",
        "target/std_mean",
        "target/dead_dim_fraction",
        "target/effective_rank",
        "pred/std_mean",
        "pred/effective_rank",
        "mask/overlap_fraction",
        "mask/context_ratio",
        "mask/target_ratio",
        "ema_tau",
        "ema/relative_param_l2",
    ]
    parts = [f"global_step={step}"]
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        parts.append(f"{key}={format_metric_value(key, value, metrics, color=color)}")
    return " | ".join(parts)


def strip_ansi(value: str) -> str:
    """Remove ANSI color codes for display-width estimates."""
    return ANSI_PATTERN.sub("", value)


def visible_len(value: str) -> int:
    """Count visible terminal columns for strings with ANSI colors."""
    return len(strip_ansi(value))


def clip_ansi(value: str, max_visible: int) -> str:
    """Clip ANSI-colored text without leaving a color active."""
    if max_visible <= 0:
        return ""
    if visible_len(value) <= max_visible:
        return value
    if max_visible <= 3:
        return "." * max_visible

    limit = max_visible - 3
    visible = 0
    cursor = 0
    chunks: list[str] = []

    for match in ANSI_PATTERN.finditer(value):
        text = value[cursor : match.start()]
        remaining = limit - visible
        if remaining <= 0:
            break
        chunks.append(text[:remaining])
        visible += min(len(text), remaining)
        if visible >= limit:
            break
        chunks.append(match.group())
        cursor = match.end()
    else:
        text = value[cursor:]
        remaining = limit - visible
        if remaining > 0:
            chunks.append(text[:remaining])

    return "".join(chunks) + ANSI_RESET + "..."


def format_live_metric_group(metrics: dict[str, float], keys: list[tuple[str, str]], *, color: bool) -> str:
    """Format one group of live metrics."""
    parts = []
    for key, alias in keys:
        value = metrics.get(key)
        if value is None:
            continue
        parts.append(f"{alias}={format_live_metric_value(key, value, metrics, color=color)}")
    return ",".join(parts)


def format_live_display(metrics: dict[str, float], *, color: bool) -> tuple[str, str]:
    """Split live metrics across two stable terminal lines."""
    core_keys = [
        ("loss", "loss"),
        ("pred_target/cosine", "cos"),
        ("pred_target/mse", "mse"),
        ("grad_norm", "grad"),
    ]
    diagnostic_keys = [
        ("target/std_mean", "tstd"),
        ("target/effective_rank", "trank"),
        ("pred/std_mean", "pstd"),
        ("pred/effective_rank", "prank"),
        ("ema/relative_param_l2", "ema"),
        ("mask/overlap_fraction", "ovlp"),
        ("mask/context_ratio", "ctx"),
        ("mask/target_ratio", "tgt"),
    ]
    top_line = format_live_metric_group(metrics, diagnostic_keys, color=color)
    bottom_line = format_live_metric_group(metrics, core_keys, color=color)
    return top_line, bottom_line


def format_duration(seconds: float) -> str:
    """Format a duration for the live progress renderer."""
    seconds = max(0.0, seconds)
    minutes, whole_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}"
    return f"{minutes:02d}:{whole_seconds:02d}"


def compact_progress_description(description: str) -> str:
    """Shorten the fixed progress description for narrow terminals."""
    parts = description.split()
    if len(parts) == 4 and parts[0] == "pretrain" and parts[2] == "epoch":
        return f"{parts[1]} e{parts[3]}"
    return description.replace(" epoch ", " e")


def short_progress_description(description: str) -> str:
    """Keep only the epoch marker for very narrow terminals."""
    parts = description.split()
    if len(parts) == 4 and parts[0] == "pretrain" and parts[2] == "epoch":
        return f"e{parts[3]}"
    return compact_progress_description(description)


class LiveCliProgress:
    """Stable two-line terminal renderer for live diagnostics."""

    def __init__(self, *, description: str, total: int) -> None:
        self.description = description
        self.compact_description = compact_progress_description(description)
        self.short_description = short_progress_description(description)
        self.total = total
        self.start_time = time.perf_counter()
        self.lines_rendered = 0
        self.enabled = sys.stderr.isatty()

    def update(self, current: int, *, top_line: str, bottom_line: str) -> None:
        """Render a two-line progress update."""
        if not self.enabled:
            return

        terminal_width = shutil.get_terminal_size(fallback=(120, 20)).columns
        top_line = clip_ansi(top_line, terminal_width)
        progress_line = clip_ansi(
            self.progress_line(current, bottom_line, terminal_width=terminal_width),
            terminal_width,
        )

        if self.lines_rendered:
            sys.stderr.write(f"\x1b[{self.lines_rendered}A")

        sys.stderr.write(f"\r\x1b[2K{top_line}\n")
        sys.stderr.write(f"\r\x1b[2K{progress_line}\n")
        sys.stderr.flush()
        self.lines_rendered = 2

    def close(self) -> None:
        """Finish the renderer without leaving the cursor on a metrics line."""
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def progress_line(self, current: int, bottom_line: str, *, terminal_width: int) -> str:
        """Build the bottom progress line."""
        elapsed = time.perf_counter() - self.start_time
        rate = current / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - current) / rate if rate > 0 else 0.0
        fraction = min(1.0, current / self.total) if self.total else 0.0
        bar_width = 10
        filled = round(bar_width * fraction)
        bar = "█" * filled + " " * (bar_width - filled)
        time_text = f"{format_duration(elapsed)}<{format_duration(remaining)} {rate:.2f}/s"
        detailed_line = (
            f"{self.description}: {fraction:>4.0%}|{bar}| {current}/{self.total} "
            f"[{time_text}] {bottom_line}"
        )
        if visible_len(detailed_line) <= terminal_width:
            return detailed_line

        compact_line = (
            f"{self.compact_description} {fraction:>3.0%} {current}/{self.total} "
            f"{format_duration(elapsed)}<{format_duration(remaining)} {rate:.1f}/s {bottom_line}"
        )
        if visible_len(compact_line) <= terminal_width:
            return compact_line

        return (
            f"{self.short_description} {fraction:>3.0%} {current}/{self.total} "
            f"{format_duration(elapsed)}<{format_duration(remaining)} {rate:.1f}/s {bottom_line}"
        )


def sample_block_masks(
    *,
    batch_size: int,
    grid_size: int,
    context_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample rectangular target masks and complementary context masks."""
    num_patches = grid_size * grid_size
    target_count = max(1, min(num_patches - 1, round(num_patches * (1.0 - context_ratio))))
    block_h = max(1, grid_size // 4)
    block_w = max(1, grid_size // 4)

    target_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

    for batch_index in range(batch_size):
        while int(target_mask[batch_index].sum().item()) < target_count:
            top = torch.randint(0, grid_size - block_h + 1, (), device=device).item()
            left = torch.randint(0, grid_size - block_w + 1, (), device=device).item()
            for row in range(top, top + block_h):
                start = row * grid_size + left
                target_mask[batch_index, start : start + block_w] = True

        extra = int(target_mask[batch_index].sum().item()) - target_count
        if extra > 0:
            indices = torch.nonzero(target_mask[batch_index], as_tuple=False).flatten()
            target_mask[batch_index, indices[:extra]] = False

    keep_mask = ~target_mask
    return keep_mask, target_mask


def default_image_size(dataset: str, requested: int | None) -> int:
    """Choose a dataset-native image size unless the caller overrides it."""
    if requested is not None:
        return requested
    return DATASET_IMAGE_SIZE[dataset]


def build_loader(
    dataset: CIFAR10 | STL10,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """Build a PyTorch dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )


def run_pretrain(args: argparse.Namespace) -> None:
    """Run I-JEPA pretraining."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_name = require_arg(args.dataset, "dataset")
    run_name = require_arg(args.run_name, "run_name")
    image_size = default_image_size(dataset_name, args.image_size)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.dry_run:
        print(
            "pretrain "
            f"dataset={dataset_name} data_dir={data_dir} output_dir={output_dir} "
            f"run_name={run_name} image_size={image_size} patch_size={args.patch_size} "
            f"epochs={args.epochs} batch_size={args.batch_size} device={device}",
        )
        return

    dataset = build_pretrain_dataset(dataset_name, data_dir, image_size)
    loader = build_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
        drop_last=True,
    )

    encoder_config = {
        "image_size": image_size,
        "patch_size": args.patch_size,
        "in_channels": 3,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
    }
    model = IJEPAForImages(encoder_config).to(device)
    trainable_params = list(trainable_jepa_parameters(model))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    grid_size = image_size // args.patch_size
    run_dir = output_dir / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        base_desc = f"pretrain {dataset_name} epoch {epoch + 1}/{args.epochs}"
        live_progress = LiveCliProgress(description=base_desc, total=len(loader)) if args.live_full_cli else None
        progress = loader if args.live_full_cli else tqdm(loader, desc=base_desc, dynamic_ncols=True)
        try:
            for step, batch in enumerate(progress):
                if args.max_train_batches is not None and step >= args.max_train_batches:
                    break

                images = batch[0].to(device, non_blocking=True)
                keep_mask, target_mask = sample_block_masks(
                    batch_size=images.size(0),
                    grid_size=grid_size,
                    context_ratio=args.context_ratio,
                    device=device,
                )

                pred_tokens, target_tokens = model(images, keep_mask)
                pred_target_tokens = pred_tokens[target_mask]
                target_target_tokens = target_tokens[target_mask]
                loss = F.smooth_l1_loss(pred_target_tokens, target_target_tokens)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
                optimizer.step()
                update_ema(model, args.ema_tau)

                with torch.no_grad():
                    cosine = F.cosine_similarity(pred_target_tokens, target_target_tokens, dim=-1).mean()
                    mse = F.mse_loss(pred_target_tokens, target_target_tokens)
                    mae = F.l1_loss(pred_target_tokens, target_target_tokens)
                    metrics = {
                        "epoch": float(epoch + 1),
                        "step": float(step + 1),
                        "loss": tensor_scalar(loss),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "grad_norm": tensor_scalar(grad_norm),
                        "pred_target/mse": tensor_scalar(mse),
                        "pred_target/mae": tensor_scalar(mae),
                        "pred_target/cosine": tensor_scalar(cosine),
                        "ema_tau": float(args.ema_tau),
                    }
                    metrics.update(mask_diagnostics(keep_mask, target_mask))

                    should_print_full = args.cli_log_every_steps > 0 and (
                        global_step == 0 or (global_step + 1) % args.cli_log_every_steps == 0
                    )
                    needs_full_metrics = args.live_full_cli or should_print_full
                    if needs_full_metrics:
                        metrics.update(representation_diagnostics("target", target_target_tokens))
                        metrics.update(representation_diagnostics("pred", pred_target_tokens))
                        metrics.update(ema_diagnostics(model))

                    if should_print_full and not args.live_full_cli:
                        progress.write(format_cli_metrics(global_step + 1, metrics, color=args.color_cli))

                if args.live_full_cli:
                    top_line, bottom_line = format_live_display(metrics, color=args.color_cli)
                    if live_progress is not None:
                        live_progress.update(step + 1, top_line=top_line, bottom_line=bottom_line)
                else:
                    progress.set_postfix(
                        loss=format_metric_value("loss", metrics["loss"], metrics, color=args.color_cli),
                        cos=format_metric_value(
                            "pred_target/cosine",
                            metrics["pred_target/cosine"],
                            metrics,
                            color=args.color_cli,
                        ),
                        mse=format_metric_value("pred_target/mse", metrics["pred_target/mse"], metrics, color=args.color_cli),
                        grad=format_metric_value("grad_norm", metrics["grad_norm"], metrics, color=args.color_cli),
                        mask=format_metric_value(
                            "mask/overlap_fraction",
                            metrics["mask/overlap_fraction"],
                            metrics,
                            color=args.color_cli,
                        ),
                    )
                global_step += 1
        finally:
            if live_progress is not None:
                live_progress.close()

    checkpoint_path = checkpoint_dir / "ijepa_pretrain_last.pt"
    torch.save(
        {
            "dataset": dataset_name,
            "model_config": encoder_config,
            "online_encoder": model.online_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved pretraining checkpoint: {checkpoint_path}")


def evaluate_classifier(
    model: FineTuneClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float]:
    """Evaluate loss and accuracy."""
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break

            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss_total += loss.item() * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0

    return loss_total / total, correct / total


def run_finetune(args: argparse.Namespace) -> None:
    """Run supervised fine-tuning from a pretrained I-JEPA checkpoint."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_name = require_arg(args.dataset, "dataset")
    run_name = require_arg(args.run_name, "run_name")
    source_checkpoint_path = Path(require_arg(args.checkpoint, "checkpoint"))

    if args.dry_run:
        print(
            "finetune "
            f"dataset={dataset_name} data_dir={args.data_dir} output_dir={args.output_dir} "
            f"run_name={run_name} checkpoint={source_checkpoint_path} "
            f"epochs={args.epochs} batch_size={args.batch_size} device={device}",
        )
        return

    checkpoint = load_torch_checkpoint(source_checkpoint_path, device)
    encoder_config = dict(checkpoint["model_config"])
    image_size = int(encoder_config["image_size"])

    train_dataset, val_dataset = build_finetune_datasets(dataset_name, Path(args.data_dir), image_size)
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
        drop_last=False,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
        drop_last=False,
    )

    encoder = TinyViTEncoder(**encoder_config)
    encoder.load_state_dict(checkpoint["online_encoder"])
    model = FineTuneClassifier(encoder, num_classes=10).to(device)

    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_dir = Path(args.output_dir) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"finetune {dataset_name} epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(progress):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            acc = (logits.argmax(dim=1) == labels).float().mean()
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")

        val_loss, val_acc = evaluate_classifier(
            model,
            val_loader,
            device=device,
            max_batches=args.max_val_batches,
        )
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "dataset": dataset_name,
                    "source_checkpoint": str(source_checkpoint_path),
                    "model_config": encoder_config,
                    "model": model.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoint_dir / "finetune_best.pt",
            )

    checkpoint_path = checkpoint_dir / "finetune_last.pt"
    torch.save(
        {
            "dataset": dataset_name,
            "source_checkpoint": str(source_checkpoint_path),
            "model_config": encoder_config,
            "model": model.state_dict(),
            "best_val_acc": best_acc,
        },
        checkpoint_path,
    )
    print(f"Saved fine-tuning checkpoint: {checkpoint_path}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by both commands."""
    parser.add_argument("--config")
    parser.add_argument("--dataset", choices=("cifar10", "stl10"))
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--cli-log-every-steps", type=int, default=20)
    parser.add_argument("--color-cli", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-full-cli", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain", help="Run I-JEPA pretraining.")
    add_common_args(pretrain)
    pretrain.add_argument("--image-size", type=int)
    pretrain.add_argument("--patch-size", type=int, default=4)
    pretrain.add_argument("--embed-dim", type=int, default=128)
    pretrain.add_argument("--depth", type=int, default=4)
    pretrain.add_argument("--num-heads", type=int, default=4)
    pretrain.add_argument("--mlp-ratio", type=float, default=4.0)
    pretrain.add_argument("--context-ratio", type=float, default=0.6)
    pretrain.add_argument("--ema-tau", type=float, default=0.996)
    pretrain.set_defaults(func=run_pretrain)

    finetune = subparsers.add_parser("finetune", help="Fine-tune a pretrained encoder.")
    add_common_args(finetune)
    finetune.add_argument("--checkpoint")
    finetune.add_argument("--max-val-batches", type=int)
    finetune.add_argument("--freeze-encoder", action="store_true")
    finetune.set_defaults(func=run_finetune)

    return parser.parse_args()


def main() -> None:
    """Run the selected command."""
    args = apply_config(parse_args())
    args.func(args)


if __name__ == "__main__":
    main()
