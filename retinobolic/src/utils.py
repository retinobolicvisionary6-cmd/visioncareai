"""
Utility functions for VISIONARY6 — VINAYAK Module.

Covers: reproducibility, experiment ID generation, device info printing,
metric formatting, JSON helpers, and model size calculation.
"""
import os
import json
import random
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Returns a consistently-formatted module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Sets all random seeds for full reproducibility across Python, NumPy, and PyTorch.
    Must be called before any data loading or model initialization.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    _log.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Device Info
# ---------------------------------------------------------------------------
def print_device_info(device: str) -> None:
    """
    Prints available hardware information:
    - device name
    - GPU name if CUDA
    - VRAM if CUDA
    """
    print(f"{'='*50}")
    print(f"  DEVICE     : {device.upper()}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"  GPU NAME   : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        total_mb = props.total_memory // (1024 ** 2)
        print(f"  VRAM TOTAL : {total_mb} MB")
    else:
        import platform
        print(f"  CPU        : {platform.processor()}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Experiment Tracking
# ---------------------------------------------------------------------------
def generate_experiment_id(model_name: str, seed: int) -> str:
    """
    Generates a unique, timestamped experiment ID.
    Format: YYYYMMDD_HHMMSS_{model}_{seed}
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{model_name}_{seed}"


def create_experiment_dir(experiments_base: Path, experiment_id: str) -> Path:
    """
    Creates a timestamped experiment directory and returns its path.
    """
    exp_dir = experiments_base / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Experiment directory: %s", exp_dir)
    return exp_dir


def save_experiment_config(exp_dir: Path, config: dict[str, Any]) -> None:
    """
    Saves the full experiment configuration as JSON in the experiment directory.
    """
    config_path = exp_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    _log.info("Experiment config saved to: %s", config_path)


# ---------------------------------------------------------------------------
# JSON / File Helpers
# ---------------------------------------------------------------------------
def save_json(data: dict | list, path: Path) -> None:
    """Saves data as a pretty-printed JSON file. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _log.info("JSON saved: %s", path)


def load_json(path: Path) -> dict | list:
    """Loads a JSON file and returns its content."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model Introspection
# ---------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Returns (total_params, trainable_params) for a PyTorch model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Estimates model size in MB by counting parameter bytes.
    """
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return round(total_bytes / (1024 ** 2), 2)


def print_model_summary(model: torch.nn.Module, model_name: str) -> None:
    """Prints a concise model summary."""
    total, trainable = count_parameters(model)
    size_mb = get_model_size_mb(model)
    print(f"\n{'='*50}")
    print(f"  Model          : {model_name}")
    print(f"  Total Params   : {total:,}")
    print(f"  Trainable      : {trainable:,}")
    print(f"  Model Size     : {size_mb} MB")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Metric Formatting
# ---------------------------------------------------------------------------
def format_metrics_table(metrics: dict[str, float]) -> str:
    """Returns a neatly-formatted string table of metric → value pairs."""
    lines = [f"{'Metric':<35} {'Value':>10}"]
    lines.append("-" * 47)
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<33} {v*100:>9.2f}%")
        else:
            lines.append(f"  {k:<33} {str(v):>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image Hash (for duplicate detection)
# ---------------------------------------------------------------------------
def md5_file(path: Path, chunk_size: int = 8192) -> str:
    """
    Computes an MD5 hash of a file for duplicate detection.
    Returns the hex digest string.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
