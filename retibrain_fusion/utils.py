from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve device string into a torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_cuda_visible_devices(cuda_visible_devices: Optional[str]) -> None:
    """Optionally set CUDA_VISIBLE_DEVICES before CUDA is initialized."""
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)


def setup_logger(log_file: str | Path | None = None, name: str = "retibrain") -> logging.Logger:
    """Create a simple console/file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class EarlyStopping:
    """Stop training when validation loss does not improve."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        improved = val_loss <= self.best_loss - self.min_delta
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        self.early_stop = self.counter >= self.patience
        return self.early_stop
