from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn


def extract_state_dict(checkpoint, state_key: Optional[str] = None):
    """Extract a model state dict from a checkpoint object."""
    if state_key and isinstance(checkpoint, dict) and state_key in checkpoint:
        return checkpoint[state_key]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_matching_state_dict(
    model: nn.Module,
    checkpoint_path: str | Path | None,
    device: torch.device,
    state_key: Optional[str] = None,
    logger: logging.Logger | None = None,
) -> int:
    """Load only parameters whose names and shapes match the target model.

    This is useful when reusing an encoder from a larger checkpoint while keeping
    newly defined regression heads randomly initialized.
    """
    if checkpoint_path is None:
        return 0

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        if logger:
            logger.warning("Checkpoint not found: %s", checkpoint_path)
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = extract_state_dict(checkpoint, state_key=state_key)
    target_state = model.state_dict()
    matched_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    model.load_state_dict(matched_state, strict=False)
    if logger:
        logger.info("Loaded %d matching parameters from %s", len(matched_state), checkpoint_path)
    return len(matched_state)


def save_checkpoint(state_dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
