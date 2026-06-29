from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr


def inverse_label_transform(values: np.ndarray, transform: str = "log1p") -> np.ndarray:
    """Map model-space labels/predictions back to the original scale."""
    values = np.asarray(values).reshape(-1)
    if transform == "log1p":
        return np.expm1(values)
    if transform in {"none", "identity", None}:
        return values
    raise ValueError(f"Unsupported label_transform: {transform}")


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Pearson correlation with robust handling of empty or constant arrays."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2 or len(y_pred) < 2:
        return float("nan"), float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(y_true, y_pred)
    return float(r), float(p)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_transform: str = "log1p",
) -> Dict[str, float]:
    """Compute regression metrics on the original target scale."""
    y_true = inverse_label_transform(y_true, label_transform)
    y_pred = inverse_label_transform(y_pred, label_transform)
    pearson_r, pearson_p = safe_pearsonr(y_true, y_pred)
    error = y_pred - y_true
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }
