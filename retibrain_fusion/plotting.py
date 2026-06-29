from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import inverse_label_transform, safe_pearsonr


def setup_matplotlib(font_family: str = "Arial", dpi: int = 300) -> None:
    """Set publication-friendly matplotlib defaults while keeping SVG text editable."""
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "figure.dpi": dpi,
            "svg.fonttype": "none",
            "text.usetex": False,
        }
    )


def arcsinh_transform(values: np.ndarray, scale: float = 5.0) -> np.ndarray:
    return np.arcsinh(np.asarray(values) / scale)


def _raw_ticks(min_value: float, max_value: float, step: int = 10) -> np.ndarray:
    lower = np.floor(min_value / step) * step
    upper = np.ceil(max_value / step) * step
    return np.arange(lower, upper + step, step)


def save_prediction_table(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": np.asarray(y_true).reshape(-1), "prediction": np.asarray(y_pred).reshape(-1)}).to_csv(
        output_path, index=False
    )


def save_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_prefix: str | Path,
    file_format: str = "svg",
    arcsinh_scale: float = 5.0,
) -> Dict[str, float]:
    """Save scatter and Bland-Altman plots on an arcsinh display scale.

    Inputs should already be on the target scale that you want to show in the figures.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    pearson_r, pearson_p = safe_pearsonr(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    save_prediction_table(y_true, y_pred, f"{output_prefix}_predictions.csv")
    _save_scatter_plot(y_true, y_pred, title, output_prefix, pearson_r, pearson_p, rmse, file_format, arcsinh_scale)
    _save_bland_altman_plot(y_true, y_pred, output_prefix, file_format, arcsinh_scale)
    return {"pearson_r": pearson_r, "pearson_p": pearson_p, "rmse": rmse}


def _save_scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_prefix: Path,
    pearson_r: float,
    pearson_p: float,
    rmse: float,
    file_format: str,
    arcsinh_scale: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    x = arcsinh_transform(y_true, arcsinh_scale)
    y = arcsinh_transform(y_pred, arcsinh_scale)
    ax.scatter(x, y, alpha=0.7, s=10)

    min_value = min(float(y_true.min()), float(y_pred.min()))
    max_value = max(float(y_true.max()), float(y_pred.max()))
    line = np.linspace(min_value, max_value, 200)
    line_display = arcsinh_transform(line, arcsinh_scale)
    ax.plot(line_display, line_display, "--", linewidth=1, label="y = x")

    ticks = _raw_ticks(0, max_value, step=10)
    tick_positions = arcsinh_transform(ticks, arcsinh_scale)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])

    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{title}\nPearson={pearson_r:.3f} (p={pearson_p:.1e}) | RMSE={rmse:.3f}")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_scatter.{file_format}", bbox_inches="tight")
    plt.close(fig)


def _save_bland_altman_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_prefix: Path,
    file_format: str,
    arcsinh_scale: float,
) -> None:
    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(arcsinh_transform(mean, arcsinh_scale), arcsinh_transform(diff, arcsinh_scale), alpha=0.7, s=10)
    ax.axhline(arcsinh_transform(mean_diff, arcsinh_scale), linestyle="--", linewidth=1, label="Mean diff")
    ax.axhline(arcsinh_transform(mean_diff + 1.96 * sd_diff, arcsinh_scale), linestyle="--", linewidth=1, label="+1.96 SD")
    ax.axhline(arcsinh_transform(mean_diff - 1.96 * sd_diff, arcsinh_scale), linestyle="--", linewidth=1, label="-1.96 SD")

    mean_ticks = _raw_ticks(0, float(mean.max()), step=10)
    diff_abs_max = float(np.max(np.abs(diff)))
    diff_ticks = _raw_ticks(-diff_abs_max, diff_abs_max, step=10)
    ax.set_xticks(arcsinh_transform(mean_ticks, arcsinh_scale))
    ax.set_yticks(arcsinh_transform(diff_ticks, arcsinh_scale))
    ax.set_xticklabels([f"{t:g}" for t in mean_ticks])
    ax.set_yticklabels([f"{t:g}" for t in diff_ticks])

    ax.set_xlabel("Mean of prediction and ground truth")
    ax.set_ylabel("Difference (prediction − ground truth)")
    ax.set_title(f"Bland–Altman Plot\nMean Diff={mean_diff:.3f}, SD={sd_diff:.3f}")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_bland_altman.{file_format}", bbox_inches="tight")
    plt.close(fig)


def save_component_diagnostics(
    buffers: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: str | Path,
    epoch: int,
    label_transform: str = "log1p",
    file_format: str = "svg",
    arcsinh_scale: float = 5.0,
) -> None:
    """Save diagnostics for final, CFP, morphology and metadata predictions."""
    output_dir = Path(output_dir)
    components = ("final", "cfp", "morph", "meta")
    for split, split_buffers in buffers.items():
        for component in components:
            pred_key = f"{component}_pred"
            label_key = f"{component}_label"
            if pred_key not in split_buffers or label_key not in split_buffers:
                continue
            y_pred = inverse_label_transform(split_buffers[pred_key], label_transform)
            y_true = inverse_label_transform(split_buffers[label_key], label_transform)
            prefix = output_dir / f"{split}_epoch{epoch:03d}_{component}_orig"
            save_regression_diagnostics(
                y_true=y_true,
                y_pred=y_pred,
                title=f"{component.upper()} {split} Epoch {epoch}",
                output_prefix=prefix,
                file_format=file_format,
                arcsinh_scale=arcsinh_scale,
            )
