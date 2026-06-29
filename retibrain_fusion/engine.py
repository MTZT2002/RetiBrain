from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import regression_metrics
from .plotting import save_component_diagnostics
from .utils import EarlyStopping, ensure_dir

COMPONENTS = ("final", "cfp", "morph", "meta")


def _init_buffers() -> Dict[str, list[torch.Tensor]]:
    buffers: Dict[str, list[torch.Tensor]] = {}
    for component in COMPONENTS:
        buffers[f"{component}_pred"] = []
        buffers[f"{component}_label"] = []
    return buffers


def _append_outputs(buffers: Dict[str, list[torch.Tensor]], outputs, labels: torch.Tensor) -> None:
    final, pred_cfp, pred_morph, pred_meta = outputs
    for component, pred in zip(COMPONENTS, (final, pred_cfp, pred_morph, pred_meta)):
        buffers[f"{component}_pred"].append(pred.detach().cpu())
        buffers[f"{component}_label"].append(labels.detach().cpu())


def _buffers_to_numpy(buffers: Mapping[str, list[torch.Tensor]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, values in buffers.items():
        if len(values) == 0:
            out[key] = np.array([])
        else:
            out[key] = torch.cat(values).numpy().reshape(-1)
    return out


def fusion_loss(outputs, labels: torch.Tensor, criterion: nn.Module, aux_loss_weight: float = 0.2) -> torch.Tensor:
    final, pred_cfp, pred_morph, pred_meta = outputs
    return (
        criterion(final, labels)
        + aux_loss_weight * criterion(pred_cfp, labels)
        + aux_loss_weight * criterion(pred_morph, labels)
        + aux_loss_weight * criterion(pred_meta, labels)
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    aux_loss_weight: float,
) -> tuple[float, Dict[str, np.ndarray]]:
    model.train()
    buffers = _init_buffers()
    total_loss = 0.0

    for image, morph, meta, labels in loader:
        image = image.to(device).float()
        morph = morph.to(device).float()
        meta = meta.to(device).float()
        labels = labels.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(image, morph, meta)
        loss = fusion_loss(outputs, labels, criterion, aux_loss_weight)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        _append_outputs(buffers, outputs, labels)

    return total_loss / max(len(loader), 1), _buffers_to_numpy(buffers)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, Dict[str, np.ndarray]]:
    model.eval()
    buffers = _init_buffers()
    total_loss = 0.0

    for image, morph, meta, labels in loader:
        image = image.to(device).float()
        morph = morph.to(device).float()
        meta = meta.to(device).float()
        labels = labels.to(device).float()

        outputs = model(image, morph, meta)
        total_loss += float(criterion(outputs[0], labels).item())
        _append_outputs(buffers, outputs, labels)

    return total_loss / max(len(loader), 1), _buffers_to_numpy(buffers)


def build_optimizer(
    model,
    existing_parts: Sequence[str] | None,
    new_parts: Sequence[str],
    lr_high: float,
    lr_low: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Use a higher LR for new fusion/head parameters and a lower LR for pretrained encoders."""
    high_lr_params = []
    high_lr_params += list(model.cfp_morph_net.cfp_head.parameters())
    high_lr_params += list(model.cfp_morph_net.morph_head.parameters())
    high_lr_params += [model.w_cfp, model.w_morph, model.w_meta, model.bias]

    for name in new_parts:
        if name in model.meta_net.encoders:
            high_lr_params += list(model.meta_net.encoders[name].parameters())
        if name in model.meta_net.heads:
            high_lr_params += list(model.meta_net.heads[name].parameters())
        if name in model.meta_net.alphas:
            high_lr_params.append(model.meta_net.alphas[name])

    low_lr_params = []
    low_lr_params += list(model.cfp_morph_net.left_eye_net.parameters())
    low_lr_params += list(model.cfp_morph_net.right_eye_net.parameters())
    low_lr_params += list(model.cfp_morph_net.image_fusion.parameters())
    low_lr_params += list(model.cfp_morph_net.morph_mlp.parameters())

    for name in existing_parts or []:
        if name in model.meta_net.encoders:
            low_lr_params += list(model.meta_net.encoders[name].parameters())
        if name in model.meta_net.heads:
            low_lr_params += list(model.meta_net.heads[name].parameters())

    param_groups = []
    if high_lr_params:
        param_groups.append({"params": high_lr_params, "lr": lr_high})
    if low_lr_params:
        param_groups.append({"params": low_lr_params, "lr": lr_low})
    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def save_epoch_predictions(buffers: Mapping[str, np.ndarray], output_path: str | Path) -> None:
    data = {}
    for key, value in buffers.items():
        data[key] = np.asarray(value).reshape(-1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(output_path, index=False)


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    output_dir: str | Path,
    device: torch.device,
    epochs: int = 100,
    aux_loss_weight: float = 0.2,
    label_transform: str = "log1p",
    checkpoint_every: int = 2,
    plot_every: int = 0,
    save_component_plots: bool = True,
    plot_format: str = "svg",
    arcsinh_scale: float = 5.0,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    vis_dir = ensure_dir(output_dir / "vis")
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

    best_val_loss = float("inf")
    best_loss_epoch = -1
    best_pearson = -float("inf")
    best_pearson_epoch = -1
    rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_buffers = train_one_epoch(model, train_loader, optimizer, criterion, device, aux_loss_weight)
        val_loss, val_buffers = evaluate(model, val_loader, criterion, device)
        val_metrics = regression_metrics(
            y_true=val_buffers["final_label"],
            y_pred=val_buffers["final_pred"],
            label_transform=label_transform,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pearson_r_origin": val_metrics["pearson_r"],
            "val_pearson_p_origin": val_metrics["pearson_p"],
            "val_rmse_origin": val_metrics["rmse"],
            "val_mae_origin": val_metrics["mae"],
            "w_cfp": float(model.w_cfp.detach().cpu()),
            "w_morph": float(model.w_morph.detach().cpu()),
            "w_meta": float(model.w_meta.detach().cpu()),
        }
        rows.append(row)

        if logger:
            logger.info(
                "Epoch %03d | train %.4f | val %.4f | Pearson %.4f | weights %.3f/%.3f/%.3f",
                epoch,
                train_loss,
                val_loss,
                val_metrics["pearson_r"],
                row["w_cfp"],
                row["w_morph"],
                row["w_meta"],
            )

        if plot_every > 0 and save_component_plots and epoch % plot_every == 0:
            save_component_diagnostics(
                {"train": train_buffers, "val": val_buffers},
                output_dir=vis_dir,
                epoch=epoch,
                label_transform=label_transform,
                file_format=plot_format,
                arcsinh_scale=arcsinh_scale,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            torch.save(model.state_dict(), output_dir / "fusion_best_loss.pth")
            save_epoch_predictions(val_buffers, output_dir / "val_predictions_best_loss.csv")

        if val_metrics["pearson_r"] > best_pearson:
            best_pearson = val_metrics["pearson_r"]
            best_pearson_epoch = epoch
            torch.save(model.state_dict(), output_dir / "fusion_best_pearson.pth")
            save_epoch_predictions(val_buffers, output_dir / "val_predictions_best_pearson.csv")

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            torch.save(model.state_dict(), output_dir / f"fusion_epoch_{epoch:03d}.pth")

        if early_stopping.step(val_loss):
            if logger:
                logger.info("Early stopping at epoch %d", epoch)
            break

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "epoch_metrics.csv", index=False)
    if logger:
        logger.info("Best val loss %.4f at epoch %d", best_val_loss, best_loss_epoch)
        logger.info("Best Pearson %.4f at epoch %d", best_pearson, best_pearson_epoch)
    return metrics_df
