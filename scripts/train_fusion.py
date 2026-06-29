from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision import transforms

from retibrain_fusion.checkpoints import load_matching_state_dict
from retibrain_fusion.config import get_metainfo_columns, load_config, render_template
from retibrain_fusion.datasets import FusionDataset
from retibrain_fusion.engine import build_optimizer, train_fold
from retibrain_fusion.models import MetaAdditiveRegressor, SplitCFPMorphModel, TripleFusionModel
from retibrain_fusion.plotting import setup_matplotlib
from retibrain_fusion.utils import ensure_dir, resolve_device, seed_everything, set_cuda_visible_devices, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CFP + morphology + metadata fusion model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir in the config.")
    parser.add_argument("--device", type=str, default=None, help="Override device in the config, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg.experiment.output_dir = args.output_dir
    if args.device is not None:
        cfg.experiment.device = args.device

    set_cuda_visible_devices(cfg.experiment.cuda_visible_devices)
    seed_everything(cfg.experiment.seed)
    device = resolve_device(cfg.experiment.device)

    output_root = ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name)
    logger = setup_logger(output_root / "train.log")
    logger.info("Using device: %s", device)
    setup_matplotlib(font_family=cfg.plot.font_family, dpi=cfg.plot.dpi)

    meta_cols, meta_parts_dim, parts = get_metainfo_columns(cfg.data.metainfo, cfg.data.metainfo_map)
    existing_parts = parts[:-1] if len(parts) > 1 else []
    new_parts = [part for part in parts if part not in existing_parts]

    df = pd.read_csv(cfg.data.csv_path)
    kfold = KFold(n_splits=cfg.training.n_splits, shuffle=True, random_state=cfg.training.random_state)
    target_folds = set(cfg.training.target_folds) if cfg.training.target_folds is not None else None

    transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
            transforms.ToTensor(),
        ]
    )

    for fold, (train_index, val_index) in enumerate(kfold.split(df)):
        if target_folds is not None and fold not in target_folds:
            logger.info("Skipping fold %d", fold)
            continue

        fold_dir = ensure_dir(output_root / cfg.data.target / cfg.data.metainfo / cfg.data.cfp_type / f"fold_{fold}")
        logger.info("Starting fold %d | target=%s | metainfo=%s | CFP=%s", fold, cfg.data.target, cfg.data.metainfo, cfg.data.cfp_type)

        train_df = df.iloc[train_index].reset_index(drop=True)
        val_df = df.iloc[val_index].reset_index(drop=True)
        train_df.to_csv(fold_dir / "train.csv", index=False)
        val_df.to_csv(fold_dir / "val.csv", index=False)

        train_dataset = FusionDataset(
            train_df,
            target=cfg.data.target,
            meta_cols=meta_cols,
            cfp_type=cfg.data.cfp_type,
            transform=transform,
            path_replacements=cfg.data.path_replacements,
        )
        val_dataset = FusionDataset(
            val_df,
            target=cfg.data.target,
            meta_cols=meta_cols,
            cfp_type=cfg.data.cfp_type,
            transform=transform,
            path_replacements=cfg.data.path_replacements,
            meta_mean=train_dataset.meta_mean,
            meta_std=train_dataset.meta_std,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            drop_last=cfg.training.drop_last,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            drop_last=False,
        )

        cfp_morph_model = SplitCFPMorphModel(morph_dim=train_dataset.morph_dim).to(device)
        meta_model = MetaAdditiveRegressor(
            meta_parts_dim=meta_parts_dim,
            mid_dim=cfg.model.meta_mid_dim,
            out_dim_per_meta=cfg.model.out_dim_per_meta,
            existing_parts=existing_parts,
        ).to(device)
        model = TripleFusionModel(
            cfp_morph_net=cfp_morph_model,
            meta_net=meta_model,
            cfp_weight_init=cfg.model.cfp_weight_init,
            morph_weight_init=cfg.model.morph_weight_init,
            meta_weight_init=cfg.model.meta_weight_init,
        ).to(device)

        template_vars = {
            "data_source": cfg.data.data_source,
            "target": cfg.data.target,
            "metainfo": cfg.data.metainfo,
            "cfp_type": cfg.data.cfp_type,
            "fold": fold,
        }
        cfp_ckpt = render_template(cfg.pretrained.cfp_checkpoint_template, **template_vars)
        meta_ckpt = render_template(cfg.pretrained.meta_checkpoint_template, **template_vars)
        load_matching_state_dict(model.cfp_morph_net, cfp_ckpt, device, state_key=cfg.pretrained.cfp_state_key, logger=logger)
        load_matching_state_dict(model.meta_net, meta_ckpt, device, state_key=cfg.pretrained.meta_state_key, logger=logger)

        optimizer = build_optimizer(
            model=model,
            existing_parts=existing_parts,
            new_parts=new_parts,
            lr_high=cfg.training.lr_high,
            lr_low=cfg.training.lr_low,
            weight_decay=cfg.training.weight_decay,
        )

        train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            output_dir=fold_dir,
            device=device,
            epochs=cfg.training.epochs,
            aux_loss_weight=cfg.training.aux_loss_weight,
            label_transform=cfg.data.label_transform,
            checkpoint_every=cfg.training.checkpoint_every,
            plot_every=cfg.training.plot_every if cfg.plot.enabled else 0,
            save_component_plots=cfg.training.save_component_plots,
            plot_format=cfg.plot.file_format,
            arcsinh_scale=cfg.plot.arcsinh_scale,
            early_stopping_patience=cfg.training.early_stopping_patience,
            early_stopping_min_delta=cfg.training.early_stopping_min_delta,
            logger=logger,
        )


if __name__ == "__main__":
    main()
