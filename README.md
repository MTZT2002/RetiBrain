# RetiBrain Fusion Training

This repository provides a clean training pipeline for fusing bilateral color fundus photographs (CFP), retinal morphology features, and clinical metadata for continuous brain biomarker regression.

The refactored code is organized for reproducibility and readability rather than as a single experiment notebook/script.

## Repository structure

```text
retibrain_fusion_github/
├── configs/
│   └── example_kailuan_fusion.yaml   # editable experiment configuration
├── retibrain_fusion/
│   ├── config.py                     # YAML config parsing and metadata-part selection
│   ├── datasets.py                   # CFP + morphology + metadata dataset
│   ├── models.py                     # CFP, morphology, metadata and fusion models
│   ├── engine.py                     # train/evaluate loops
│   ├── checkpoints.py                # checkpoint loading/saving helpers
│   ├── metrics.py                    # regression metrics
│   ├── plotting.py                   # scatter and Bland-Altman diagnostics
│   └── utils.py                      # seed, logger, device, early stopping
├── scripts/
│   └── train_fusion.py               # main training entry
├── train.py                          # short wrapper entry
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
pip install -r requirements.txt
python train.py --config configs/example_kailuan_fusion.yaml
```

You can also override the output directory or device:

```bash
python train.py --config configs/example_kailuan_fusion.yaml \
  --output-dir ./outputs/debug \
  --device cuda:0
```

## Expected CSV columns

The dataset expects one row per subject/sample and requires:

- CFP path columns:
  - `OP_L_macular`, `OP_R_macular` for `cfp_type: macular`
  - `OPT_L_optic_disc`, `OPT_R_optic_disc` for `cfp_type: optic_disc`
- morphology feature columns with matching left/right prefixes:
  - `OP_L_macular_*`, `OP_R_macular_*`, or
  - `OPT_L_optic_disc_*`, `OPT_R_optic_disc_*`
- target column, for example `WMH_log1p`
- metadata columns listed in `data.metainfo_map`

DICOM (`.dcm`/`.dicom`) and common image formats are supported.

## Notes on preprocessing

Metadata normalization is fitted on the training split and reused for validation. This avoids validation-set distribution leakage and makes fold-level evaluation easier to reproduce.

If the target is stored as `log1p(value)`, keep `label_transform: log1p` so validation metrics and diagnostic plots are reported on the original scale.

## Checkpoint reuse

Pretrained CFP and metadata checkpoints are optional. Configure them in the `pretrained` section of the YAML file. The loader only imports parameters whose names and shapes match the current model, which allows new regression/fusion heads to remain randomly initialized.

## Outputs

For each fold, the training script saves:

- `train.csv` and `val.csv`
- `epoch_metrics.csv`
- `fusion_best_loss.pth`, `fusion_best_pearson.pth`
- `meta_best_loss.pth`, `meta_best_pearson.pth`
- validation prediction CSVs for the best checkpoints
- optional SVG diagnostic plots under `vis/`

## Major cleanup from the original single-file script

- Removed hard-coded GPU assignment and private experiment path variants.
- Moved experiment settings into YAML.
- Split dataset, model, metrics, plotting and training logic into reusable modules.
- Removed commented legacy plotting blocks and duplicate plotting functions.
- Parameterized morphology feature dimension instead of hard-coding 144.
- Replaced deprecated `pretrained=False` usage with `weights=None` when available.
- Avoided validation metadata normalization leakage by applying training-split mean/std to validation.
