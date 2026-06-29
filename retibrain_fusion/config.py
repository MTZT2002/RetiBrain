from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml


@dataclass
class ExperimentConfig:
    name: str = "fusion_experiment"
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "auto"
    cuda_visible_devices: Optional[str] = None


@dataclass
class DataConfig:
    csv_path: str = ""
    data_source: str = "Kailuan"
    target: str = "WMH_log1p"
    metainfo: str = "part1+part2+part4"
    cfp_type: str = "optic_disc"
    image_size: Tuple[int, int] = (512, 512)
    label_transform: str = "log1p"
    path_replacements: Dict[str, str] = field(default_factory=dict)
    metainfo_map: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 100
    n_splits: int = 5
    target_folds: Optional[List[int]] = None
    random_state: int = 42
    drop_last: bool = True
    aux_loss_weight: float = 0.2
    lr_high: float = 1e-3
    lr_low: float = 1e-6
    weight_decay: float = 0.0
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    checkpoint_every: int = 2
    plot_every: int = 0
    save_component_plots: bool = True


@dataclass
class ModelConfig:
    meta_mid_dim: int = 128
    out_dim_per_meta: int = 16
    cfp_weight_init: float = 0.2
    morph_weight_init: float = 0.0
    meta_weight_init: float = 0.7


@dataclass
class PretrainedConfig:
    cfp_checkpoint_template: Optional[str] = None
    cfp_state_key: str = "cfp_model"
    meta_checkpoint_template: Optional[str] = None
    meta_state_key: str = "model_state_dict"


@dataclass
class PlotConfig:
    enabled: bool = True
    file_format: str = "svg"
    dpi: int = 300
    font_family: str = "Arial"
    arcsinh_scale: float = 5.0


@dataclass
class FusionConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrained: PretrainedConfig = field(default_factory=PretrainedConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


def _as_tuple(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Invalid image_size: {value}")


def _filter_kwargs(cls: type, values: Mapping[str, Any]) -> Dict[str, Any]:
    valid = set(cls.__dataclass_fields__.keys())
    return {k: v for k, v in values.items() if k in valid}


def load_config(path: str | Path) -> FusionConfig:
    """Load a YAML configuration file into dataclasses."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    exp = ExperimentConfig(**_filter_kwargs(ExperimentConfig, raw.get("experiment", {})))

    data_raw = raw.get("data", {})
    data_raw = dict(data_raw)
    data_raw["image_size"] = _as_tuple(data_raw.get("image_size"), DataConfig.image_size)
    data = DataConfig(**_filter_kwargs(DataConfig, data_raw))

    training = TrainingConfig(**_filter_kwargs(TrainingConfig, raw.get("training", {})))
    model = ModelConfig(**_filter_kwargs(ModelConfig, raw.get("model", {})))
    pretrained = PretrainedConfig(**_filter_kwargs(PretrainedConfig, raw.get("pretrained", {})))
    plot = PlotConfig(**_filter_kwargs(PlotConfig, raw.get("plot", {})))

    return FusionConfig(
        experiment=exp,
        data=data,
        training=training,
        model=model,
        pretrained=pretrained,
        plot=plot,
    )


def get_metainfo_columns(metainfo: str, metainfo_map: Mapping[str, List[str]]) -> tuple[list[str], dict[str, int], list[str]]:
    """Return flattened metadata columns and per-part dimensions."""
    parts = [p.strip() for p in metainfo.split("+") if p.strip()]
    columns: list[str] = []
    dims: dict[str, int] = {}

    missing = [p for p in parts if p not in metainfo_map]
    if missing:
        raise KeyError(f"Unknown metainfo part(s): {missing}. Available: {list(metainfo_map)}")

    for part in parts:
        part_cols = list(metainfo_map[part])
        columns.extend(part_cols)
        dims[part] = len(part_cols)
    return columns, dims, parts


def render_template(template: str | None, **kwargs: Any) -> Optional[str]:
    """Render a checkpoint path template with experiment variables."""
    if not template:
        return None
    return template.format(**kwargs)
