from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


CFP_COLUMN_MAP = {
    "macular": ("OP_L_macular", "OP_R_macular", "OP_L_macular_", "OP_R_macular_"),
    "optic_disc": ("OPT_L_optic_disc", "OPT_R_optic_disc", "OPT_L_optic_disc_", "OPT_R_optic_disc_"),
}


class FusionDataset(Dataset):
    """Dataset for paired CFP images, retinal morphological features, metadata and one regression target.

    The dataset expects one row per subject. For each row it reads left/right CFP paths,
    concatenates the two transformed images along the channel dimension, concatenates
    left/right morphological features, and returns normalized metadata.
    """

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        target: str,
        meta_cols: Sequence[str],
        cfp_type: str = "optic_disc",
        transform=None,
        path_replacements: Mapping[str, str] | None = None,
        meta_mean: np.ndarray | None = None,
        meta_std: np.ndarray | None = None,
    ) -> None:
        if isinstance(data, (str, Path)):
            self.df = pd.read_csv(data)
        else:
            self.df = data.reset_index(drop=True).copy()

        if cfp_type not in CFP_COLUMN_MAP:
            raise ValueError(f"cfp_type must be one of {list(CFP_COLUMN_MAP)}, got {cfp_type!r}")

        self.target = target
        self.meta_cols = list(meta_cols)
        self.cfp_type = cfp_type
        self.transform = transform
        self.path_replacements = dict(path_replacements or {})

        self.left_path_col, self.right_path_col, left_prefix, right_prefix = CFP_COLUMN_MAP[cfp_type]
        self.left_feature_cols = [c for c in self.df.columns if c.startswith(left_prefix)]
        self.right_feature_cols = [c for c in self.df.columns if c.startswith(right_prefix)]

        self._validate_columns()
        self.meta_values, self.meta_mean, self.meta_std = self._prepare_metadata(meta_mean, meta_std)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def morph_dim(self) -> int:
        return len(self.left_feature_cols) + len(self.right_feature_cols)

    def _validate_columns(self) -> None:
        required = [self.left_path_col, self.right_path_col, self.target, *self.meta_cols]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise KeyError(f"Missing required column(s): {missing}")
        if len(self.left_feature_cols) == 0 or len(self.right_feature_cols) == 0:
            raise KeyError(
                "No morphology feature columns were found. "
                f"For cfp_type={self.cfp_type!r}, expected prefixes from {CFP_COLUMN_MAP[self.cfp_type][2:]}"
            )
        if len(self.left_feature_cols) != len(self.right_feature_cols):
            raise ValueError(
                f"Left/right morphology feature counts differ: "
                f"{len(self.left_feature_cols)} vs {len(self.right_feature_cols)}"
            )

    def _prepare_metadata(
        self,
        meta_mean: np.ndarray | None,
        meta_std: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self.df[self.meta_cols].values.astype(np.float32)
        if meta_mean is None:
            meta_mean = values.mean(axis=0, keepdims=True)
        if meta_std is None:
            meta_std = values.std(axis=0, keepdims=True) + 1e-6
        return (values - meta_mean) / meta_std, meta_mean, meta_std

    def _replace_path(self, path: str) -> str:
        path = str(path)
        for old, new in self.path_replacements.items():
            path = path.replace(old, new)
        return path

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """Load a CFP from DICOM or common image formats and convert to RGB."""
        suffix = Path(path).suffix.lower()
        if suffix in {".dcm", ".dicom"}:
            import pydicom

            array = pydicom.dcmread(path).pixel_array
            return Image.fromarray(array).convert("RGB")
        return Image.open(path).convert("RGB")

    def _load_and_transform(self, path: str) -> torch.Tensor:
        image = self._load_image(self._replace_path(path))
        if self.transform is not None:
            return self.transform(image)
        array = np.asarray(image).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array).float()

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        left_img = self._load_and_transform(row[self.left_path_col])
        right_img = self._load_and_transform(row[self.right_path_col])
        image = torch.cat([left_img, right_img], dim=0)

        left_morph = row[self.left_feature_cols].values.astype(np.float32)
        right_morph = row[self.right_feature_cols].values.astype(np.float32)
        morph = torch.from_numpy(np.concatenate([left_morph, right_morph])).float()

        meta = torch.tensor(self.meta_values[index], dtype=torch.float32)
        label = torch.tensor(row[self.target], dtype=torch.float32)
        return image, morph, meta, label
