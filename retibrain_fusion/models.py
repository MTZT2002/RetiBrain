from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import nn
import torchvision.models as tv_models


class ChannelAttention(nn.Module):
    """A lightweight channel-attention block used after the ResNet feature map."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.avg_pool(x).view(batch, channels)
        weights = self.sigmoid(self.mlp(weights)).view(batch, channels, 1, 1)
        return x * weights


class ResNet34AttentionBackbone(nn.Module):
    """ResNet-34 encoder followed by channel attention and a projection layer."""

    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        try:
            base = tv_models.resnet34(weights=None)
        except TypeError:  # torchvision < 0.13
            base = tv_models.resnet34(pretrained=False)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.attention = ChannelAttention(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.attention(features)
        features = self.pool(features).flatten(1)
        return self.proj(features)


class SplitCFPMorphModel(nn.Module):
    """Two-branch CFP/morphology model with separate regression heads.

    The image branch encodes left and right CFPs using two ResNet-34 encoders and
    fuses left, right and absolute-difference embeddings. The morphology branch
    consumes pre-extracted retinal features.
    """

    def __init__(self, morph_dim: int = 144, image_embedding_dim: int = 512) -> None:
        super().__init__()
        self.left_eye_net = ResNet34AttentionBackbone(output_dim=image_embedding_dim)
        self.right_eye_net = ResNet34AttentionBackbone(output_dim=image_embedding_dim)

        fusion_in_dim = image_embedding_dim * 3
        self.image_fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        self.morph_mlp = nn.Sequential(
            nn.BatchNorm1d(morph_dim),
            nn.Linear(morph_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.cfp_head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        self.morph_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward_image_branch(self, image: torch.Tensor) -> torch.Tensor:
        left_features = self.left_eye_net(image[:, :3])
        right_features = self.right_eye_net(image[:, 3:])
        diff_features = torch.abs(left_features - right_features)
        fused = self.image_fusion(torch.cat([left_features, right_features, diff_features], dim=1))
        return self.cfp_head(fused).squeeze(1)

    def forward_morph_branch(self, morph: torch.Tensor) -> torch.Tensor:
        features = self.morph_mlp(morph)
        return self.morph_head(features).squeeze(1)


class MetaAdditiveRegressor(nn.Module):
    """Additive metadata regressor with one encoder/head per metadata group."""

    def __init__(
        self,
        meta_parts_dim: Mapping[str, int],
        mid_dim: int = 128,
        out_dim_per_meta: int = 16,
        existing_parts: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.meta_parts_dim = dict(meta_parts_dim)
        existing_set = set(existing_parts or [])

        self.encoders = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.alphas = nn.ParameterDict()

        for name, dim in self.meta_parts_dim.items():
            self.encoders[name] = nn.Sequential(
                nn.Linear(dim, mid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mid_dim, mid_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(mid_dim // 2, out_dim_per_meta),
            )
            self.heads[name] = nn.Sequential(nn.Linear(out_dim_per_meta, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
            init_alpha = 0.9 if name in existing_set else 0.1
            self.alphas[name] = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        start = 0
        total: torch.Tensor | float = 0.0
        for name, dim in self.meta_parts_dim.items():
            part = meta[:, start : start + dim]
            encoded = self.encoders[name](part)
            pred = self.heads[name](encoded).squeeze(1)
            total = total + self.alphas[name] * pred
            start += dim
        return total


class TripleFusionModel(nn.Module):
    """Late-fusion model combining CFP, morphology and metadata predictions."""

    def __init__(
        self,
        cfp_morph_net: SplitCFPMorphModel,
        meta_net: MetaAdditiveRegressor,
        cfp_weight_init: float = 0.2,
        morph_weight_init: float = 0.0,
        meta_weight_init: float = 0.7,
    ) -> None:
        super().__init__()
        self.cfp_morph_net = cfp_morph_net
        self.meta_net = meta_net
        self.w_cfp = nn.Parameter(torch.tensor(cfp_weight_init, dtype=torch.float32))
        self.w_morph = nn.Parameter(torch.tensor(morph_weight_init, dtype=torch.float32))
        self.w_meta = nn.Parameter(torch.tensor(meta_weight_init, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, image: torch.Tensor, morph: torch.Tensor, meta: torch.Tensor):
        pred_cfp = self.cfp_morph_net.forward_image_branch(image)
        pred_morph = self.cfp_morph_net.forward_morph_branch(morph)
        pred_meta = self.meta_net(meta)
        final = self.w_cfp * pred_cfp + self.w_morph * pred_morph + self.w_meta * pred_meta + self.bias
        return final, pred_cfp, pred_morph, pred_meta
