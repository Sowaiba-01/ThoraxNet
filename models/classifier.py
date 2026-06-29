"""
Multi-label classification head with MC Dropout.

Architecture:
    BioMedCLIP ViT-B/16 → LayerNorm → Dropout → Linear(512, 256)
                        → GELU → Dropout → Linear(256, 14)

The double-dropout design enables Monte Carlo uncertainty estimation at
inference time by keeping dropout active (model.train() mode).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.backbone import BioMedCLIPVisionBackbone
from data.dataset import NUM_CLASSES


class ChestAIClassifier(nn.Module):
    """
    Full model: backbone + classification head.

    Args:
        backbone_name: HF model ID for BioMedCLIP.
        num_classes: Number of output labels (14 for NIH ChestX-ray14).
        dropout_rate: Dropout probability — used in both training and MC inference.
        freeze_backbone: Start with frozen backbone (set False after warm-up).
    """

    def __init__(
        self,
        backbone_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = BioMedCLIPVisionBackbone(
            model_name=backbone_name,
            pretrained=True,
            freeze=freeze_backbone,
        )
        embed_dim = self.backbone.embed_dim

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

        self._dropout_rate = dropout_rate
        self._init_head_weights()

    def _init_head_weights(self) -> None:
        """Xavier uniform init for classification head linear layers."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (logits, not probabilities).

        Args:
            x: (B, 3, H, W) image tensor.

        Returns:
            (B, num_classes) raw logits. Apply sigmoid for probabilities.
        """
        features = self.backbone(x)   # (B, embed_dim)
        return self.head(features)    # (B, num_classes)

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze API (called by trainer during warm-up)
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def unfreeze_backbone(self, last_n_blocks: int | None = None) -> None:
        self.backbone.unfreeze(last_n_blocks)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def summary(self) -> str:
        p = self.num_parameters
        return (
            f"ChestAIClassifier | "
            f"total={p['total']:,} params | "
            f"trainable={p['trainable']:,} params"
        )
