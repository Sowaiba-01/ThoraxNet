"""
Vision backbone: BioMedCLIP ViT-B/16 fine-tuned on medical literature.

Microsoft's BiomedCLIP (https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
is pre-trained on 15M biomedical image-text pairs from PubMed — giving it domain
knowledge that vanilla ImageNet ViTs lack.

We extract only the vision encoder and expose its patch embeddings for:
  1. The classification head (see classifier.py)
  2. GradCAM attention rollout (see explainability/gradcam.py)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained


class BioMedCLIPVisionBackbone(nn.Module):
    """
    Vision encoder extracted from BioMedCLIP.

    The model is loaded once and cached; subsequent instantiations reuse weights.

    Args:
        model_name: HuggingFace model identifier.
        pretrained: Load pretrained weights (always True in production).
        freeze: If True, freeze all backbone parameters (for warm-up phase).
    """

    _instance: "BioMedCLIPVisionBackbone | None" = None

    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        # open_clip handles BioMedCLIP weights natively via HF Hub.
        clip_model, _ = create_model_from_pretrained(
            f"hf-hub:{model_name}" if pretrained else "ViT-B-16",
        )
        # Extract only the visual trunk — discard text encoder.
        self.visual = clip_model.visual
        self.embed_dim: int = self.visual.output_dim  # 512 for ViT-B/16

        if freeze:
            self.freeze()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized image tensor.

        Returns:
            (B, embed_dim) CLS token embedding.
        """
        return self.visual(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return intermediate patch tokens for GradCAM.

        Returns:
            (B, num_patches+1, embed_dim) — includes CLS token at index 0.
        """
        # Bypass the final pooling/projection to get raw patch tokens.
        vt = self.visual.trunk          # timm VisionTransformer
        x = vt.patch_embed(x)
        x = vt._pos_embed(x)
        x = vt.norm_pre(x)
        x = vt.blocks(x)
        x = vt.norm(x)
        return x  # (B, 1 + num_patches, D)

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        for p in self.visual.parameters():
            p.requires_grad_(False)

    def unfreeze(self, unfreeze_last_n_blocks: int | None = None) -> None:
        """
        Unfreeze backbone parameters.

        Args:
            unfreeze_last_n_blocks: If given, only unfreeze the last N
                transformer blocks (plus norm and head). Useful for staged
                fine-tuning to prevent catastrophic forgetting.
        """
        if unfreeze_last_n_blocks is None:
            for p in self.visual.parameters():
                p.requires_grad_(True)
        else:
            blocks = list(self.visual.trunk.blocks)
            n = len(blocks)
            for i, block in enumerate(blocks):
                requires_grad = i >= (n - unfreeze_last_n_blocks)
                for p in block.parameters():
                    p.requires_grad_(requires_grad)
            # Always unfreeze final norm and projection.
            for p in self.visual.trunk.norm.parameters():
                p.requires_grad_(True)
            if hasattr(self.visual, "head"):
                for p in self.visual.head.parameters():
                    p.requires_grad_(True)

    @property
    def num_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
