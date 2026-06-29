"""
GradCAM for Vision Transformers (ViT-GradCAM).

Standard GradCAM hooks into convolutional feature maps. For ViTs we hook into
the last transformer block's attention output, reshape the 1D patch sequence
back to 2D, and overlay it on the original image.

References:
    - Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (2017)
    - Chefer et al. "Transformer Interpretability Beyond Attention Visualization" (2021)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from data.transforms import denormalize


class ViTGradCAM:
    """
    GradCAM for the BioMedCLIP ViT backbone.

    Hooks into the output of the last transformer block. Gradients flowing
    back through that output are averaged over the patch dimension to produce
    per-patch importance weights, then reshaped to a spatial heatmap.

    Usage:
        cam = ViTGradCAM(model)
        heatmap_pil = cam.generate(image_tensor, class_idx)
        overlay_pil = cam.overlay(original_pil, heatmap_pil)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hook_handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Hook into the last ViT transformer block."""
        last_block = self.model.backbone.visual.trunk.blocks[-1]

        def forward_hook(module, input, output):
            # output: (B, num_patches+1, D) — includes CLS token
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: (B, num_patches+1, D)
            self._gradients = grad_output[0].detach()

        self._hook_handles.append(
            last_block.register_forward_hook(forward_hook)
        )
        self._hook_handles.append(
            last_block.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int,
        patch_size: int = 16,
        image_size: int = 224,
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap for a single image and target class.

        Args:
            image_tensor: (1, 3, H, W) normalized tensor.
            class_idx: Index into the 14-class output to explain.
            patch_size: ViT patch size (16 for ViT-B/16).
            image_size: Input image spatial size.

        Returns:
            (H, W) float32 heatmap in [0, 1].
        """
        self.model.zero_grad()
        self.model.eval()
        # Keep gradients for the target class logit.
        image_tensor = image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)          # (1, 14)
        score  = logits[0, class_idx]
        score.backward()

        # activations / gradients: (1, num_patches+1, D)
        acts = self._activations[0, 1:]            # drop CLS token → (num_patches, D)
        grads = self._gradients[0, 1:]             # (num_patches, D)

        # Weight each patch's activation by its gradient magnitude.
        weights = grads.mean(dim=-1)               # (num_patches,)
        cam = (weights[:, None] * acts).sum(dim=-1)  # (num_patches,)
        cam = torch.relu(cam)                      # only positive attributions

        # Reshape to spatial grid.
        num_patches_side = image_size // patch_size   # 14 for 224/16
        cam = cam.reshape(num_patches_side, num_patches_side).cpu().numpy()

        # Upsample to full image resolution.
        cam = cv2.resize(cam, (image_size, image_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)

    def overlay(
        self,
        original_pil: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """
        Blend GradCAM heatmap over the original X-ray.

        Args:
            original_pil: Original PIL image (grayscale or RGB).
            heatmap: (H, W) float32 array from generate().
            alpha: Heatmap opacity (0 = invisible, 1 = full overlay).
            colormap: OpenCV colormap constant.

        Returns:
            PIL.Image with heatmap blended.
        """
        orig = np.array(original_pil.convert("RGB"))
        orig_resized = cv2.resize(orig, (heatmap.shape[1], heatmap.shape[0]))

        heatmap_u8 = (heatmap * 255).astype(np.uint8)
        colored    = cv2.applyColorMap(heatmap_u8, colormap)
        colored    = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(orig_resized, 1 - alpha, colored, alpha, 0)
        return Image.fromarray(blended)

    def generate_all_classes(
        self,
        image_tensor: torch.Tensor,
        probs: np.ndarray,
        threshold: float = 0.5,
        image_size: int = 224,
    ) -> dict[str, np.ndarray]:
        """
        Generate GradCAM heatmaps for all classes above threshold.

        Returns:
            {class_name: heatmap_array}
        """
        from data.dataset import CLASSES
        results = {}
        for i, cls in enumerate(CLASSES):
            if probs[i] >= threshold:
                results[cls] = self.generate(image_tensor, i, image_size=image_size)
        return results
