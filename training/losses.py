
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalBCELoss(nn.Module):
    """
    Per-class weighted focal binary cross-entropy.

    Args:
        pos_weight: (num_classes,) tensor — ratio of negatives to positives per class.
                    Computed from training set via dataset.get_pos_weights().
        gamma: Focal exponent. 0 = standard BCE, 2 = default focal.
        alpha: Balancing factor for positive class. None = no balancing.
        reduction: 'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        # pos_weight is registered as buffer so it moves with .to(device).
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw model outputs (pre-sigmoid).
            targets: (B, C) float binary labels in {0, 1}.

        Returns:
            Scalar loss.
        """
        # Standard BCE with logits (numerically stable).
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",           # keep (B, C) for focal weighting
            pos_weight=self.pos_weight,
        )

        # Focal weight: (1 - p_t)^gamma
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha balancing
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce  # (B, C)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def build_loss(cfg: dict, pos_weight: torch.Tensor | None = None) -> WeightedFocalBCELoss:
    """Construct the loss from config."""
    loss_cfg = cfg.get("loss", {})
    return WeightedFocalBCELoss(
        pos_weight=pos_weight * loss_cfg.get("pos_weight_scale", 5.0)
        if pos_weight is not None else None,
        gamma=loss_cfg.get("focal_gamma", 2.0),
        alpha=loss_cfg.get("focal_alpha", 0.25),
    )
