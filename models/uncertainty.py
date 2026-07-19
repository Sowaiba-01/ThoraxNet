from __future__ import annotations
import torch
import torch.nn as nn
def enable_dropout(model: nn.Module) -> None:
    """Switch only Dropout layers to train mode (keep BatchNorm in eval mode)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
) -> dict[str, torch.Tensor]:
    """
    Run Monte Carlo Dropout inference.

    Args:
        model: ChestAIClassifier instance.
        x: (B, 3, H, W) image tensor (on correct device).
        n_samples: Number of stochastic forward passes.

    Returns:
        dict with keys:
            'mean'    : (B, C) mean probability across samples
            'std'     : (B, C) std deviation (per-class uncertainty)
            'entropy' : (B,)   predictive entropy (total uncertainty)
            'samples' : (T, B, C) all probability samples (for plotting)
    """
    model.eval()
    enable_dropout(model)

    samples = []
    for _ in range(n_samples):
        logits = model(x)                         # (B, C)
        probs = torch.sigmoid(logits)             # (B, C)
        samples.append(probs)

    samples = torch.stack(samples)               # (T, B, C)
    mean  = samples.mean(dim=0)                  # (B, C)
    std   = samples.std(dim=0)                   # (B, C)

    # Predictive entropy: H = -sum_c [p_c * log(p_c) + (1-p_c) * log(1-p_c)]
    eps = 1e-8
    entropy = -(
        mean * (mean + eps).log() + (1 - mean) * (1 - mean + eps).log()
    ).sum(dim=-1)                                # (B,)

    model.eval()  # restore full eval mode

    return {
        "mean": mean,
        "std": std,
        "entropy": entropy,
        "samples": samples,
    }


def uncertainty_flag(
    std: torch.Tensor,
    threshold: float = 0.15,
) -> list[list[str]]:
    """
    Return human-readable uncertainty flags per sample per class.

    Args:
        std: (B, C) uncertainty tensor.
        threshold: std above this value is flagged as uncertain.

    Returns:
        List of lists: outer = batch, inner = class names with HIGH uncertainty.
    """
    from data.dataset import CLASSES
    flags = []
    for row in std:
        uncertain = [CLASSES[i] for i, v in enumerate(row) if v.item() > threshold]
        flags.append(uncertain)
    return flags
