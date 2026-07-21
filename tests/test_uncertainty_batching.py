"""
Tests for the batched Monte Carlo Dropout implementation.

The v1.1.0 perf change replaced T sequential forward passes with a single
forward over a T-tiled batch. These tests pin the properties that change had
to preserve: correct output shapes, genuine per-sample stochasticity, chunking
correctness, and statistical equivalence to the sequential version.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.uncertainty import enable_dropout, mc_predict


class TinyDropoutNet(nn.Module):
    """
    Stand-in for ChestAIClassifier: real dropout, no 300MB download.

    Uses a fixed linear projection so results are deterministic given a seed,
    with dropout as the only source of variance.
    """

    def __init__(self, num_classes: int = 14) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 8 * 8, 32)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pool whatever spatial size we get down to 8x8 so tests can use
        # small inputs while matching the real model's (B, 3, H, W) contract.
        x = nn.functional.adaptive_avg_pool2d(x, (8, 8))
        return self.fc2(self.drop(torch.relu(self.fc1(self.flatten(x)))))


def _make(batch: int = 1) -> tuple[TinyDropoutNet, torch.Tensor]:
    torch.manual_seed(0)
    model = TinyDropoutNet()
    x = torch.randn(batch, 3, 8, 8)
    return model, x


def test_output_shapes_single_image():
    model, x = _make(batch=1)
    r = mc_predict(model, x, n_samples=20)
    assert r["mean"].shape == (1, 14)
    assert r["std"].shape == (1, 14)
    assert r["entropy"].shape == (1,)
    assert r["samples"].shape == (20, 1, 14)


def test_output_shapes_multi_image_batch():
    """Batched MC must not scramble the (T, B, C) layout for B > 1."""
    model, x = _make(batch=4)
    r = mc_predict(model, x, n_samples=8)
    assert r["mean"].shape == (4, 14)
    assert r["samples"].shape == (8, 4, 14)


def test_samples_are_actually_stochastic():
    """
    Tiling along the batch dim only works if each tiled copy gets its own
    dropout mask. If masks were shared, every sample would be identical and
    std would collapse to zero — silently destroying uncertainty estimates.
    """
    model, x = _make()
    r = mc_predict(model, x, n_samples=20)
    s = r["samples"]
    assert not torch.allclose(s[0], s[1]), "MC samples are identical — dropout not active per-sample"
    assert r["std"].mean().item() > 1e-4


def test_chunking_matches_unchunked_shape():
    """max_chunk must only affect memory, never the returned shape."""
    model, x = _make()
    small = mc_predict(model, x, n_samples=20, max_chunk=3)
    large = mc_predict(model, x, n_samples=20, max_chunk=64)
    assert small["samples"].shape == large["samples"].shape == (20, 1, 14)


def test_non_divisible_chunk_returns_exact_sample_count():
    """20 samples at chunk size 7 → 7 + 7 + 6, not 21."""
    model, x = _make()
    r = mc_predict(model, x, n_samples=20, max_chunk=7)
    assert r["samples"].shape[0] == 20


def test_single_sample_std_is_finite():
    """std with n_samples=1 must be 0, not NaN (unbiased=False path)."""
    model, x = _make()
    r = mc_predict(model, x, n_samples=1)
    assert torch.isfinite(r["std"]).all()
    assert torch.allclose(r["std"], torch.zeros_like(r["std"]))


def test_probabilities_in_valid_range():
    model, x = _make()
    r = mc_predict(model, x, n_samples=10)
    assert r["mean"].min() >= 0.0 and r["mean"].max() <= 1.0
    assert r["samples"].min() >= 0.0 and r["samples"].max() <= 1.0


def test_entropy_is_non_negative_and_finite():
    model, x = _make()
    r = mc_predict(model, x, n_samples=10)
    assert torch.isfinite(r["entropy"]).all()
    assert (r["entropy"] >= 0).all()


def test_batched_matches_sequential_in_distribution():
    """
    The batched and sequential implementations draw from the same distribution,
    so their means should agree closely over enough samples. This is the test
    that would have caught a genuine behavioural regression in the rewrite.
    """
    model, x = _make()

    torch.manual_seed(123)
    batched = mc_predict(model, x, n_samples=400)["mean"]

    torch.manual_seed(123)
    model.eval()
    enable_dropout(model)
    with torch.no_grad():
        seq = torch.stack([torch.sigmoid(model(x)) for _ in range(400)]).mean(dim=0)

    # Monte Carlo error at n=400 is well under 0.05 for a bounded [0,1] variable.
    assert torch.allclose(batched, seq, atol=0.05), (
        f"batched and sequential means diverge: max diff "
        f"{(batched - seq).abs().max().item():.4f}"
    )


def test_batch_collapsing_model_raises_clear_error():
    """
    Batched MC is only valid for models that map batch N -> batch N. A model
    that fixes or collapses the batch dimension must fail loudly, not reshape
    into misaligned samples.
    """
    import pytest

    class BatchCollapsingNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(2, 14)  # ignores input batch entirely

    with pytest.raises(RuntimeError, match="preserve the batch dimension"):
        mc_predict(BatchCollapsingNet(), torch.randn(1, 3, 8, 8), n_samples=5)


def test_model_left_in_eval_mode_after_call():
    """
    mc_predict flips dropout to train mode. If it does not restore eval state,
    the *next* ordinary prediction silently becomes stochastic — a nasty
    action-at-a-distance bug.
    """
    model, x = _make()
    mc_predict(model, x, n_samples=5)
    assert not model.training
