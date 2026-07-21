"""Unit tests for model architecture and uncertainty estimation."""
import pytest
import torch
from models.classifier import ChestAIClassifier
from models.uncertainty import mc_predict, enable_dropout


@pytest.fixture(scope="module")
def model():
    """
    Lightweight model for testing — skips the BioMedCLIP download in CI.

    The stub backbone must honour the input batch size. An earlier version
    returned a hard-coded `torch.zeros(2, 512)` regardless of what it was
    given, which meant:

      * the batch dimension was silently wrong for any input that wasn't
        batch-2, so shape assertions in these tests could never hold; and
      * all-zero features propagate as zeros through LayerNorm -> Linear
        (biases are zero-initialised), so dropout produced no variance at all
        and the uncertainty assertions were vacuous.

    The stub below returns fixed, non-zero features shaped to the input batch,
    so the only source of variance across MC passes is dropout — which is
    precisely what these tests are meant to measure.
    """
    import unittest.mock as mock

    _FIXED_FEATURES = torch.randn(1, 512, generator=torch.Generator().manual_seed(0))

    def fake_visual(x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, 512), identical features for every call.
        return _FIXED_FEATURES.expand(x.shape[0], 512).clone()

    with mock.patch("models.backbone.create_model_from_pretrained") as m:
        mock_visual = mock.MagicMock(side_effect=fake_visual)
        mock_visual.output_dim = 512
        m.return_value = (mock.MagicMock(visual=mock_visual), None)
        clf = ChestAIClassifier(num_classes=14, freeze_backbone=False)
    return clf


def test_output_shape(model):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 14), f"Expected (2,14), got {out.shape}"


def test_output_range_after_sigmoid(model):
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_mc_dropout_produces_variance(model):
    x = torch.randn(1, 3, 224, 224)
    result = mc_predict(model, x, n_samples=5)
    assert result["std"].shape == (1, 14)
    # Uncertainty should be positive (dropout creates variance)
    assert result["std"].mean().item() > 0


def test_uncertainty_keys(model):
    x = torch.randn(1, 3, 224, 224)
    result = mc_predict(model, x, n_samples=5)
    assert set(result.keys()) == {"mean", "std", "entropy", "samples"}
    assert result["samples"].shape[0] == 5


def test_enable_dropout_sets_train_mode():
    import torch.nn as nn
    model_small = nn.Sequential(nn.Linear(10, 10), nn.Dropout(0.3), nn.Linear(10, 5))
    model_small.eval()
    enable_dropout(model_small)
    dropout_layers = [m for m in model_small.modules() if isinstance(m, nn.Dropout)]
    assert all(d.training for d in dropout_layers)
