"""Unit tests for clinical metrics computation."""
import numpy as np
import pytest
from training.metrics import compute_metrics


def test_perfect_predictions():
    targets = np.eye(14, dtype=np.float32)
    probs   = np.eye(14, dtype=np.float32)
    metrics = compute_metrics(targets, probs, prefix="test/")
    assert metrics["test/macro/auc"] == pytest.approx(1.0, abs=0.01)


def test_random_predictions_auc_near_half():
    np.random.seed(42)
    targets = (np.random.rand(200, 14) > 0.8).astype(np.float32)
    probs   = np.random.rand(200, 14).astype(np.float32)
    metrics = compute_metrics(targets, probs)
    # Random predictions should give AUC near 0.5
    assert 0.4 < metrics["macro/auc"] < 0.6


def test_metrics_keys_present():
    targets = (np.random.rand(100, 14) > 0.8).astype(np.float32)
    probs   = np.random.rand(100, 14).astype(np.float32)
    metrics = compute_metrics(targets, probs, prefix="val/")
    required = ["val/macro/auc", "val/macro/sensitivity", "val/macro/specificity", "val/macro/f1"]
    for key in required:
        assert key in metrics, f"Missing key: {key}"
