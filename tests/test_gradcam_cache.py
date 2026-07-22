"""
Tests for GradCAM caching and the overlay-store regression.

Context: before v1.1.0, api/routes/predict.py read
`pipeline.gradcam._last_overlays`, an attribute that was never assigned —
the pipeline built overlays into a local variable and dropped them. Every
GradCAM retrieval failed. test_last_overlays_is_populated pins the fix.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from explainability.gradcam import ViTGradCAM


class _StubGradCAM(ViTGradCAM):
    """
    Exercises the cache and overlay plumbing without a real ViT.

    Bypasses __init__'s hook registration (which needs a real backbone) and
    replaces generate() with a counted stub, so we can assert on cache
    behaviour rather than on heatmap values.
    """

    def __init__(self, cache_size: int = 4) -> None:
        from collections import OrderedDict
        self.model = None
        self._activations = None
        self._gradients = None
        self._hook_handles = []
        self._last_overlays = {}
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.generate_calls = 0

    def generate(self, image_tensor, class_idx, patch_size=16, image_size=224):
        self.generate_calls += 1
        rng = np.random.default_rng(class_idx)
        return rng.random((image_size, image_size)).astype(np.float32)


@pytest.fixture
def cam():
    return _StubGradCAM()


@pytest.fixture
def tensor():
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def pil_image():
    return Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L")


# --------------------------------------------------------------------------
# Cache key
# --------------------------------------------------------------------------

def test_image_key_is_stable():
    assert ViTGradCAM.image_key(b"abc") == ViTGradCAM.image_key(b"abc")


def test_image_key_differs_for_different_bytes():
    assert ViTGradCAM.image_key(b"abc") != ViTGradCAM.image_key(b"abd")


# --------------------------------------------------------------------------
# Cache behaviour
# --------------------------------------------------------------------------

def test_second_call_hits_cache(cam, tensor):
    cam.generate_cached(tensor, 0, image_key="img1")
    cam.generate_cached(tensor, 0, image_key="img1")
    assert cam.generate_calls == 1
    assert cam.cache_hits == 1
    assert cam.cache_misses == 1


def test_cached_result_is_identical(cam, tensor):
    a = cam.generate_cached(tensor, 3, image_key="img1")
    b = cam.generate_cached(tensor, 3, image_key="img1")
    assert np.array_equal(a, b)


def test_different_class_is_a_separate_entry(cam, tensor):
    """Cache is keyed on (image, class) — Effusion must not return the
    Cardiomegaly heatmap for the same X-ray."""
    cam.generate_cached(tensor, 0, image_key="img1")
    cam.generate_cached(tensor, 1, image_key="img1")
    assert cam.generate_calls == 2


def test_different_image_is_a_separate_entry(cam, tensor):
    cam.generate_cached(tensor, 0, image_key="img1")
    cam.generate_cached(tensor, 0, image_key="img2")
    assert cam.generate_calls == 2


def test_no_key_bypasses_cache(cam, tensor):
    cam.generate_cached(tensor, 0, image_key=None)
    cam.generate_cached(tensor, 0, image_key=None)
    assert cam.generate_calls == 2
    assert cam.cache_hits == 0


def test_lru_evicts_oldest_entry(cam, tensor):
    for i in range(5):  # cache_size is 4
        cam.generate_cached(tensor, 0, image_key=f"img{i}")
    assert len(cam._cache) == 4
    assert ("img0", 0) not in cam._cache
    assert ("img4", 0) in cam._cache


def test_lru_recency_promotes_on_hit(cam, tensor):
    """A re-used entry must survive eviction ahead of a stale one."""
    for i in range(4):
        cam.generate_cached(tensor, 0, image_key=f"img{i}")
    cam.generate_cached(tensor, 0, image_key="img0")   # promote img0
    cam.generate_cached(tensor, 0, image_key="img9")   # forces one eviction
    assert ("img0", 0) in cam._cache
    assert ("img1", 0) not in cam._cache


def test_cache_stats_reports_hit_rate(cam, tensor):
    cam.generate_cached(tensor, 0, image_key="img1")
    cam.generate_cached(tensor, 0, image_key="img1")
    stats = cam.cache_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_cache_stats_hit_rate_zero_when_unused(cam):
    assert cam.cache_stats()["hit_rate"] == 0.0


# --------------------------------------------------------------------------
# Overlay store — the actual regression
# --------------------------------------------------------------------------

def test_last_overlays_is_populated(cam, tensor, pil_image):
    """
    REGRESSION: _last_overlays must exist and be filled after
    generate_overlays(). api/routes/predict.py reads it directly to build
    the session store; when it was missing, every GradCAM fetch 404'd.
    """
    overlays = cam.generate_overlays(tensor, pil_image, ["Atelectasis", "Effusion"])
    assert set(overlays) == {"Atelectasis", "Effusion"}
    assert cam._last_overlays == overlays
    assert all(isinstance(v, Image.Image) for v in overlays.values())


def test_generate_overlays_resets_between_calls(cam, tensor, pil_image):
    """Stale overlays from a previous scan must not leak into a new session."""
    cam.generate_overlays(tensor, pil_image, ["Atelectasis"])
    cam.generate_overlays(tensor, pil_image, ["Hernia"])
    assert set(cam._last_overlays) == {"Hernia"}


def test_generate_overlays_empty_class_list(cam, tensor, pil_image):
    assert cam.generate_overlays(tensor, pil_image, []) == {}


def test_generate_overlays_uses_cache(cam, tensor, pil_image):
    cam.generate_overlays(tensor, pil_image, ["Atelectasis"], image_key="img1")
    cam.generate_overlays(tensor, pil_image, ["Atelectasis"], image_key="img1")
    assert cam.generate_calls == 1


def test_overlay_output_size_matches_heatmap(cam, tensor, pil_image):
    overlays = cam.generate_overlays(tensor, pil_image, ["Atelectasis"])
    assert overlays["Atelectasis"].size == (224, 224)
