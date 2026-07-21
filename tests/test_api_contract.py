"""
Contract tests for the API schemas and clinical decision thresholds.

These run without torch weights or a live server — they pin the parts of the
public contract that clients (the Next.js frontend) depend on, plus the
per-class thresholds that determine what counts as a positive finding.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    FindingResult,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from data.dataset import CLASSES, NUM_CLASSES


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------

def test_every_class_has_a_threshold():
    """A missing threshold silently falls back to 0.5, which is wrong for
    every class in this model. Fail loudly instead."""
    from api.inference import THRESHOLDS
    missing = set(CLASSES) - set(THRESHOLDS)
    assert not missing, f"classes with no tuned threshold: {sorted(missing)}"


def test_thresholds_are_in_calibrated_range():
    """Tuned thresholds live in [0.58, 0.75]. A value outside that band means
    someone edited it by hand rather than re-running calibration."""
    from api.inference import THRESHOLDS
    for cls, t in THRESHOLDS.items():
        assert 0.5 <= t <= 0.9, f"{cls} threshold {t} outside plausible range"


def test_no_extra_thresholds():
    from api.inference import THRESHOLDS
    extra = set(THRESHOLDS) - set(CLASSES)
    assert not extra, f"thresholds for unknown classes: {sorted(extra)}"


def test_class_list_length_matches_num_classes():
    assert len(CLASSES) == NUM_CLASSES == 14


def test_class_order_is_stable():
    """Class indices are baked into the checkpoint. Reordering CLASSES would
    silently remap every prediction."""
    assert CLASSES[0] == "Atelectasis"
    assert CLASSES[-1] == "Hernia"


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

def _finding(**kw) -> FindingResult:
    base = dict(
        name="Atelectasis", probability=0.73, uncertainty=0.08,
        present=True, high_uncertainty=False,
    )
    base.update(kw)
    return FindingResult(**base)


def test_finding_rejects_probability_above_one():
    with pytest.raises(ValidationError):
        _finding(probability=1.4)


def test_finding_rejects_negative_probability():
    with pytest.raises(ValidationError):
        _finding(probability=-0.1)


def test_finding_rejects_negative_uncertainty():
    with pytest.raises(ValidationError):
        _finding(uncertainty=-0.01)


def test_prediction_response_allows_null_report():
    """generate_report=false returns findings with report=None. The schema
    must permit that or the fast path 500s."""
    r = PredictionResponse(
        findings=[_finding()],
        entropy=1.2,
        report=None,
        gradcam_available=False,
        inference_time_ms=120.0,
        model_version="1.1.0",
    )
    assert r.report is None


def test_stage_timings_defaults_to_empty_dict():
    r = PredictionResponse(
        findings=[_finding()], entropy=1.0, report="x",
        gradcam_available=False, inference_time_ms=1.0, model_version="1.1.0",
    )
    assert r.stage_timings_ms == {}


def test_stage_timings_round_trip():
    stages = {"preprocess": 12.4, "mc_dropout": 96.1, "gradcam": 61.8, "report": 12.9}
    r = PredictionResponse(
        findings=[_finding()], entropy=1.0, report="x",
        gradcam_available=True, gradcam_classes=["Atelectasis"],
        gradcam_session_id="abc-123",
        inference_time_ms=183.2, stage_timings_ms=stages, model_version="1.1.0",
    )
    assert r.stage_timings_ms["mc_dropout"] == 96.1
    assert r.gradcam_session_id == "abc-123"


def test_gradcam_session_id_optional():
    r = PredictionResponse(
        findings=[_finding()], entropy=1.0, report="x",
        gradcam_available=False, inference_time_ms=1.0, model_version="1.1.0",
    )
    assert r.gradcam_session_id is None


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("gender,expected", [("m", "M"), ("F", "F"), ("f", "F")])
def test_gender_is_normalised_to_uppercase(gender, expected):
    assert PredictionRequest(patient_gender=gender).patient_gender == expected


def test_gender_rejects_invalid_value():
    with pytest.raises(ValidationError):
        PredictionRequest(patient_gender="X")


def test_gender_allows_none():
    assert PredictionRequest(patient_gender=None).patient_gender is None


@pytest.mark.parametrize("age", [-1, 121])
def test_age_out_of_range_rejected(age):
    with pytest.raises(ValidationError):
        PredictionRequest(patient_age=age)


@pytest.mark.parametrize("age", [0, 45.5, 120])
def test_age_in_range_accepted(age):
    assert PredictionRequest(patient_age=age).patient_age == age


def test_health_response_shape():
    h = HealthResponse(status="ok", model_loaded=True, device="cuda", version="1.1.0")
    assert h.model_loaded is True
