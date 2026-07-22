"""
Pydantic schemas for API request/response validation.

Using Pydantic v2 for stricter type checking and better OpenAPI docs.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Optional metadata sent alongside the image upload.
    Image itself is sent as multipart/form-data, not JSON.
    """
    patient_age: Optional[float] = Field(
        None, ge=0, le=120, description="Patient age in years (optional)"
    )
    patient_gender: Optional[str] = Field(
        None, description="Patient gender: 'M' or 'F' (optional)"
    )

    @field_validator("patient_gender")
    @classmethod
    def validate_gender(cls, v):
        if v is not None and v.upper() not in ("M", "F"):
            raise ValueError("patient_gender must be 'M' or 'F'")
        return v.upper() if v else v


class FindingResult(BaseModel):
    """Single pathology finding with probability and uncertainty."""
    name: str = Field(description="Pathology class name")
    probability: float = Field(ge=0.0, le=1.0, description="Mean predicted probability")
    uncertainty: float = Field(ge=0.0, description="MC Dropout std deviation")
    present: bool = Field(description="True if probability >= threshold")
    high_uncertainty: bool = Field(description="True if uncertainty >= 0.15")


class PredictionResponse(BaseModel):
    """Full prediction response returned to the client."""
    findings: list[FindingResult] = Field(description="Per-class findings")
    entropy: float = Field(description="Overall predictive entropy")
    report: Optional[str] = Field(
        None,
        description="Auto-generated radiology report (null if generate_report=false)",
    )
    gradcam_available: bool = Field(
        description="True if GradCAM images were generated"
    )
    gradcam_classes: list[str] = Field(
        default_factory=list,
        description="Classes for which GradCAM heatmaps are available",
    )
    gradcam_session_id: Optional[str] = Field(
        None,
        description=(
            "Session id for retrieving overlays at "
            "/api/v1/gradcam/{session_id}/{class_name}"
        ),
    )
    inference_time_ms: float = Field(description="Server-side inference time in ms")
    stage_timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-stage latency breakdown in ms: preprocess, mc_dropout, "
            "gradcam, report. Used for profiling and regression tracking."
        ),
    )
    model_version: str = Field(description="Model version identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "findings": [
                    {
                        "name": "Atelectasis",
                        "probability": 0.73,
                        "uncertainty": 0.08,
                        "present": True,
                        "high_uncertainty": False,
                    }
                ],
                "entropy": 1.24,
                "report": "FINDINGS:\nAtelectasis present...",
                "gradcam_available": True,
                "gradcam_classes": ["Atelectasis"],
                "inference_time_ms": 183.2,
                "stage_timings_ms": {
                    "preprocess": 12.4,
                    "mc_dropout": 96.1,
                    "gradcam": 61.8,
                    "report": 12.9,
                },
                "model_version": "1.1.0",
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str
