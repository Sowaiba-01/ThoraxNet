"""
/api/v1/predict — main prediction endpoint.

Accepts:
  - multipart/form-data with 'file' (image) and optional metadata fields.

Returns:
  - JSON PredictionResponse
  - GradCAM images served at /api/v1/gradcam/{session_id}/{class_name}
"""

from __future__ import annotations

import io
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image

from api.inference import pipeline
from api.schemas import PredictionResponse

router = APIRouter(prefix="/api/v1", tags=["prediction"])

# In-memory GradCAM store — maps session_id → {class_name: PIL Image}
# In production, replace with Redis or object storage.
_gradcam_store: dict[str, dict] = {}
_MAX_STORE_SIZE = 100   # evict oldest when over limit


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Analyze chest X-ray",
    description=(
        "Upload a chest X-ray image (PNG or JPEG). "
        "Returns multi-label pathology predictions with uncertainty estimates, "
        "an auto-generated radiology report, and GradCAM visualization links."
    ),
)
async def predict(
    file: Annotated[UploadFile, File(description="Chest X-ray image (PNG/JPEG)")],
    patient_age: Annotated[Optional[float], Form()] = None,
    patient_gender: Annotated[Optional[str], Form()] = None,
    generate_report: Annotated[
        bool, Form(description="Generate the LLM narrative report (adds ~1.3s)")
    ] = True,
    generate_gradcam: Annotated[
        bool, Form(description="Generate GradCAM overlays for positive findings")
    ] = True,
) -> PredictionResponse:
    if not pipeline.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading. Please retry in a few seconds.",
        )

    # Validate content type.
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Use PNG or JPEG.",
        )

    # Validate file size (max 10 MB).
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds 10 MB limit.",
        )

    # Validate it's actually an image.
    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File is not a valid image.",
        )

    # Run inference.
    try:
        response = await pipeline.predict(
            image_bytes=contents,
            patient_age=patient_age,
            patient_gender=patient_gender,
            generate_report=generate_report,
            generate_gradcam=generate_gradcam,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )

    # Store GradCAM overlays so the client can fetch them by session id.
    #
    # BUG FIX (v1.1.0): this previously read `pipeline.gradcam._last_overlays`,
    # an attribute that never existed — the pipeline built its overlays in a
    # local variable and discarded them. Every /gradcam/{id}/{cls} request
    # therefore 404'd or crashed with AttributeError. The pipeline now records
    # overlays on the ViTGradCAM instance via generate_overlays().
    if response.gradcam_available:
        session_id = str(uuid.uuid4())
        _gradcam_store[session_id] = dict(pipeline.gradcam._last_overlays)
        # Evict oldest if over limit (OrderedDict semantics: dicts are ordered).
        while len(_gradcam_store) > _MAX_STORE_SIZE:
            oldest_key = next(iter(_gradcam_store))
            del _gradcam_store[oldest_key]
        response.gradcam_session_id = session_id

    return response


@router.get(
    "/gradcam/{session_id}/{class_name}",
    summary="Retrieve GradCAM heatmap",
    response_class=StreamingResponse,
)
async def get_gradcam(session_id: str, class_name: str) -> StreamingResponse:
    """Return the GradCAM overlay image for a specific session and class."""
    if session_id not in _gradcam_store:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    overlays = _gradcam_store[session_id]
    if class_name not in overlays:
        raise HTTPException(
            status_code=404,
            detail=f"No GradCAM available for class '{class_name}' in this session.",
        )
    pil_img = overlays[class_name]
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
