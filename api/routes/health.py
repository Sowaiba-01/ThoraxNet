from fastapi import APIRouter
from api.inference import pipeline
from api.schemas import HealthResponse
import torch

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=pipeline.is_loaded,
        device=str(pipeline.device),
        version="1.0.0",
    )


@router.get("/", include_in_schema=False)
async def root():
    return {"message": "ChestAI API — see /docs for Swagger UI."}
