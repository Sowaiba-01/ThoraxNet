"""
Inference pipeline: image → predictions + GradCAM + report.

Loaded once at startup (singleton pattern) to avoid per-request model reload.
Thread-safe via asyncio lock for the GradCAM backward pass (not thread-safe).
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.transforms import build_transforms
from models.classifier import ChestAIClassifier
from models.uncertainty import mc_predict
from explainability.gradcam import ViTGradCAM
from report_generation.generator import RadiologyReportGenerator, generate_report_fallback
from api.schemas import FindingResult, PredictionResponse
from data.dataset import CLASSES


# Probability threshold for flagging a finding as present.
THRESHOLD = 0.5
UNCERTAINTY_THRESHOLD = 0.15


class InferencePipeline:
    """
    Singleton inference pipeline. Initialise once at app startup.

    Usage:
        pipeline = InferencePipeline()
        await pipeline.load()
        response = await pipeline.predict(image_bytes, metadata)
    """

    _instance: "InferencePipeline | None" = None

    def __init__(self) -> None:
        self.model: ChestAIClassifier | None = None
        self.gradcam: ViTGradCAM | None = None
        self.report_gen: RadiologyReportGenerator | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_transforms("val", image_size=224)
        self._lock = asyncio.Lock()
        self.model_version = "1.0.0"

    async def load(self) -> None:
        """Load model weights from HuggingFace Hub or local path."""
        from huggingface_hub import hf_hub_download

        model_repo = os.environ.get("MODEL_HUB_REPO", "")
        checkpoint_path = os.environ.get("MODEL_CHECKPOINT", "")

        if not checkpoint_path and model_repo:
            print(f"[Pipeline] Downloading model from HF Hub: {model_repo}")
            checkpoint_path = hf_hub_download(
                repo_id=model_repo,
                filename="chestai_best.pt",
            )
        elif not checkpoint_path:
            raise RuntimeError(
                "Set MODEL_CHECKPOINT (local path) or MODEL_HUB_REPO (HF Hub repo ID) env var."
            )

        print(f"[Pipeline] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model = ChestAIClassifier()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.gradcam = ViTGradCAM(self.model)

        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            self.report_gen = RadiologyReportGenerator(api_key=groq_key)
            print("[Pipeline] Groq report generator initialized.")
        else:
            print("[Pipeline] GROQ_API_KEY not set — using fallback report generator.")

        print(f"[Pipeline] Model loaded on {self.device}.")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    async def predict(
        self,
        image_bytes: bytes,
        patient_age: float | None = None,
        patient_gender: str | None = None,
    ) -> PredictionResponse:
        """
        Full async inference pipeline for a single X-ray image.

        Args:
            image_bytes: Raw image bytes (PNG or JPEG).
            patient_age: Optional patient age for report context.
            patient_gender: Optional 'M' or 'F'.

        Returns:
            PredictionResponse with findings, GradCAM, and report.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call await pipeline.load() first.")

        t0 = time.perf_counter()

        # Preprocess image.
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # MC Dropout inference (async-safe via lock for gradcam backward pass).
        async with self._lock:
            mc_results = mc_predict(self.model, tensor, n_samples=20)
            mean_probs = mc_results["mean"][0].cpu().numpy()    # (14,)
            std_probs  = mc_results["std"][0].cpu().numpy()     # (14,)
            entropy    = float(mc_results["entropy"][0].item())

            # GradCAM for all positive findings.
            gradcam_classes = [
                cls for cls, p in zip(CLASSES, mean_probs) if p >= THRESHOLD
            ]
            gradcam_images = {}
            for cls_name in gradcam_classes:
                cls_idx = CLASSES.index(cls_name)
                heatmap = self.gradcam.generate(tensor, cls_idx)
                overlay = self.gradcam.overlay(pil_image, heatmap)
                gradcam_images[cls_name] = overlay  # PIL Image (store server-side)

        # Build findings list.
        findings = [
            FindingResult(
                name=cls,
                probability=float(p),
                uncertainty=float(s),
                present=float(p) >= THRESHOLD,
                high_uncertainty=float(s) >= UNCERTAINTY_THRESHOLD,
            )
            for cls, p, s in zip(CLASSES, mean_probs, std_probs)
        ]

        # Generate report.
        patient_info = {}
        if patient_age is not None:
            patient_info["age"] = patient_age
        if patient_gender is not None:
            patient_info["gender"] = patient_gender

        try:
            if self.report_gen:
                report = self.report_gen.generate(
                    probs=mean_probs.tolist(),
                    stds=std_probs.tolist(),
                    patient_info=patient_info or None,
                )
            else:
                report = generate_report_fallback(mean_probs.tolist(), std_probs.tolist())
        except Exception as e:
            print(f"[Pipeline] Report generation failed: {e}")
            report = generate_report_fallback(mean_probs.tolist(), std_probs.tolist())

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PredictionResponse(
            findings=findings,
            entropy=entropy,
            report=report,
            gradcam_available=bool(gradcam_classes),
            gradcam_classes=gradcam_classes,
            inference_time_ms=round(elapsed_ms, 1),
            model_version=self.model_version,
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# Module-level singleton.
pipeline = InferencePipeline()
