
from __future__ import annotations

import asyncio
import io
import os
import time

import torch
import torch.nn as nn
from PIL import Image

from data.transforms import build_transforms
from models.classifier import ChestAIClassifier
from models.uncertainty import mc_predict
from explainability.gradcam import ViTGradCAM
from report_generation.generator import (
    GROQ_AVAILABLE,
    RadiologyReportGenerator,
    generate_report_fallback,
)
from api.schemas import FindingResult, PredictionResponse
from data.dataset import CLASSES

THRESHOLDS = {
    "Atelectasis": 0.63, "Cardiomegaly": 0.74, "Effusion": 0.66,
    "Infiltration": 0.58, "Mass": 0.64, "Nodule": 0.58,
    "Pneumonia": 0.67, "Pneumothorax": 0.66, "Consolidation": 0.67,
    "Edema": 0.75, "Emphysema": 0.61, "Fibrosis": 0.60,
    "Pleural_Thickening": 0.61, "Hernia": 0.62,
}
UNCERTAINTY_THRESHOLD = 0.15


class InferencePipeline:
    _instance: "InferencePipeline | None" = None

    def __init__(self) -> None:
        self.model: ChestAIClassifier | None = None
        self.gradcam: ViTGradCAM | None = None
        self.report_gen: RadiologyReportGenerator | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_transforms("val", image_size=224)
        self._lock = asyncio.Lock()
        self.model_version = "1.1.0"
        self.mc_samples = int(os.environ.get("MC_SAMPLES", "20"))

    async def load(self) -> None:
        from huggingface_hub import hf_hub_download

        model_repo = os.environ.get("MODEL_HUB_REPO", "Sowaiba01/chestai-model")
        checkpoint_path = os.environ.get("MODEL_CHECKPOINT", "")

        if not checkpoint_path and model_repo:
            print(f"[Pipeline] Downloading model from HF Hub: {model_repo}")
            checkpoint_path = hf_hub_download(
                repo_id=model_repo,
                filename="chestai_best.pt",
            )
        elif not checkpoint_path:
            raise RuntimeError(
                "Set MODEL_CHECKPOINT or MODEL_HUB_REPO env var."
            )

        print(f"[Pipeline] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        # The published checkpoint's classification head does not necessarily
        # match the default architecture in classifier.py (the shipped weights
        # use a 512-wide intermediate layer, while the class default is 256).
        # Read the true dimensions straight from the checkpoint and rebuild the
        # head to match, so the same code loads either variant. This block is
        # load-bearing: removing it produces a size-mismatch RuntimeError at
        # startup and the Space fails to boot.
        head_key = "head.2.weight"
        if head_key in state_dict:
            intermediate_dim = state_dict[head_key].shape[0]
            embed_dim = state_dict[head_key].shape[1]
        else:
            intermediate_dim = 512
            embed_dim = 512
        num_classes = len(CLASSES)
        dropout_rate = 0.3

        print(f"[Pipeline] Detected head: Linear({embed_dim} → {intermediate_dim} → {num_classes})")

        self.model = ChestAIClassifier()
        self.model.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, num_classes),
        ).to(self.device)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Pipeline] Missing keys: {missing}")
        if unexpected:
            print(f"[Pipeline] Unexpected keys: {unexpected}")

        self.model.to(self.device)
        self.model.eval()

        self.gradcam = ViTGradCAM(self.model)

        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key and GROQ_AVAILABLE:
            self.report_gen = RadiologyReportGenerator(api_key=groq_key)
            print("[Pipeline] Groq report generator initialized.")
        elif groq_key and not GROQ_AVAILABLE:
            print("[Pipeline] GROQ_API_KEY is set but the 'groq' package is not "
                  "installed — using fallback report generator.")
        else:
            print("[Pipeline] GROQ_API_KEY not set — using fallback report generator.")

        print(f"[Pipeline] Model loaded on {self.device}.")

    async def _build_report(
        self,
        mean_probs,
        std_probs,
        patient_info: dict,
    ) -> str:
        """
        Generate the narrative report WITHOUT blocking the event loop.

        RadiologyReportGenerator.generate() is a synchronous network call to
        Groq (~1.2-1.8s). Awaiting it directly inside an async handler blocks
        the loop and serialises every concurrent request behind it, which is
        why p95 collapsed under load before this change. asyncio.to_thread
        moves it to the default executor.
        """
        def _sync() -> str:
            try:
                if self.report_gen:
                    return self.report_gen.generate(
                        probs=mean_probs.tolist(),
                        stds=std_probs.tolist(),
                        patient_info=patient_info or None,
                    )
                return generate_report_fallback(mean_probs.tolist(), std_probs.tolist())
            except Exception as e:  # noqa: BLE001 - never fail a scan on report error
                print(f"[Pipeline] Report generation failed: {e}")
                return generate_report_fallback(mean_probs.tolist(), std_probs.tolist())

        return await asyncio.to_thread(_sync)

    async def predict(
        self,
        image_bytes: bytes,
        patient_age: float | None = None,
        patient_gender: str | None = None,
        generate_report: bool = True,
        generate_gradcam: bool = True,
    ) -> PredictionResponse:
        """
        Full inference: preprocess → MC dropout → GradCAM → report.

        Set generate_report=False or generate_gradcam=False to skip those
        stages; both are optional and neither affects the pathology
        probabilities. See stage_timings_ms on the response for a per-stage
        breakdown.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call await pipeline.load() first.")

        t0 = time.perf_counter()
        stage: dict[str, float] = {}

        def _mark(name: str, start: float) -> float:
            now = time.perf_counter()
            stage[name] = round((now - start) * 1000, 1)
            return now

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        t_pre = _mark("preprocess", t0)

        image_key = ViTGradCAM.image_key(image_bytes)

        # The model itself is not re-entrant (MC dropout toggles module state,
        # GradCAM mutates hook buffers), so GPU work stays under the lock.
        async with self._lock:
            mc_results = mc_predict(self.model, tensor, n_samples=self.mc_samples)
            mean_probs = mc_results["mean"][0].cpu().numpy()
            std_probs  = mc_results["std"][0].cpu().numpy()
            entropy    = float(mc_results["entropy"][0].item())
            t_mc = _mark("mc_dropout", t_pre)

            gradcam_classes = [
                cls for cls, p in zip(CLASSES, mean_probs)
                if p >= THRESHOLDS.get(cls, 0.5)
            ]

            if generate_gradcam and gradcam_classes:
                self.gradcam.generate_overlays(
                    tensor,
                    pil_image,
                    gradcam_classes,
                    image_key=image_key,
                )
            else:
                self.gradcam._last_overlays = {}
                gradcam_classes = [] if not generate_gradcam else gradcam_classes
            t_cam = _mark("gradcam", t_mc)

        findings = [
            FindingResult(
                name=cls,
                probability=float(p),
                uncertainty=float(s),
                present=float(p) >= THRESHOLDS.get(cls, 0.5),
                high_uncertainty=float(s) >= UNCERTAINTY_THRESHOLD,
            )
            for cls, p, s in zip(CLASSES, mean_probs, std_probs)
        ]

        patient_info = {}
        if patient_age is not None:
            patient_info["age"] = patient_age
        if patient_gender is not None:
            patient_info["gender"] = patient_gender

        if generate_report:
            report = await self._build_report(mean_probs, std_probs, patient_info)
        else:
            report = None
        _mark("report", t_cam)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PredictionResponse(
            findings=findings,
            entropy=entropy,
            report=report,
            gradcam_available=bool(generate_gradcam and gradcam_classes),
            gradcam_classes=gradcam_classes,
            inference_time_ms=round(elapsed_ms, 1),
            stage_timings_ms=stage,
            model_version=self.model_version,
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


pipeline = InferencePipeline()
