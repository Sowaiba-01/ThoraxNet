
FROM python:3.11-slim AS base

# System dependencies for OpenCV + PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# ── Copy source ───────────────────────────────────────────────────────────────
COPY configs/ ./configs/
COPY data/     ./data/
COPY models/   ./models/
COPY training/ ./training/
COPY explainability/ ./explainability/
COPY fairness/ ./fairness/
COPY report_generation/ ./report_generation/
COPY api/      ./api/
# ── Non-root user (HF Spaces requirement) ────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 7860
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
