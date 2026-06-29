<div align="center">

# 🫁 ChestAI

**Uncertainty-aware multi-label chest X-ray diagnostic platform**

[![CI](https://github.com/YOUR_USERNAME/chestai/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/chestai/actions)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/YOUR_USERNAME/chestai-model)
[![Demo](https://img.shields.io/badge/🚀%20Demo-HF%20Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/chestai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What This Does

ChestAI detects **14 thoracic pathologies** in chest X-rays, tells you **how confident it is**, generates a structured **radiology report**, and highlights **which region of the image** triggered each finding.

| Feature | Implementation |
|---|---|
| Multi-label classification | BioMedCLIP ViT-B/16 fine-tuned on NIH ChestX-ray14 |
| Uncertainty estimation | Monte Carlo Dropout (20 stochastic passes) |
| Explainability | ViT-GradCAM heatmap overlay |
| Fairness audit | Per-subgroup AUC (age / gender) with disparity flagging |
| Report generation | LLaMA 3 70B via Groq free API |
| API | FastAPI + async inference |
| Frontend | Next.js PWA — installable on Android/iOS |
| Deployment | HuggingFace Spaces (free) + Vercel (free) |
| Training | Kaggle P100 GPU (free, 30 hrs/week) |

---

## Architecture

```
chest X-ray
     │
     ▼
BioMedCLIP ViT-B/16 ── (pre-trained on 15M PubMed biomedical image-text pairs)
     │
     ▼
Classification Head ── LayerNorm → Dropout → Linear(512→256) → GELU → Dropout → Linear(256→14)
     │
     ├── MC Dropout (×20) ──► mean probability + std uncertainty + entropy
     │
     ├── ViT-GradCAM ──────► spatial heatmap per positive class
     │
     ├── Fairness Audit ───► per-subgroup AUC (age/gender)
     │
     └── Groq LLaMA 3 70B ► structured radiology report
```

---

## Results (NIH ChestX-ray14 Test Set)

> Populated after training. Update with your actual numbers.

| Class | AUC | Sensitivity | Specificity |
|---|---|---|---|
| Atelectasis | — | — | — |
| Cardiomegaly | — | — | — |
| Effusion | — | — | — |
| Infiltration | — | — | — |
| Mass | — | — | — |
| Nodule | — | — | — |
| Pneumonia | — | — | — |
| Pneumothorax | — | — | — |
| Consolidation | — | — | — |
| Edema | — | — | — |
| Emphysema | — | — | — |
| Fibrosis | — | — | — |
| Pleural Thickening | — | — | — |
| Hernia | — | — | — |
| **Macro Average** | **—** | **—** | **—** |

*Compared to [CheXNet (Rajpurkar et al., 2017)](https://arxiv.org/abs/1711.05225) baseline: macro AUC 0.841*

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/chestai
cd chestai
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env:
#   MODEL_HUB_REPO=YOUR_USERNAME/chestai-model
#   GROQ_API_KEY=gsk_...  (free at console.groq.com)
```

### 3. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
# Swagger UI: http://localhost:7860/docs
```

### 4. Run the frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:7860 npm run dev
# http://localhost:3000
```

---

## Training (Free — Kaggle GPU)

1. Go to [kaggle.com](https://kaggle.com) → **Create Notebook**
2. Add dataset: search **"NIH Chest X-rays"** → Add Data
3. Enable **GPU P100** in Settings → Accelerator
4. Upload `notebooks/training_kaggle.ipynb`
5. Set your W&B key + HF token in cell 2/3
6. Run All — trains for ~30 epochs, saves best model to HuggingFace Hub

---

## Deployment (Free)

### API → HuggingFace Spaces

```bash
# After training, set GitHub secret HF_TOKEN
# Push to main → GitHub Actions auto-deploys to HF Spaces
git push origin main
```

### Frontend → Vercel

```bash
cd frontend
npx vercel --prod
# Set env var NEXT_PUBLIC_API_URL=https://YOUR_USERNAME-chestai.hf.space
```

---

## Project Structure

```
chestai/
├── configs/config.yaml          # All hyperparameters (single source of truth)
├── data/
│   ├── dataset.py               # NIH ChestX-ray14 PyTorch Dataset
│   └── transforms.py            # Train/val augmentation pipelines
├── models/
│   ├── backbone.py              # BioMedCLIP ViT-B/16 vision encoder
│   ├── classifier.py            # Multi-label head with MC Dropout
│   └── uncertainty.py           # MC Dropout inference + uncertainty metrics
├── training/
│   ├── trainer.py               # Training loop (AMP, staged unfreezing, W&B)
│   ├── losses.py                # Weighted Focal BCE loss
│   └── metrics.py               # AUC-ROC, sensitivity, specificity, F1
├── explainability/gradcam.py    # ViT-GradCAM heatmap generation
├── fairness/audit.py            # Demographic subgroup AUC audit
├── report_generation/generator.py  # Groq LLaMA radiology report generator
├── api/
│   ├── main.py                  # FastAPI app (lifespan, CORS, rate limiting)
│   ├── inference.py             # Singleton inference pipeline
│   ├── schemas.py               # Pydantic request/response models
│   └── routes/                  # predict.py + health.py
├── frontend/                    # Next.js PWA
├── notebooks/training_kaggle.ipynb  # End-to-end Kaggle training notebook
├── tests/                       # pytest unit tests
├── docker/Dockerfile            # Production container for HF Spaces
└── .github/workflows/ci.yml    # Lint + test + deploy pipeline
```

---

## What Makes This Different

Most chest X-ray projects train DenseNet121 and call it done. This project:

1. **Uncertainty quantification** — Monte Carlo Dropout gives per-prediction confidence intervals, flagging ambiguous cases for radiologist review. Almost no public projects have this.

2. **Medical foundation model** — BioMedCLIP pre-trained on 15M PubMed biomedical image-text pairs, not ImageNet. Domain-specific pretraining matters.

3. **Fairness audit** — Per-class AUC broken down by age and gender subgroups, with disparity flagging. Required for any real clinical AI deployment.

4. **Full system, not just a model** — Async API, PWA frontend, GradCAM, report generation, CI/CD, Docker. Deployed and accessible.

---

## Disclaimer

ChestAI is a research project. It is **not approved for clinical use** and should not be used for medical diagnosis. Always consult a qualified radiologist.

---

## License

MIT — see [LICENSE](LICENSE)
