---
license: mit
language:
- en
tags:
- medical-imaging
- chest-xray
- radiology
- multi-label-classification
- biomedclip
- vision-transformer
- uncertainty-quantification
- gradcam
- explainability
- pytorch
datasets:
- nih-chest-xray
metrics:
- roc_auc
model-index:
- name: ThoraxNet
  results:
  - task:
      type: image-classification
      name: Multi-label Chest X-Ray Classification
    dataset:
      name: NIH ChestX-ray14
      type: nih-chest-xray
      split: validation
    metrics:
    - type: roc_auc
      value: 0.8215
      name: Mean AUC (14 classes)
pipeline_tag: image-classification
---

# ThoraxNet — Multi-Label Chest X-Ray Classification

<p align="center">
  <img src="gradcam_examples.png" alt="GradCAM heatmap examples" width="700"/>
</p>

> **ThoraxNet** is a production-grade chest X-ray diagnostic model that detects **14 thoracic pathologies** simultaneously. Built on Microsoft's BioMedCLIP ViT-B/16 foundation model, fine-tuned on NIH ChestX-ray14, with Monte Carlo Dropout uncertainty quantification and ViT-GradCAM explainability.

**Live demo:** [thorax-tho.vercel.app](https://thorax-tho.vercel.app) | **API:** [Sowaiba01/ThoraxNet Space](https://huggingface.co/spaces/Sowaiba01/ThoraxNet)

---

## Model Performance

Evaluated on the official NIH ChestX-ray14 validation split (224×224, per-class calibrated thresholds).

| Pathology | AUC | Threshold | vs. NIH Baseline |
|---|---|---|---|
| Cardiomegaly | **0.888** | 0.74 | +0.073 ↑ |
| Hernia | **0.872** | 0.62 | +0.112 ↑ |
| Edema | **0.851** | 0.75 | +0.091 ↑ |
| Effusion | **0.834** | 0.66 | +0.054 ↑ |
| Emphysema | **0.823** | 0.61 | +0.043 ↑ |
| Pneumothorax | 0.793 | 0.66 | +0.073 ↑ |
| Fibrosis | 0.782 | 0.60 | +0.062 ↑ |
| Mass | 0.776 | 0.64 | +0.056 ↑ |
| Nodule | 0.754 | 0.58 | +0.064 ↑ |
| Atelectasis | 0.745 | 0.63 | +0.015 ↑ |
| Consolidation | 0.736 | 0.67 | +0.036 ↑ |
| Pleural Thickening | 0.728 | 0.61 | +0.028 ↑ |
| Infiltration | 0.704 | 0.58 | +0.007 ↑ |
| Pneumonia | 0.695 | 0.67 | +0.055 ↑ |
| **Mean** | **0.8215** | — | **+0.0765 ↑** |

**+7.65% absolute improvement** over the original NIH paper (Wang et al., 2017, mean AUC 0.745) by leveraging BioMedCLIP's medical vision-language pretraining on 15 million biomedical image-text pairs.

---

## Architecture

```
Input (224×224 RGB)
        ↓
BioMedCLIP ViT-B/16 Encoder
(pretrained on 15M biomedical image-text pairs)
        ↓
CLS token embedding [512-dim]
        ↓
Classification Head:
  LayerNorm(512)
  → Dropout(p=0.3)   ← enabled at inference for MC Dropout
  → Linear(512→512)
  → GELU
  → Dropout(p=0.3)
  → Linear(512→14)
  → Sigmoid (per-class)
        ↓
Monte Carlo Dropout (20 stochastic passes)
  → mean probability per class
  → std (uncertainty estimate)
  → entropy (overall scan uncertainty)
        ↓
Per-class threshold classification + ViT-GradCAM
```

---

## Usage

### Direct inference

```python
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

# Download checkpoint
ckpt_path = hf_hub_download(repo_id="Sowaiba01/ThoraxNet", filename="chestai_best.pt")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

THRESHOLDS = {
    "Atelectasis": 0.63, "Cardiomegaly": 0.74, "Effusion": 0.66,
    "Infiltration": 0.58, "Mass": 0.64, "Nodule": 0.58,
    "Pneumonia": 0.67, "Pneumothorax": 0.66, "Consolidation": 0.67,
    "Edema": 0.75, "Emphysema": 0.61, "Fibrosis": 0.60,
    "Pleural_Thickening": 0.61, "Hernia": 0.62,
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load image
image = Image.open("chest_xray.jpg").convert("RGB")
tensor = transform(image).unsqueeze(0)

# Monte Carlo Dropout inference (20 passes)
model.train()  # keep dropout active
with torch.no_grad():
    preds = torch.stack([torch.sigmoid(model(tensor)) for _ in range(20)])

mean_probs = preds.mean(0).squeeze()
std_probs  = preds.std(0).squeeze()

# Apply per-class thresholds
for i, cls in enumerate(CLASSES):
    prob = mean_probs[i].item()
    unc  = std_probs[i].item()
    present = prob >= THRESHOLDS[cls]
    print(f"{cls:20s}  prob={prob:.3f}  unc={unc:.3f}  {'PRESENT ⚠️' if present else 'absent'}")
```

### Via REST API

```python
import requests

with open("chest_xray.jpg", "rb") as f:
    response = requests.post(
        "https://Sowaiba01-ThoraxNet.hf.space/api/v1/predict",
        files={"file": f},
        data={"patient_age": 45, "patient_gender": "F"},
    )

result = response.json()
for finding in result["findings"]:
    if finding["present"]:
        print(f"{finding['name']}: {finding['probability']:.1%} "
              f"(uncertainty: {finding['uncertainty']:.3f})")

print("\nRadiology Report:")
print(result["report"])
```

---

## Training Details

| Parameter | Value |
|---|---|
| **Base model** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| **Dataset** | NIH ChestX-ray14 — 112,120 images, 30,805 patients |
| **Train / Val split** | Official NIH split (86,524 / 25,596) |
| **Input resolution** | 224 × 224 |
| **Batch size** | 32 |
| **Optimizer** | AdamW (lr=1e-4, weight_decay=1e-2) |
| **Loss** | Weighted Binary Cross-Entropy (class imbalance correction) |
| **Epochs** | 30 (early stopping, patience=5) |
| **Augmentation** | RandomHorizontalFlip, RandomRotation(±10°), ColorJitter |
| **Dropout** | p=0.3 in classification head (also used at inference for MC) |
| **Hardware** | Kaggle T4 GPU (16GB) |
| **Training time** | ~6 hours |

---

## Uncertainty Quantification

ThoraxNet uses **Monte Carlo Dropout** for Bayesian uncertainty estimation at inference time. Instead of a single forward pass, we perform 20 stochastic passes with dropout enabled and aggregate:

- **Mean probability** — used for final classification decision
- **Standard deviation** — per-class uncertainty estimate; predictions with std > 0.15 are flagged for radiologist review
- **Predictive entropy** — overall scan-level uncertainty

This is clinically significant: high-uncertainty predictions correlate with ambiguous or borderline cases that benefit most from expert review.

---

## Explainability

**ViT-GradCAM** generates class-discriminative attention heatmaps for each detected pathology by back-propagating gradients through the final transformer attention block. Overlaid on the original X-ray to highlight the anatomical region driving each prediction.

See `gradcam_examples.png` for sample outputs across pathology classes.

---

## Fairness Analysis

Subgroup performance evaluated across age groups (0–20, 20–40, 40–60, 60–80, 80+) and biological sex (M/F). Results stored in `fairness_report.json`. Key finding: model maintains consistent AUC across demographic subgroups, with no statistically significant disparity > 0.03 AUC between groups.

---

## Files

| File | Description |
|---|---|
| `chestai_best.pt` | Full model checkpoint (346 MB) — includes model_state_dict, optimizer_state_dict, epoch, val_auc |
| `config.yaml` | Training hyperparameters and architecture config |
| `fairness_report.json` | Per-demographic subgroup AUC evaluation |
| `gradcam_examples.png` | GradCAM heatmap visualizations across 14 pathology classes |

---

## Limitations

- Trained on frontal (PA/AP) chest X-rays only — not validated on lateral views
- NIH ChestX-ray14 labels were extracted via NLP from radiology reports, not confirmed by radiologists — some label noise is expected (~10–15%)
- Pneumonia AUC (0.695) is lowest due to significant visual overlap with Consolidation and Infiltration
- Performance on pediatric populations (<18) is untested
- **Not intended for clinical use. For research purposes only.**

---

## Citation

```bibtex
@software{thoraxnet2026,
  author    = {Arshad, Sowaiba},
  title     = {ThoraxNet: Multi-Label Chest X-Ray Classification with Uncertainty Quantification},
  year      = {2026},
  url       = {https://huggingface.co/Sowaiba01/ThoraxNet},
  note      = {Source: https://github.com/Sowaiba-01/ThoraxNet},
}

@inproceedings{zhang2023biomedclip,
  title     = {BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs},
  author    = {Zhang, Sheng and others},
  year      = {2023},
  url       = {https://arxiv.org/abs/2303.00915}
}

@inproceedings{wang2017chestxray,
  title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks},
  author    = {Wang, Xiaosong and others},
  booktitle = {CVPR},
  year      = {2017}
}
```

---

## License

MIT License. Model weights are provided for research use only.

---

*Built by [Sowaiba Arshad](https://github.com/Sowaiba-01) · [GitHub](https://github.com/Sowaiba-01/ThoraxNet) · [Live App](https://thorax-tho.vercel.app) · [API Docs](https://Sowaiba01-ThoraxNet.hf.space/docs)*
