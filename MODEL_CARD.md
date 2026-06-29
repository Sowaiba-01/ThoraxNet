---
language: en
license: mit
tags:
  - medical-imaging
  - chest-xray
  - multi-label-classification
  - uncertainty-quantification
  - pytorch
  - vision-transformer
datasets:
  - NIH ChestX-ray14
metrics:
  - roc_auc
---

# ChestAI Model Card

## Model Summary

BioMedCLIP ViT-B/16 fine-tuned for multi-label classification of 14 thoracic
pathologies on the NIH ChestX-ray14 dataset, augmented with Monte Carlo Dropout
for calibrated uncertainty estimation.

## Intended Use

**Intended users:** Researchers, medical AI developers, radiologist decision support tools.

**Intended use cases:** Screening assistance, research benchmarking, demonstration of uncertainty-aware medical AI.

**Out-of-scope uses:** Clinical diagnosis, replacing radiologist interpretation, use in any real patient care setting.

## Training Data

**Dataset:** NIH ChestX-ray14 (Wang et al., 2017)
- 112,120 frontal-view chest X-ray images
- 30,805 unique patients
- 14 pathology labels (multi-label, not mutually exclusive)
- Patient demographics: age (mean ≈ 46, range 1–95), gender (56% M, 44% F)

**Split:** Patient-level 90/10 train/val split (no patient leakage) + official NIH test set.

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | BioMedCLIP ViT-B/16 (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| Head | LayerNorm → Dropout(0.3) → Linear(512→256) → GELU → Dropout(0.3) → Linear(256→14) |
| Parameters | ~86M total |
| Input | 224×224 RGB (ImageNet normalization) |
| Output | 14 logits → sigmoid → probabilities |

## Training Procedure

| Setting | Value |
|---|---|
| Loss | Weighted Focal BCE (γ=2.0, α=0.25, pos_weight_scale=5×) |
| Optimizer | AdamW (backbone LR = 1e-5, head LR = 1e-4) |
| Scheduler | Linear warmup (3 ep) + Cosine annealing |
| Epochs | 30 (early stopping patience=5) |
| AMP | Yes (float16) |
| Backbone unfreezing | Last 6 blocks after epoch 2 |
| Hardware | Kaggle P100 16GB |

## Evaluation

### Quantitative Results (Test Set)

> Insert your actual numbers after training.

| Class | AUC | Sensitivity@0.5 | Specificity@0.5 |
|---|---|---|---|
| **Macro Average** | TBD | TBD | TBD |

### Uncertainty Calibration

MC Dropout (20 passes) std deviation is used as a proxy for epistemic uncertainty.
Findings with std > 0.15 are flagged as uncertain in the UI and report.

## Fairness Analysis

Evaluated across demographic subgroups. See `fairness_report.json` for full results.

**Subgroups evaluated:**
- Gender: Male, Female
- Age: <40, 40–60, >60

Disparities (ΔAUC > 0.05) are flagged. See model card addendum after training.

## Limitations

1. **Training data bias:** NIH ChestX-ray14 was collected at a single US institution (NIH Clinical Center). Performance may degrade on X-rays from different imaging equipment, patient populations, or geographic regions.

2. **Label noise:** NIH labels were extracted via NLP from radiology reports, not manually verified. Some label noise is expected, particularly for subtle findings.

3. **No lateral views:** Model uses only frontal (PA/AP) views. Lateral view fusion would improve accuracy for some pathologies.

4. **Threshold sensitivity:** The default 0.5 threshold may not be optimal for all classes. In clinical use, sensitivity-specificity tradeoffs should be tuned per deployment context.

5. **Not FDA cleared.** This model has not undergone regulatory review and must not be used in clinical settings.

## Ethical Considerations

- Model outputs should always be reviewed by a qualified radiologist before informing any clinical decision.
- Uncertainty estimates help identify cases requiring human review but do not guarantee correctness.
- Demographic fairness is evaluated but not guaranteed — subgroup performance should be monitored in any deployment.

## Citation

If you use this model in research, please cite:
```
@misc{chestai2024,
  title={ChestAI: Uncertainty-Aware Multi-Label Chest X-Ray Diagnostics},
  author={YOUR NAME},
  year={2024},
  url={https://github.com/YOUR_USERNAME/chestai}
}
```

And the original dataset:
```
@inproceedings{wang2017chestx,
  title={ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  booktitle={CVPR},
  year={2017}
}
```
