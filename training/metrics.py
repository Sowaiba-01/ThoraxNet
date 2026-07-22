
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
)

from data.dataset import CLASSES


def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute all clinical metrics.

    Args:
        targets: (N, C) binary ground truth.
        probs: (N, C) predicted probabilities in [0, 1].
        threshold: Decision threshold for binary classification metrics.
        prefix: Optional string prefix for metric keys (e.g. 'val/').

    Returns:
        Flat dict of metric_name → float.
    """
    n_classes = targets.shape[1]
    results: dict[str, float] = {}

    aucs, aps, sens_list, spec_list, ppv_list, npv_list, f1_list = (
        [], [], [], [], [], [], []
    )

    for i, cls in enumerate(CLASSES[:n_classes]):
        y_true = targets[:, i]
        y_prob = probs[:, i]
        y_pred = (y_prob >= threshold).astype(int)

        # Skip classes with no positives in this split (e.g., Hernia in small batches).
        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            continue

        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)   # recall
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)           # precision
        npv = tn / max(tn + fn, 1)
        f1  = f1_score(y_true, y_pred, zero_division=0)

        results[f"{prefix}{cls}/auc"]         = auc
        results[f"{prefix}{cls}/ap"]          = ap
        results[f"{prefix}{cls}/sensitivity"] = sensitivity
        results[f"{prefix}{cls}/specificity"] = specificity
        results[f"{prefix}{cls}/ppv"]         = ppv
        results[f"{prefix}{cls}/npv"]         = npv
        results[f"{prefix}{cls}/f1"]          = f1

        aucs.append(auc)
        aps.append(ap)
        sens_list.append(sensitivity)
        spec_list.append(specificity)
        ppv_list.append(ppv)
        npv_list.append(npv)
        f1_list.append(f1)

    # Macro averages (headline metrics for W&B and README).
    results[f"{prefix}macro/auc"]         = float(np.mean(aucs))
    results[f"{prefix}macro/ap"]          = float(np.mean(aps))
    results[f"{prefix}macro/sensitivity"] = float(np.mean(sens_list))
    results[f"{prefix}macro/specificity"] = float(np.mean(spec_list))
    results[f"{prefix}macro/ppv"]         = float(np.mean(ppv_list))
    results[f"{prefix}macro/npv"]         = float(np.mean(npv_list))
    results[f"{prefix}macro/f1"]          = float(np.mean(f1_list))

    return results


def metrics_dataframe(metrics: dict[str, float]) -> pd.DataFrame:
    """
    Convert flat metrics dict into a per-class DataFrame for pretty printing.

    Returns:
        DataFrame with columns [Class, AUC, AP, Sensitivity, Specificity, F1].
    """
    rows = []
    for cls in CLASSES:
        # Metric keys may carry a split prefix ("val/", "test/") or none at all.
        # Recover it from the first matching key so lookups below work either way.
        p = ""
        for k in metrics:
            if k.endswith(f"{cls}/auc"):
                p = k[: k.index(cls)]
                break
        row = {
            "Class":       cls,
            "AUC":         metrics.get(f"{p}{cls}/auc",         float("nan")),
            "AP":          metrics.get(f"{p}{cls}/ap",          float("nan")),
            "Sensitivity": metrics.get(f"{p}{cls}/sensitivity", float("nan")),
            "Specificity": metrics.get(f"{p}{cls}/specificity", float("nan")),
            "F1":          metrics.get(f"{p}{cls}/f1",          float("nan")),
        }
        rows.append(row)
    return pd.DataFrame(rows)
