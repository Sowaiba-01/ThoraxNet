"""
Fairness audit: per-demographic-subgroup AUC analysis.

Why this matters:
  ML models trained on population data frequently show performance disparities
  across age and sex subgroups. In radiology AI, a model that works well on
  average but fails for elderly women (a high-risk group for several conditions)
  could cause real harm.

  This module computes AUC-ROC per class for each demographic subgroup and
  flags significant disparities (Δ AUC > 0.05) for reporting in the model card.

Usage:
    from fairness.audit import FairnessAuditor
    auditor = FairnessAuditor(cfg)
    report = auditor.run(targets, probs, metadata_list)
    auditor.save_report(report, "fairness_report.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data.dataset import CLASSES


class FairnessAuditor:
    """
    Evaluate model performance across demographic subgroups.

    Args:
        cfg: Full config dict (uses cfg['fairness'] section).
    """

    def __init__(self, cfg: dict) -> None:
        fair_cfg = cfg.get("fairness", {})
        self.age_bins   = fair_cfg.get("age_bins",   [0, 40, 60, 120])
        self.age_labels = fair_cfg.get("age_labels", ["<40", "40-60", ">60"])

    # ------------------------------------------------------------------
    # Main audit entry point
    # ------------------------------------------------------------------

    def run(
        self,
        targets: np.ndarray,
        probs: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Compute per-subgroup AUC for all 14 pathology classes.

        Args:
            targets: (N, 14) binary ground truth.
            probs: (N, 14) predicted probabilities.
            metadata: List of dicts with keys 'age' (float) and 'gender' (str 'M'/'F').

        Returns:
            Nested dict: {subgroup_key: {class_name: auc_value, ...}, ...}
        """
        ages    = np.array([m["age"] for m in metadata])
        genders = np.array([m["gender"] for m in metadata])

        age_groups = pd.cut(
            ages,
            bins=self.age_bins,
            labels=self.age_labels,
            right=False,
        ).astype(str)

        report: dict[str, Any] = {
            "overall": self._compute_group_aucs(targets, probs),
        }

        # Gender subgroups
        for g in ["M", "F"]:
            mask = genders == g
            if mask.sum() > 10:
                label = "male" if g == "M" else "female"
                report[f"gender/{label}"] = self._compute_group_aucs(
                    targets[mask], probs[mask]
                )

        # Age subgroups
        for ag in self.age_labels:
            mask = age_groups == ag
            if mask.sum() > 10:
                report[f"age/{ag}"] = self._compute_group_aucs(
                    targets[mask], probs[mask]
                )

        report["disparities"] = self._detect_disparities(report)
        report["summary"] = self._summarize(report)
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_group_aucs(
        self, targets: np.ndarray, probs: np.ndarray
    ) -> dict[str, float]:
        """Return {class_name: auc} for a subgroup."""
        aucs = {}
        for i, cls in enumerate(CLASSES):
            y_true = targets[:, i]
            y_prob = probs[:, i]
            # Need at least one positive and one negative example.
            if y_true.sum() >= 1 and (1 - y_true).sum() >= 1:
                try:
                    aucs[cls] = float(roc_auc_score(y_true, y_prob))
                except Exception:
                    aucs[cls] = float("nan")
            else:
                aucs[cls] = float("nan")
        valid = [v for v in aucs.values() if not np.isnan(v)]
        aucs["macro_auc"] = float(np.mean(valid)) if valid else float("nan")
        return aucs

    def _detect_disparities(
        self, report: dict[str, Any], threshold: float = 0.05
    ) -> list[dict]:
        """
        Flag (class, subgroup_a, subgroup_b) triples where |AUC_a - AUC_b| > threshold.
        """
        flagged = []
        overall = report.get("overall", {})
        for key, group_aucs in report.items():
            if key in ("overall", "disparities", "summary"):
                continue
            for cls in CLASSES:
                overall_auc = overall.get(cls, float("nan"))
                group_auc   = group_aucs.get(cls, float("nan"))
                if np.isnan(overall_auc) or np.isnan(group_auc):
                    continue
                delta = overall_auc - group_auc
                if abs(delta) > threshold:
                    flagged.append({
                        "class": cls,
                        "subgroup": key,
                        "overall_auc": round(overall_auc, 4),
                        "subgroup_auc": round(group_auc, 4),
                        "delta": round(delta, 4),
                        "direction": "underperforms" if delta > 0 else "overperforms",
                    })
        return sorted(flagged, key=lambda x: abs(x["delta"]), reverse=True)

    def _summarize(self, report: dict[str, Any]) -> dict[str, Any]:
        disparities = report.get("disparities", [])
        return {
            "total_disparities_flagged": len(disparities),
            "worst_disparity": disparities[0] if disparities else None,
            "recommendation": (
                "Model shows significant performance disparities. "
                "Consider reweighting, targeted augmentation, or per-group calibration "
                "before clinical deployment."
            ) if disparities else "No significant disparities detected.",
        }

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_report(self, report: dict[str, Any], path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[FairnessAuditor] Report saved to {path}")

    def to_dataframe(self, report: dict[str, Any]) -> pd.DataFrame:
        """Wide-format DataFrame: rows = classes, cols = subgroups."""
        rows = {}
        for group_key, aucs in report.items():
            if group_key in ("disparities", "summary"):
                continue
            rows[group_key] = {cls: aucs.get(cls, float("nan")) for cls in CLASSES}
        return pd.DataFrame(rows, index=CLASSES).round(4)
