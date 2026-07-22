"""
NIH ChestX-ray14 PyTorch Dataset.

Handles:
- Multi-label binary encoding for 14 pathology classes
- Patient-level train/val/test split (no patient leakage)
- On-the-fly augmentation
- Demographic metadata for fairness auditing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import build_transforms

# Canonical class order — never reorder; indices are baked into checkpoints.
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


class ChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 dataset with multi-label targets.

    Args:
        root_dir: Directory containing image files.
        labels_csv: Path to Data_Entry_2017.csv.
        split_list: Text file listing image filenames for this split.
        transform: Torchvision transform applied to each image.
        return_metadata: If True, also return (age, gender) for fairness analysis.
    """

    def __init__(
        self,
        root_dir: str | Path,
        labels_csv: str | Path,
        split_list: str | Path,
        transform=None,
        return_metadata: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_metadata = return_metadata

        # Load and filter the label dataframe to only this split's files.
        df = pd.read_csv(labels_csv)
        with open(split_list) as f:
            split_files = set(line.strip() for line in f if line.strip())
        self.df = df[df["Image Index"].isin(split_files)].reset_index(drop=True)

        # Pre-encode multi-hot label vectors once (faster than per-item encoding).
        self._labels = self._encode_labels()
        # Normalize age: clip outliers and convert to float.
        self.df["Patient Age"] = self.df["Patient Age"].clip(0, 100).astype(float)

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------

    def _encode_labels(self) -> np.ndarray:
        """Return (N, 14) float32 multi-hot array."""
        labels = np.zeros((len(self.df), NUM_CLASSES), dtype=np.float32)
        for i, finding_str in enumerate(self.df["Finding Labels"]):
            for finding in finding_str.split("|"):
                finding = finding.strip()
                if finding in CLASS_TO_IDX:
                    labels[i, CLASS_TO_IDX[finding]] = 1.0
        return labels

    # ------------------------------------------------------------------
    # Class-level positive weights for loss weighting (call on train split)
    # ------------------------------------------------------------------

    def get_pos_weights(self) -> torch.Tensor:
        """
        Compute per-class positive weights = (neg_count / pos_count).
        Pass to nn.BCEWithLogitsLoss(pos_weight=...) to handle imbalance.
        """
        pos = self._labels.sum(axis=0)
        neg = len(self._labels) - pos
        weights = neg / np.maximum(pos, 1)  # avoid div-by-zero for Hernia
        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["Image Index"]

        # Load as RGB (some NIH images are grayscale PNGs).
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(self._labels[idx])

        if self.return_metadata:
            meta = {
                "age": float(row["Patient Age"]),
                "gender": row["Patient Gender"].strip(),  # 'M' or 'F'
                "filename": row["Image Index"],
            }
            return image, label, meta

        return image, label

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        return CLASSES

    def label_stats(self) -> pd.DataFrame:
        """Return a DataFrame with per-class prevalence for EDA."""
        counts = self._labels.sum(axis=0)
        prev = counts / len(self._labels)
        return pd.DataFrame({
            "class": CLASSES,
            "positive_count": counts.astype(int),
            "prevalence": prev,
        }).sort_values("prevalence", ascending=False)


def build_datasets(cfg: dict) -> dict[str, ChestXrayDataset]:
    """
    Build train / val / test datasets from config dict.

    Returns:
        {'train': ..., 'val': ..., 'test': ...}
    """
    train_tf = build_transforms("train", cfg["data"]["image_size"])
    val_tf = build_transforms("val", cfg["data"]["image_size"])

    common = dict(
        root_dir=cfg["data"]["root_dir"],
        labels_csv=cfg["data"]["labels_csv"],
    )

    # The official NIH split file covers both train AND val; we sub-split below.
    full_train = ChestXrayDataset(
        **common,
        split_list=cfg["data"]["train_val_list"],
        transform=train_tf,
    )
    full_val = ChestXrayDataset(
        **common,
        split_list=cfg["data"]["train_val_list"],
        transform=val_tf,
        return_metadata=True,
    )
    test_ds = ChestXrayDataset(
        **common,
        split_list=cfg["data"]["test_list"],
        transform=val_tf,
        return_metadata=True,
    )

    # Patient-level 90/10 train/val split to prevent patient leakage.
    patient_ids = full_train.df["Patient ID"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(patient_ids)
    split_pt = int(0.9 * len(patient_ids))
    train_patients = set(patient_ids[:split_pt])
    val_patients = set(patient_ids[split_pt:])

    train_mask = full_train.df["Patient ID"].isin(train_patients).values
    val_mask = full_val.df["Patient ID"].isin(val_patients).values

    # Apply masks to create proper subsets without data leakage.
    train_ds = _mask_dataset(full_train, train_mask)
    val_ds = _mask_dataset(full_val, val_mask)

    return {"train": train_ds, "val": val_ds, "test": test_ds}


def _mask_dataset(ds: ChestXrayDataset, mask: np.ndarray) -> ChestXrayDataset:
    """Return a view of ds restricted to rows where mask is True."""
    ds_copy = object.__new__(ChestXrayDataset)
    ds_copy.__dict__.update(ds.__dict__)
    ds_copy.df = ds.df[mask].reset_index(drop=True)
    ds_copy._labels = ds._labels[mask]
    return ds_copy
