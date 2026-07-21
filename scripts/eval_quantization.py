#!/usr/bin/env python3
"""
Measure the accuracy cost of INT8 quantization, per class.

Compares FP32 and INT8 ONNX exports on the validation split and emits a
per-class AUC delta table. Publish the result whichever way it comes out —
a small regression honestly reported is more informative than a claim of
"no accuracy loss" with no evidence.

Usage:
    python scripts/eval_quantization.py \
        --fp32 export/thoraxnet.onnx \
        --int8 export/thoraxnet_int8.onnx \
        --data-root /path/to/images \
        --labels-csv /path/to/Data_Entry_2017.csv \
        --split-list splits/val_list.txt

    # Also time both models
    python scripts/eval_quantization.py ... --bench

Output: Markdown table on stdout, optional --json for CI tracking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import CLASSES, ChestXrayDataset  # noqa: E402
from data.transforms import build_transforms  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime is required:  pip install onnxruntime")

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    sys.exit("scikit-learn is required:  pip install scikit-learn")

from torch.utils.data import DataLoader  # noqa: E402


def run_inference(session: ort.InferenceSession, loader: DataLoader, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels), both (N, 14)."""
    input_name = session.get_inputs()[0].name
    all_probs, all_labels = [], []
    seen = 0

    for batch in loader:
        images, labels = batch[0], batch[1]
        logits = session.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        all_probs.append(probs)
        all_labels.append(labels.numpy())

        seen += images.shape[0]
        if limit and seen >= limit:
            break
        if seen % 512 == 0:
            print(f"    {seen} images ...", flush=True)

    return np.concatenate(all_probs), np.concatenate(all_labels)


def per_class_auc(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    out = {}
    for i, cls in enumerate(CLASSES):
        y = labels[:, i]
        # roc_auc_score is undefined if a class has only one label value present
        if len(np.unique(y)) < 2:
            out[cls] = float("nan")
        else:
            out[cls] = float(roc_auc_score(y, probs[:, i]))
    return out


def benchmark_session(session: ort.InferenceSession, batch: int = 20, n: int = 20) -> float:
    """Mean ms per forward pass at the given batch size."""
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(batch, 3, 224, 224).astype(np.float32)
    for _ in range(3):
        session.run(None, {input_name: dummy})
    t0 = time.perf_counter()
    for _ in range(n):
        session.run(None, {input_name: dummy})
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--split-list", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N images (for a quick check)")
    ap.add_argument("--bench", action="store_true", help="Also time both models")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dataset = ChestXrayDataset(
        root_dir=args.data_root,
        labels_csv=args.labels_csv,
        split_list=args.split_list,
        transform=build_transforms("val", image_size=224),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"[eval] {len(dataset)} validation images")

    providers = ["CPUExecutionProvider"]
    print("[eval] FP32 ...")
    fp32_probs, labels = run_inference(ort.InferenceSession(args.fp32, providers=providers), loader, args.limit)
    print("[eval] INT8 ...")
    int8_probs, _ = run_inference(ort.InferenceSession(args.int8, providers=providers), loader, args.limit)

    fp32_auc = per_class_auc(fp32_probs, labels)
    int8_auc = per_class_auc(int8_probs, labels)

    rows = []
    for cls in CLASSES:
        f, i = fp32_auc[cls], int8_auc[cls]
        rows.append({"class": cls, "fp32": f, "int8": i, "delta": i - f})

    valid = [r for r in rows if not np.isnan(r["fp32"])]
    mean_fp32 = float(np.mean([r["fp32"] for r in valid]))
    mean_int8 = float(np.mean([r["int8"] for r in valid]))

    print("\n| Pathology | FP32 AUC | INT8 AUC | Δ |")
    print("|---|---|---|---|")
    for r in rows:
        sign = "+" if r["delta"] >= 0 else ""
        print(f"| {r['class'].replace('_', ' ')} | {r['fp32']:.4f} | {r['int8']:.4f} | {sign}{r['delta']:.4f} |")
    d = mean_int8 - mean_fp32
    print(f"| **Mean** | **{mean_fp32:.4f}** | **{mean_int8:.4f}** | **{'+' if d >= 0 else ''}{d:.4f}** |")

    worst = min(rows, key=lambda r: r["delta"] if not np.isnan(r["delta"]) else 0)
    print(f"\nLargest single-class regression: {worst['class']} ({worst['delta']:+.4f})")

    result = {
        "mean_fp32": mean_fp32,
        "mean_int8": mean_int8,
        "mean_delta": d,
        "per_class": rows,
        "n_images": int(labels.shape[0]),
    }

    if args.bench:
        print("\n[bench] timing both models (batch=20, CPU) ...")
        t_fp32 = benchmark_session(ort.InferenceSession(args.fp32, providers=providers))
        t_int8 = benchmark_session(ort.InferenceSession(args.int8, providers=providers))
        print("| Model | ms / batch-20 forward | Speedup |")
        print("|---|---|---|")
        print(f"| FP32 | {t_fp32:.1f} | 1.00x |")
        print(f"| INT8 | {t_int8:.1f} | {t_fp32 / t_int8:.2f}x |")
        result["bench"] = {"fp32_ms": t_fp32, "int8_ms": t_int8, "speedup": t_fp32 / t_int8}

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nRaw results → {args.json}")


if __name__ == "__main__":
    main()
