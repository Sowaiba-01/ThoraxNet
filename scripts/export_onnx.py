#!/usr/bin/env python3
"""
Export ChestAI to TorchScript and ONNX, then produce an INT8 quantized ONNX.

Usage:
    python scripts/export_onnx.py --checkpoint chestai_best.pt --outdir export/

    # Skip quantization (e.g. if onnxruntime isn't installed)
    python scripts/export_onnx.py --checkpoint chestai_best.pt --no-quantize

Outputs (in --outdir):
    thoraxnet_scripted.pt   TorchScript, removes Python dispatch overhead
    thoraxnet.onnx          FP32 ONNX, dynamic batch axis
    thoraxnet_int8.onnx     Dynamic INT8 quantized (weights int8, activations fp32)

Why dynamic quantization and not static:
    Static (calibrated) quantization needs a representative calibration set and
    quantizes activations too, which for a ViT means quantizing the attention
    softmax path. That is where transformer accuracy degrades most. Dynamic
    quantization only touches Linear weights — for ViT-B/16 that is still the
    large majority of the parameter count, so you keep most of the size and
    bandwidth win at a much smaller accuracy risk.

    Run scripts/eval_quantization.py afterwards to measure the actual delta
    rather than assuming it is negligible.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Repo root on path so `models` etc. import cleanly when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.classifier import ChestAIClassifier  # noqa: E402


def load_model(checkpoint_path: str, device: torch.device) -> ChestAIClassifier:
    print(f"[export] loading {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ChestAIClassifier()
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def export_torchscript(model: torch.nn.Module, dummy: torch.Tensor, out: Path) -> None:
    print("[export] TorchScript ...")
    # trace, not script: the BioMedCLIP backbone contains constructs that
    # torch.jit.script rejects, and inference has no data-dependent control flow.
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)
    traced = torch.jit.freeze(traced)
    traced.save(str(out))
    print(f"[export]   → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def export_onnx(model: torch.nn.Module, dummy: torch.Tensor, out: Path, opset: int) -> None:
    print(f"[export] ONNX (opset {opset}) ...")
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        # Dynamic batch is essential: batched MC dropout sends T*B through
        # in one pass, so batch size varies at runtime.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[export]   → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def quantize(src: Path, dst: Path) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("[export] onnxruntime not installed — skipping INT8. "
              "pip install onnxruntime")
        return False

    print("[export] INT8 dynamic quantization ...")
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )
    ratio = src.stat().st_size / dst.stat().st_size
    print(f"[export]   → {dst}  ({dst.stat().st_size / 1e6:.1f} MB, {ratio:.2f}x smaller)")
    return True


def verify(onnx_path: Path, dummy: torch.Tensor, torch_out: torch.Tensor) -> None:
    """Confirm the exported graph reproduces the PyTorch output."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed — skipping numerical check")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.cpu().numpy()})[0]
    max_diff = float(np.abs(onnx_out - torch_out.cpu().numpy()).max())
    status = "OK" if max_diff < 1e-3 else "MISMATCH"
    print(f"[verify] {onnx_path.name}: max|torch - onnx| = {max_diff:.2e}  [{status}]")


def bench(fn, n: int = 20) -> float:
    """Mean ms per call, after 3 warmup calls."""
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", default="export")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=20,
                    help="Dummy batch size — use your MC sample count")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--no-torchscript", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # export on CPU for portability
    model = load_model(args.checkpoint, device)
    dummy = torch.randn(args.batch, 3, 224, 224, device=device)

    with torch.no_grad():
        torch_out = model(dummy)

    if not args.no_torchscript:
        export_torchscript(model, dummy, outdir / "thoraxnet_scripted.pt")

    onnx_path = outdir / "thoraxnet.onnx"
    export_onnx(model, dummy, onnx_path, args.opset)
    verify(onnx_path, dummy, torch_out)

    if not args.no_quantize:
        int8_path = outdir / "thoraxnet_int8.onnx"
        if quantize(onnx_path, int8_path):
            verify(int8_path, dummy, torch_out)

    print("\n[export] done. Next: "
          "python scripts/eval_quantization.py --fp32 export/thoraxnet.onnx "
          "--int8 export/thoraxnet_int8.onnx")


if __name__ == "__main__":
    main()
