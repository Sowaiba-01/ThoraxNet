# Changelog

All notable changes to ChestAI / ThoraxNet are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `scripts/export_onnx.py` — TorchScript trace + ONNX export with a dynamic
  batch axis, and dynamic INT8 quantization via ONNX Runtime.
- `scripts/eval_quantization.py` — per-class FP32-vs-INT8 AUC delta table, so
  the accuracy cost of quantization is measured rather than assumed.

### Notes
- INT8 weights are exported but **not yet serving** production traffic. The
  accuracy delta table must be published before the quantized graph is
  promoted.

---

## [1.1.0] — 2026-07-22

Performance release. No change to model weights or per-class AUC.

Measured on HF Spaces free tier (CPU, 2 vCPU), 30 requests, same image,
0 failures.

Single-request (concurrency 1):

| Version | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|
| v1.0.0 | 6,387 ms | 7,525 ms | 8,026 ms | 0.15 req/s |
| v1.1.0 | 3,327 ms | 3,862 ms | 4,768 ms | 0.30 req/s |

Every percentile improved ~2× (p50 1.9×, throughput 2.0×) with no change to
model weights or accuracy. Server-side v1.1.0 stage breakdown: mc_dropout
~2,870 ms · preprocess 16 ms · report 92 ms (async) · gradcam 6 ms (cache hit).
The win is entirely in how the model is executed — batching the 20 MC Dropout
passes into a single forward pass, moving the Groq call off the critical path,
and caching GradCAM.

### Changed
- **Batched Monte Carlo Dropout** (`models/uncertainty.py`). MC Dropout ran
  `n_samples` sequential forward passes, each with batch size 1 — 20 launches
  of a ViT-B/16 that the GPU was almost entirely idle through. The input is
  now tiled along the batch dimension and evaluated in a single pass; dropout
  masks are sampled per batch element, so the T tiled copies are exactly the T
  independent stochastic samples the estimator requires. Chunked at
  `max_chunk=32` samples to bound memory. On the CPU host this is the primary
  driver of the ~2× end-to-end improvement (see the measured table above).
- **Groq report generation moved off the event loop** (`api/inference.py`).
  `RadiologyReportGenerator.generate()` is a blocking HTTP call that was being
  awaited directly inside an async handler, serialising every concurrent
  request behind it. Now dispatched via `asyncio.to_thread`.
- **Report and GradCAM are now optional per request.** `POST /api/v1/predict`
  accepts `generate_report` and `generate_gradcam` form flags (both default
  `true`). Clients that only need probabilities can skip both.
- Default `mc_samples` is now configurable via the `MC_SAMPLES` env var.

### Added
- **GradCAM LRU cache** keyed on `(sha1(image_bytes), class_idx)`, 128 entries.
  Each heatmap needs its own full backward pass, so a scan with four positive
  findings paid for four backward passes on every retry of the same study.
- **Per-stage latency breakdown** on the prediction response
  (`stage_timings_ms`: `preprocess`, `mc_dropout`, `gradcam`, `report`).
  Profiling in production beats guessing which stage regressed.
- `gradcam_session_id` on the prediction response — previously the client had
  no way to learn the id needed to fetch overlays from
  `/api/v1/gradcam/{session_id}/{class_name}`.
- `scripts/benchmark.py` — p50/p95/p99 and throughput at configurable
  concurrency, with warmup discard and optional cost-per-1k-inferences.
- `requirements-ci.txt` — test-only dependency set.
- Test suite expanded from 8 to 58 tests: batched-vs-sequential MC
  equivalence, chunk-boundary handling, LRU eviction and recency, GradCAM
  overlay-store regression, threshold coverage, and API schema contracts.
- `mc_predict` now raises a descriptive `RuntimeError` if the model fails to
  preserve its batch dimension, rather than failing inside `view()` with an
  opaque size error.

### Fixed
- **Radiology reports were silently failing in production.** Groq
  decommissioned `llama3-70b-8192`; every report request returned HTTP 400
  `model_decommissioned` and fell back to the template, so users got no
  LLM-generated reports. Surfaced in the deploy logs during benchmarking.
  Default model is now `llama-3.3-70b-versatile`, overridable via the
  `GROQ_MODEL` env var so the next deprecation is a config change.
- **Startup crash from a checkpoint/architecture mismatch.** The published
  checkpoint's head is `Linear(512 → 512 → 14)` but `classifier.py` defaults
  to a 256-wide intermediate layer. `InferencePipeline.load()` reads the true
  width from the checkpoint and rebuilds `model.head` before loading weights
  (`weights_only=False`, `strict=False`). A regression test
  (`test_head_can_be_rebuilt_to_match_a_512wide_checkpoint`) reproduces the
  exact size-mismatch and pins the fix.
- **GradCAM retrieval was completely broken.** `api/routes/predict.py` read
  `pipeline.gradcam._last_overlays`, an attribute that was never assigned —
  the pipeline built overlays into a local variable and discarded them at the
  end of `predict()`. Every `GET /api/v1/gradcam/{session_id}/{class}` request
  therefore 404'd or raised `AttributeError`. The frontend silently rendered
  no heatmap and no error was logged, so this shipped unnoticed.
  `ViTGradCAM.generate_overlays()` now records overlays on the instance, and
  `tests/test_gradcam_cache.py::test_last_overlays_is_populated` pins it.
- `PredictionResponse.report` was typed non-optional, which would have 500'd
  the new `generate_report=false` path.
- MC Dropout `std` with `n_samples=1` returned `NaN` (Bessel correction on a
  single sample). Now uses `unbiased=False` when `T == 1`.
- Session-store eviction used `if` rather than `while`, so it could only ever
  evict one entry per request.
- **`groq` was a hard import in `report_generation/generator.py`**, which made
  the entire inference pipeline unimportable without it — despite the code
  already having a template fallback for exactly that case. Now imported
  behind a guard with a `GROQ_AVAILABLE` flag, so the service degrades as
  designed instead of failing at import time.
- **The test backbone stub was returning a hard-coded `torch.zeros(2, 512)`**
  regardless of input. Two consequences: the batch dimension was wrong for any
  input that wasn't batch-2, and all-zero features propagate as zeros through
  LayerNorm and zero-initialised Linear layers — so dropout produced no
  variance and `assert std.mean() > 0` was passing on a vacuous truth. Both
  affected tests were already failing before the v1.1.0 work; the stub now
  returns fixed non-zero features shaped to the input batch, making dropout
  the only source of variance.

### CI
- Rewrote the workflow. It installed the full `requirements.txt` — including
  CUDA-enabled torch (~2.5 GB), wandb and uvicorn — on a GPU-less runner,
  which is what was timing out the job. Now installs torch from the CPU wheel
  index (~200 MB) plus `requirements-ci.txt`.
- Added a frontend job: TypeScript type check and production Next.js build.
- Added `concurrency.cancel-in-progress`, a 20-minute job timeout, coverage
  reporting, and a test-count summary written to the run summary.

---

## [1.0.0] — 2026-06-15

Initial release.

### Added
- BioMedCLIP ViT-B/16 fine-tuned on NIH ChestX-ray14 (112,120 images,
  30,805 patients), 14-class multi-label output. Mean AUC **0.8215** on the
  official validation split, versus 0.745 for the NIH baseline.
- Monte Carlo Dropout uncertainty quantification (20 passes) reporting mean
  probability, per-class standard deviation, and predictive entropy.
- ViT-GradCAM explainability hooked into the final transformer block.
- Per-class calibrated decision thresholds (0.58–0.75), tuned on validation
  rather than a shared 0.5 cutoff.
- Groq `llama-3.3-70b-versatile` narrative radiology report generation, with
  a deterministic template fallback when the API key is absent or the call
  fails.
- Demographic fairness audit across age bands and sex (`fairness/audit.py`).
- FastAPI serving layer with a singleton inference pipeline, Docker image, and
  HuggingFace Spaces deployment.
- Next.js 14 PWA frontend with Google OAuth, deployed on Vercel.

[Unreleased]: https://github.com/Sowaiba-01/ThoraxNet/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Sowaiba-01/ThoraxNet/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Sowaiba-01/ThoraxNet/releases/tag/v1.0.0
