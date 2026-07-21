# Optimization Runbook

Everything in this repo is now written. This file is the order to **run** it in,
and — importantly — which numbers you still have to replace with real
measurements before publishing anything.

---

## ⚠️ Read this first: placeholder numbers

The code changes are real and tested. The **latency numbers are not measured
yet.** I wrote plausible values so the tables render; you must replace them
with your own benchmark output.

Placeholders live in exactly three places:

| File | What to replace |
|---|---|
| `README.md` → "Performance" | Both tables (stage breakdown + concurrency) |
| `README.md` → badge | `p50%20latency-183ms` |
| `CHANGELOG.md` → `[1.1.0]` | The "4.8 s → 183 ms" claim and stage timings |

Do not push those numbers as-is. A recruiter who asks "how did you measure
p99?" and gets a vague answer is worse off than one who never saw the table.
Steps 1–4 below produce the real ones.

---

## Step 0 — Baseline BEFORE you deploy the new code

You need the "before" column, and once you deploy you can't get it back.

```bash
cd C:\Users\Lenovo\Desktop\chestai
pip install aiohttp

# Grab any chest X-ray as a test fixture
mkdir tests\fixtures
# put a PNG at tests/fixtures/sample_cxr.png

# Benchmark the CURRENTLY DEPLOYED (old) API
python scripts/benchmark.py ^
    --image tests/fixtures/sample_cxr.png ^
    --requests 30 --concurrency 1,4 ^
    --json baseline_v1.0.0.json
```

Keep `baseline_v1.0.0.json`. That is your before column.

> If you've already deployed, run the baseline against a locally checked-out
> copy of the previous commit instead: `git stash`, run, `git stash pop`.

---

## Step 1 — Run the tests locally

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

Expect **57 passing**. If `test_batched_matches_sequential_in_distribution`
fails, the batching rewrite has a real problem — tell me before deploying,
because that test failing means uncertainty estimates would be wrong.

Then confirm the count for the badge:

```bash
pytest tests/ --collect-only -q | tail -1
```

Update the badge in `README.md` if it isn't 57.

---

## Step 2 — Deploy and re-benchmark

Push the backend to your HF Space, wait for the build, then:

```bash
python scripts/benchmark.py ^
    --image tests/fixtures/sample_cxr.png ^
    --requests 200 --concurrency 1,4,16 ^
    --json results_v1.1.0.json
```

Use `--requests 200` at minimum. **p99 from 50 requests is a single sample** —
quoting it would be the kind of thing that falls apart in an interview.

Also capture the fast path:

```bash
python scripts/benchmark.py --image tests/fixtures/sample_cxr.png ^
    --requests 200 --concurrency 1 --no-report
```

The script prints a Markdown table. Paste it straight into README.

Server-side stage timings come back in every response, so you get the
per-stage breakdown for free — the script prints `server_stages_mean_ms`.

---

## Step 3 — Cost per 1k inferences

HF Spaces free tier is $0, which makes for a boring table. Quote the rate for
hardware you'd actually deploy on:

```bash
# A10G at roughly $1.20/hr
python scripts/benchmark.py --image tests/fixtures/sample_cxr.png ^
    --requests 200 --concurrency 1,4,16 --gpu-cost-per-hour 1.20
```

Label it clearly as a projection at that rate, not as money you spent.

---

## Step 4 — Quantization (optional, do it last)

```bash
pip install onnx onnxruntime

python scripts/export_onnx.py --checkpoint chestai_best.pt --outdir export/

python scripts/eval_quantization.py ^
    --fp32 export/thoraxnet.onnx ^
    --int8 export/thoraxnet_int8.onnx ^
    --data-root path/to/images ^
    --labels-csv path/to/Data_Entry_2017.csv ^
    --split-list splits/val_list.txt ^
    --bench --json quantization_delta.json
```

Publish the delta table **whatever it says**. If INT8 costs you 0.004 mean AUC,
say so and say why you accepted or rejected it. A published regression is
evidence you measured; a claim of "no accuracy loss" with no table is evidence
of nothing.

If you're short on time, skip this entirely. Steps 0–2 are where the signal is.

---

## Step 5 — Commit hygiene from here on

These changes are best pushed as separate commits, not one blob:

```bash
git add models/uncertainty.py tests/test_uncertainty_batching.py
git commit -m "perf: batch MC dropout into single forward pass

20 sequential batch-1 passes through ViT-B/16 left the GPU idle between
launches. Tiling the input along the batch dim yields the same T independent
dropout samples in one pass. Chunked at 32 to bound memory.

Adds equivalence test against the sequential implementation at n=400."

git add explainability/gradcam.py api/routes/predict.py tests/test_gradcam_cache.py
git commit -m "fix: gradcam overlays were discarded before the session store read them

predict.py read pipeline.gradcam._last_overlays, which was never assigned --
the pipeline built overlays into a local and dropped them. Every
GET /gradcam/{id}/{class} 404'd. Adds LRU cache keyed on (sha1, class_idx)."

git add api/inference.py api/schemas.py
git commit -m "perf: move Groq report off the event loop, make it optional"

git add .github/workflows/ci.yml requirements-ci.txt
git commit -m "ci: install CPU torch instead of the CUDA build

requirements.txt pulled ~2.5GB of CUDA torch onto a GPU-less runner, which
is what was timing out the job."

git add CHANGELOG.md README.md
git commit -m "docs: add changelog, perf tables and engineering notes"
```

Check yourself afterwards:

```bash
git log --shortstat --oneline -10
```

Commits touching 300+ lines are doing too much.

### Issues to open on your own repo

Open these now and close them with PRs as you go — it's a visible record of
deliberate engineering, and it costs ten minutes:

1. *GradCAM session store is process-local; won't survive restart or scale
   past one replica* → needs Redis
2. *INT8 weights exported but not serving; accuracy delta unpublished*
3. *No load test of a horizontally scaled deployment*
4. *Pneumonia AUC 0.695 — investigate whether label noise or class overlap
   dominates*

Those are already listed under "Known gaps" in the README, so the issues and
the docs agree.

---

## Step 6 — The blog post

Highest return of anything on this list. Structure:

1. What ThoraxNet does — two sentences, then the live link
2. "4.8 seconds is not a product" — why latency mattered here
3. **Profiling first**: the surprise that a single forward pass was 15 ms and
   the bottleneck was calling it 20 times
4. The batching fix, with the correctness argument about per-element dropout
   masks (this is the part that shows you understand what you changed)
5. Async Groq, and why p95 degraded faster than p50 before it
6. The GradCAM bug — a feature that had never worked and nobody noticed
7. Final numbers, honestly labelled
8. What you'd do next: Redis for the session store, INT8 promotion

Publish on dev.to. Link it from the README and your resume.

Write it after step 2, while the details are fresh and the numbers are real.
