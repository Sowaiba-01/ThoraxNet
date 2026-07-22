# Why My Medical AI Took 6.4 Seconds Per Scan — and How I Got It to 3.1

*Building ThoraxNet, a chest X-ray diagnostic platform, and the unglamorous
engineering that made it usable.*

---

ThoraxNet detects 14 thoracic pathologies from a chest X-ray. It runs a
fine-tuned BioMedCLIP ViT-B/16, reports Monte Carlo Dropout uncertainty, draws
GradCAM heatmaps, and writes a structured radiology report. The model was the
fun part. This post is about the part that actually decides whether something
is a product: latency.

Live demo: [thorax-tho.vercel.app](https://thorax-tho.vercel.app) ·
Code: [github.com/Sowaiba-01/ThoraxNet](https://github.com/Sowaiba-01/ThoraxNet)

---

## 6.4 seconds is not a product

The model worked. The demo worked. But every scan took about six and a half
seconds, and a six-second spinner makes anything feel broken, no matter how
good the output is.

The first thing I did was refuse to guess. I've watched myself "optimize" the
wrong thing enough times to know the intuition — *the ViT forward pass must be
the bottleneck* — is worth exactly nothing until it's measured. So I added
per-stage timing to the inference pipeline and returned it on every response:

```json
"stage_timings_ms": {
  "preprocess": 15.7,
  "mc_dropout": 2770.5,
  "gradcam": 5.4,
  "report": 42.4
}
```

The result was not what I expected. A single forward pass through the ViT is
~15 ms. The model was never the problem. **Monte Carlo Dropout was — 2.7 of
the ~2.8 server-side seconds.** And once I looked at *how* it ran, the reason
was almost embarrassing.

## The bug that wasn't a bug: 20 passes, one at a time

MC Dropout estimates uncertainty by running the model many times with dropout
left on, then looking at how much the predictions wobble. My implementation did
the obvious thing:

```python
samples = []
for _ in range(n_samples):        # n_samples = 20
    logits = model(x)             # one image, batch size 1
    samples.append(torch.sigmoid(logits))
```

Twenty sequential forward passes, each with a batch size of one. Every pass
pays the full per-call overhead, and the hardware spends most of its time
waiting between launches rather than computing.

The fix is to stop asking 20 times and ask once — tile the single image into a
batch of 20 and run a single forward pass:

```python
tiled = x.repeat(n_samples, 1, 1, 1)   # (20, 3, 224, 224)
logits = model(tiled)                  # ONE forward pass
probs = torch.sigmoid(logits).view(n_samples, batch, -1)
mean, std = probs.mean(0), probs.std(0)
```

The correctness argument is the part worth understanding, and it's the question
an interviewer will ask: *aren't those 20 copies identical?* No — dropout
samples a fresh mask per element of the batch. So the 20 tiled copies get 20
independent dropout masks, which is exactly the 20 independent stochastic
samples the estimator needs. Same statistics, one launch instead of twenty.

I didn't want to take that on faith, so there's a test that runs both the old
sequential version and the new batched version 400 times each and asserts the
means agree within Monte Carlo error:

```python
assert torch.allclose(batched_mean, sequential_mean, atol=0.05)
```

## The honest number: 2×, not 10×

Here's where I have to be straight, because it would be easy to lie here.

On a GPU, this change is enormous — often close to 10×. The whole win comes
from keeping an accelerator busy that was otherwise idle between tiny launches.

ThoraxNet runs on a free-tier CPU box (2 vCPUs). There is far less idle
parallelism to reclaim. So the same code change gave me **~2×, not ~10×**:

| Version | p50 | p95 | p99 |
|---|---|---|---|
| Before | 6,387 ms | 7,525 ms | 8,026 ms |
| After  | 3,146 ms | 3,287 ms | 3,395 ms |

Measured, 30 requests, same image, zero failures. A 2.03× end-to-end
improvement, and the tail improved more than the median (p99 2.36×) because
batching kills the per-pass overhead that hurt the slowest requests worst.

I could have written "10× faster" and most people wouldn't have checked. But
the number that survives an interview is the one you can explain: *2× on CPU,
because the batching win is bounded by idle parallelism, and MC Dropout on CPU
is still the bottleneck — the next real lever is GPU inference or INT8, not more
batching.* That sentence is worth more than a bigger fake number.

## Two bugs I found by reading the whole request path

Profiling made me read the entire path from HTTP request to response, and two
things fell out that had nothing to do with latency.

**GradCAM had never worked.** The route handler read
`pipeline.gradcam._last_overlays` — an attribute that was never assigned
anywhere. The pipeline built the heatmaps into a local variable and dropped
them when the function returned. Every heatmap request returned 404. Nothing
logged an error; the frontend just showed an empty panel. It had shipped and
sat broken because no code path ever raised. I fixed it and wrote a regression
test that fails if the overlays aren't recorded — because a silent bug deserves
a loud test.

**Every radiology report was silently failing.** While watching the deploy
logs I saw:

```
The model `llama3-70b-8192` has been decommissioned and is no longer supported.
```

Groq had retired the model months earlier. Every report request returned HTTP
400 and fell back to a template, so users got canned text instead of an
LLM-written report — and nobody noticed, because the fallback made it look
fine. One line to point at the current model, plus an env var so the next
deprecation is config, not code.

Neither bug was in my ticket. Both were only visible because I stopped trusting
the happy path and read the logs.

## What I'd do next

MC Dropout still dominates at 2.7 s. Batching took the easy win; the real
remaining levers are honest about the hardware:

- **GPU or ONNX/INT8 inference** — the scripts are written; the accuracy-delta
  table is the homework I still owe.
- **Redis for the GradCAM session store** — right now it's process-local
  memory that won't survive a restart or a second replica.

## The takeaway

The model was 15 ms. The product was 6.4 seconds. The gap was entirely in how
the model was *called*, not in the model itself — and I only found that because
I measured before I touched anything, and read the logs instead of the ticket.

The unglamorous work is the work.

---

*ThoraxNet is for research use only. Not FDA cleared. Not a substitute for a
radiologist.*
