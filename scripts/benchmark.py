#!/usr/bin/env python3
"""
Latency and throughput benchmark for the ChestAI /predict endpoint.

Measures p50/p95/p99 latency and sustained throughput at several concurrency
levels, then prints a Markdown table ready to paste into the README.

Usage:
    # Against the deployed Space
    python scripts/benchmark.py --image tests/fixtures/sample_cxr.png

    # Against a local server, skipping the LLM report to isolate model latency
    python scripts/benchmark.py \
        --url http://localhost:7860/api/v1/predict \
        --image sample.png --no-report --requests 100

    # Emit JSON for CI regression tracking
    python scripts/benchmark.py --image sample.png --json results.json

Notes on methodology:
  * The first --warmup requests are discarded. Cold-start on a HF Space
    includes model download + weight load and is not representative.
  * Latency is measured client-side (wall clock around the HTTP call), so it
    includes network RTT. The server's own breakdown is available in the
    response's stage_timings_ms field and is reported separately.
  * p99 with N=50 is a single sample and is therefore noisy. Use --requests
    500 or more if you intend to quote a p99 anywhere it matters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

try:
    import aiohttp
except ImportError:
    sys.exit("aiohttp is required:  pip install aiohttp")

DEFAULT_URL = "https://Sowaiba01-ThoraxNet.hf.space/api/v1/predict"


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile. Expects a pre-sorted list."""
    if not sorted_values:
        return float("nan")
    k = max(0, min(len(sorted_values) - 1, int(round(p / 100.0 * len(sorted_values) + 0.5)) - 1))
    return sorted_values[k]


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    image_bytes: bytes,
    filename: str,
    want_report: bool,
    want_gradcam: bool,
    send_flags: bool = True,
) -> tuple[float, dict | None, int, str | None]:
    """Return (latency_ms, stage_timings, http_status, error_detail)."""
    content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"

    form = aiohttp.FormData()
    form.add_field("file", image_bytes, filename=filename, content_type=content_type)
    # Older deployments predate these flags. They're plain form fields, so an
    # older FastAPI handler ignores them — but --no-flags exists in case a
    # strict deployment rejects unknown fields.
    if send_flags:
        form.add_field("generate_report", "true" if want_report else "false")
        form.add_field("generate_gradcam", "true" if want_gradcam else "false")

    t0 = time.perf_counter()
    try:
        async with session.post(url, data=form) as resp:
            text = await resp.text()
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status != 200:
                return elapsed, None, resp.status, f"HTTP {resp.status}: {text[:400]}"
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                return elapsed, None, resp.status, f"non-JSON response: {text[:200]}"
            stages = body.get("stage_timings_ms") if isinstance(body, dict) else None
            return elapsed, stages, resp.status, None
    except Exception as e:
        return (
            (time.perf_counter() - t0) * 1000,
            None,
            0,
            f"{type(e).__name__}: {e}",
        )


async def run_level(
    url: str,
    image_bytes: bytes,
    filename: str,
    concurrency: int,
    n_requests: int,
    warmup: int,
    want_report: bool,
    want_gradcam: bool,
    send_flags: bool = True,
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def bounded():
            async with sem:
                return await one_request(
                    session, url, image_bytes, filename,
                    want_report, want_gradcam, send_flags,
                )

        # Warmup — results discarded.
        if warmup:
            await asyncio.gather(*[bounded() for _ in range(warmup)])

        wall_start = time.perf_counter()
        results = await asyncio.gather(*[bounded() for _ in range(n_requests)])
        wall_elapsed = time.perf_counter() - wall_start

    ok = [(lat, st) for lat, st, code, _ in results if code == 200]
    failures = len(results) - len(ok)
    latencies = sorted(lat for lat, _ in ok)

    # Collect distinct failure reasons so a total failure is diagnosable
    # instead of just reported as "all requests failed".
    errors: dict[str, int] = {}
    for _, _, code, detail in results:
        if code != 200 and detail:
            errors[detail] = errors.get(detail, 0) + 1

    if not latencies:
        return {
            "concurrency": concurrency,
            "requests": n_requests,
            "failures": failures,
            "error": "all requests failed",
            "error_detail": errors,
        }

    stage_totals: dict[str, list[float]] = {}
    for _, st in ok:
        if st:
            for k, v in st.items():
                stage_totals.setdefault(k, []).append(v)

    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "failures": failures,
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
        "mean_ms": round(statistics.fmean(latencies), 1),
        "min_ms": round(latencies[0], 1),
        "max_ms": round(latencies[-1], 1),
        "throughput_rps": round(len(ok) / wall_elapsed, 2),
        "error_detail": errors,
        "server_stages_mean_ms": {
            k: round(statistics.fmean(v), 1) for k, v in sorted(stage_totals.items())
        },
    }


def render_table(rows: list[dict], cost_per_gpu_hour: float | None) -> str:
    header = "| Concurrency | p50 | p95 | p99 | Throughput | Failures |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['concurrency']} | — | — | — | — | {r['requests']} |")
            continue
        lines.append(
            f"| {r['concurrency']} | {r['p50_ms']}ms | {r['p95_ms']}ms | "
            f"{r['p99_ms']}ms | {r['throughput_rps']} req/s | {r['failures']} |"
        )

    out = "\n".join(lines)

    if cost_per_gpu_hour:
        out += "\n\n| Concurrency | Cost per 1k inferences |\n|---|---|\n"
        for r in rows:
            if "error" in r:
                continue
            # 1000 requests / (req/s) = seconds of GPU time needed.
            seconds = 1000 / r["throughput_rps"]
            cost = seconds / 3600 * cost_per_gpu_hour
            out += f"| {r['concurrency']} | ${cost:.4f} |\n"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--image", required=True, help="Path to a test chest X-ray")
    ap.add_argument("--requests", type=int, default=50, help="Requests per concurrency level")
    ap.add_argument("--warmup", type=int, default=3, help="Discarded warmup requests")
    ap.add_argument("--concurrency", default="1,4,16", help="Comma-separated levels")
    ap.add_argument("--no-report", action="store_true", help="Skip LLM report generation")
    ap.add_argument("--no-gradcam", action="store_true", help="Skip GradCAM generation")
    ap.add_argument("--gpu-cost-per-hour", type=float, default=None,
                    help="e.g. 1.20 for an A10G — adds a cost-per-1k table")
    ap.add_argument("--json", default=None, help="Write raw results to this path")
    ap.add_argument("--no-flags", action="store_true",
                    help="Don't send generate_report/generate_gradcam form fields "
                         "(for older deployments that reject unknown fields)")
    ap.add_argument("--skip-health", action="store_true",
                    help="Skip the preflight /health check")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")
    image_bytes = image_path.read_bytes()

    levels = [int(x) for x in args.concurrency.split(",")]

    print(f"Endpoint : {args.url}")
    print(f"Image    : {image_path.name} ({len(image_bytes) / 1024:.1f} KB)")
    print(f"Report   : {'off' if args.no_report else 'on'}   "
          f"GradCAM: {'off' if args.no_gradcam else 'on'}")
    print(f"Requests : {args.requests} per level (+{args.warmup} warmup)\n")

    # Preflight: a sleeping HF Space or an unloaded model produces a wall of
    # identical failures that are much easier to read about here than to infer
    # from 30 stack traces.
    if not args.skip_health:
        health_url = args.url.split("/api/")[0] + "/health"

        async def check() -> None:
            timeout = aiohttp.ClientTimeout(total=90)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(health_url) as r:
                    print(f"Health   : HTTP {r.status} — {(await r.text())[:200]}\n")

        try:
            asyncio.run(check())
        except Exception as e:
            print(f"Health   : FAILED to reach {health_url} — {type(e).__name__}: {e}")
            print("           The Space may be asleep or the URL may be wrong.\n")

    rows = []
    for c in levels:
        print(f"  running concurrency={c} ...", flush=True)
        row = asyncio.run(
            run_level(
                args.url, image_bytes, image_path.name, c,
                args.requests, args.warmup,
                not args.no_report, not args.no_gradcam,
                not args.no_flags,
            )
        )
        rows.append(row)
        if "error" not in row:
            print(f"    p50={row['p50_ms']}ms  p95={row['p95_ms']}ms  "
                  f"p99={row['p99_ms']}ms  {row['throughput_rps']} req/s")
            if row["server_stages_mean_ms"]:
                print(f"    server stages: {row['server_stages_mean_ms']}")
            if row.get("error_detail"):
                print(f"    {row['failures']} failed:")
                for msg, n in row["error_detail"].items():
                    print(f"      [{n}x] {msg}")
        else:
            print(f"    {row['error']}")
            for msg, n in (row.get("error_detail") or {}).items():
                print(f"      [{n}x] {msg}")

    print("\n" + render_table(rows, args.gpu_cost_per_hour))

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nRaw results → {args.json}")


if __name__ == "__main__":
    main()
