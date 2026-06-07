#!/usr/bin/env python3
"""Small local benchmark client for RetServe search endpoints."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any

import aiohttp


def percentile(values: list[float], percent: float) -> float:
    """Return a nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percent / 100) * len(ordered)) - 1))
    return ordered[index]


async def post_search(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
) -> tuple[bool, float]:
    """Send one search request and return success plus latency in milliseconds."""
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            await response.read()
            return response.status < 400, (time.perf_counter() - start) * 1000
    except Exception:
        return False, (time.perf_counter() - start) * 1000


async def run_benchmark(args: argparse.Namespace) -> None:
    """Run a fixed-concurrency search benchmark."""
    endpoint = args.base_url.rstrip("/") + args.path
    payload = {
        "queries": [args.query for _ in range(args.batch_size)],
        "topk": args.topk,
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    failures = 0

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:
        start = time.perf_counter()

        async def one_request() -> None:
            nonlocal failures
            async with semaphore:
                ok, latency_ms = await post_search(session, endpoint, payload)
                latencies.append(latency_ms)
                if not ok:
                    failures += 1

        await asyncio.gather(*(one_request() for _ in range(args.requests)))
        elapsed = time.perf_counter() - start

        metrics_text = ""
        if args.metrics:
            try:
                async with session.get(
                    args.base_url.rstrip("/") + "/metrics"
                ) as response:
                    metrics_text = await response.text()
            except Exception:
                metrics_text = ""

    successes = args.requests - failures
    qps = successes / elapsed if elapsed > 0 else 0.0
    print(f"endpoint={endpoint}")
    print(f"requests={args.requests} successes={successes} failures={failures}")
    print(f"elapsed_s={elapsed:.3f} qps={qps:.2f}")
    print(f"latency_ms_min={min(latencies, default=0):.2f}")
    print(f"latency_ms_avg={statistics.fmean(latencies) if latencies else 0:.2f}")
    print(f"latency_ms_p50={percentile(latencies, 50):.2f}")
    print(f"latency_ms_p95={percentile(latencies, 95):.2f}")
    print(f"latency_ms_p99={percentile(latencies, 99):.2f}")
    if metrics_text:
        for line in metrics_text.splitlines():
            if "query_cache" in line or line.startswith("retserve_requests_total"):
                print(line)


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description="Benchmark RetServe search")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--path", default="/search")
    parser.add_argument("--query", default="example retrieval query")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark CLI."""
    asyncio.run(run_benchmark(parse_args()))


if __name__ == "__main__":
    main()
