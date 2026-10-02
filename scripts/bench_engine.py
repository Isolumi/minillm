"""Measure the live Responses API with the official OpenAI Python client."""

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def percentiles(values):
    values = sorted(values)
    return {"median": statistics.median(values), "p95": values[math.ceil(0.95 * len(values)) - 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8123/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain why the sky is blue in one paragraph.")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.output_tokens, args.concurrency, args.repetitions) < 1 or args.warmup < 0:
        parser.error("output-tokens, concurrency and repetitions must be positive; warmup must be nonnegative")

    import httpx
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key="not-needed", timeout=600)
    metrics_url = args.base_url.rstrip("/").removesuffix("/v1") + "/v1/metrics"

    def metrics():
        try:
            result = httpx.get(metrics_url, timeout=10)
            result.raise_for_status()
            return result.json()
        except (httpx.HTTPError, ValueError):
            return None

    def one():
        started = time.perf_counter()
        first_visible = None
        terminal = None
        delta_text = ""
        with client.responses.create(model=args.model, input=args.prompt,
                                     max_output_tokens=args.output_tokens, stream=True,
                                     store=False, temperature=0) as stream:
            for event in stream:
                if event.type == "response.output_text.delta" and event.delta:
                    if first_visible is None:
                        first_visible = time.perf_counter() - started
                    delta_text += event.delta
                elif event.type in {"response.completed", "response.incomplete", "response.failed"}:
                    terminal = event.response
        elapsed = time.perf_counter() - started
        if terminal is None:
            raise RuntimeError("Stream ended without a terminal response event")
        if terminal.status == "failed":
            raise RuntimeError(f"Generation failed: {terminal.error}")
        final_text = terminal.output_text
        if delta_text != final_text:
            raise RuntimeError("Streamed text differs from the final response text")
        usage = terminal.usage
        return {"first_visible_seconds": first_visible, "completion_seconds": elapsed,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "cached_tokens": usage.input_tokens_details.cached_tokens if usage and usage.input_tokens_details else None,
                "status": terminal.status, "response_id": terminal.id}

    def group():
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            requests = list(pool.map(lambda _: one(), range(args.concurrency)))
        elapsed = time.perf_counter() - started
        tokens = sum(row["output_tokens"] or 0 for row in requests)
        return {"wall_seconds": elapsed, "output_tokens": tokens,
                "aggregate_tokens_per_second": tokens / elapsed, "requests": requests}

    for index in range(args.warmup):
        group()
        print(f"warmup {index + 1}/{args.warmup} complete", flush=True)
    before = metrics()
    groups = []
    for index in range(args.repetitions):
        row = group()
        groups.append(row)
        print(f"sample {index + 1}/{args.repetitions}: {row['aggregate_tokens_per_second']:.2f} output tokens/s", flush=True)
    after = metrics()
    requests = [request for group_row in groups for request in group_row["requests"]]
    first_visible = [row["first_visible_seconds"] for row in requests if row["first_visible_seconds"] is not None]
    result = {"benchmark": "engine_http_stream", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "settings": {"base_url": args.base_url, "model": args.model, "prompt": args.prompt,
                           "output_tokens": args.output_tokens, "concurrency": args.concurrency,
                           "repetitions": args.repetitions, "warmup": args.warmup},
              "summary": {"first_visible_seconds": percentiles(first_visible) if first_visible else None,
                          "completion_seconds": percentiles([row["completion_seconds"] for row in requests]),
                          "aggregate_tokens_per_second": percentiles([row["aggregate_tokens_per_second"] for row in groups]),
                          "output_tokens": sum(row["output_tokens"] or 0 for row in requests),
                          "cached_input_tokens": sum(row["cached_tokens"] or 0 for row in requests)},
              "metrics_before": before, "metrics_after": after, "samples": groups}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
