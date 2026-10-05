"""Compare PyTorch cached decode and the Triton cache reader on synthetic CUDA data."""

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def summary(values):
    values = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": values[math.ceil(0.95 * len(values)) - 1],
    }


def errors(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-6)
    return {"max_abs": difference.max().item(), "max_rel": relative.max().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths", default="128,1024,4096", help="Comma-separated KV lengths"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--cache-layout", choices=("paged", "contiguous"), default="paged"
    )
    parser.add_argument("--cache-dtype", choices=("auto", "int8"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        lengths = [int(value) for value in args.lengths.split(",")]
    except ValueError:
        parser.error("lengths must be comma-separated integers")
    if (
        not lengths
        or min(
            *lengths,
            args.batch,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.repetitions,
        )
        < 1
        or args.warmup < 0
    ):
        parser.error(
            "dimensions, lengths and repetitions must be positive; warmup must be nonnegative"
        )
    if args.q_heads % args.kv_heads:
        parser.error("q-heads must be divisible by kv-heads")

    import torch

    from minillm.attention import decode_attention, reference_attention
    from minillm.cache import ContiguousKVCache, PagedKVCache
    from minillm.kernels import paged_decode_attention

    if not torch.cuda.is_available():
        parser.error("CUDA is required for this comparison")
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1234)
    results = []

    def timed(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repetitions):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return summary(samples), samples

    for length in lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        query = torch.randn(
            (args.batch, args.q_heads, args.head_dim), device=device, dtype=dtype
        )
        keys = torch.randn(
            (args.batch, length, args.kv_heads, args.head_dim),
            device=device,
            dtype=dtype,
        )
        values = torch.randn_like(keys)
        cache_class = (
            PagedKVCache if args.cache_layout == "paged" else ContiguousKVCache
        )
        bytes_per_token = (
            2
            * args.kv_heads
            * (
                args.head_dim * (1 if args.cache_dtype == "int8" else 2)
                + (4 if args.cache_dtype == "int8" else 0)
            )
        )
        budget_mb = max(
            1, math.ceil(args.batch * length * bytes_per_token / 1024**2) + 1
        )
        cache = cache_class(
            1,
            args.kv_heads,
            args.head_dim,
            device=device,
            dtype=dtype,
            cache_dtype=args.cache_dtype,
            max_context=length,
            max_requests=args.batch,
            memory_mb=budget_mb,
            prefix_cache=False,
        )
        states = [cache.allocate() for _ in range(args.batch)]
        cache.reserve_batch(states, [length] * args.batch)
        for index, state in enumerate(states):
            cache.write(0, state, 0, keys[index], values[index])
        metadata = cache.metadata(states)

        def pytorch_cached():
            rows = []
            for index, state in enumerate(states):
                key, value = cache.read(0, state)
                rows.append(
                    decode_attention(
                        query[index : index + 1, None], key[None], value[None]
                    )[0, 0]
                )
            return torch.stack(rows)

        def triton_cached():
            return paged_decode_attention(query, cache, 0, states, metadata)

        with torch.inference_mode():
            original = reference_attention(query[:, None], keys, values, causal=False)[
                :, 0
            ]
            quantized_rows = []
            for index, state in enumerate(states):
                key, value = cache.read(0, state)
                quantized_rows.append(
                    reference_attention(
                        query[index : index + 1, None],
                        key[None],
                        value[None],
                        causal=False,
                    )[0, 0]
                )
            quantized_reference = torch.stack(quantized_rows)
            torch_output = pytorch_cached()
            triton_output = triton_cached()
            torch.cuda.synchronize()
            torch_time, torch_samples = timed(pytorch_cached)
            triton_time, triton_samples = timed(triton_cached)
            result = {
                "kv_length": length,
                "latency": {
                    "pytorch_cache_read_and_sdpa": torch_time,
                    "triton_precomputed_metadata": triton_time,
                },
                "error": {
                    "cache_quantization_vs_original_fp32_reference": errors(
                        quantized_reference, original
                    ),
                    "pytorch_sdpa_vs_cached_fp32_reference": errors(
                        torch_output, quantized_reference
                    ),
                    "triton_vs_cached_fp32_reference": errors(
                        triton_output, quantized_reference
                    ),
                    "triton_vs_pytorch": errors(triton_output, torch_output),
                },
                "cache": cache.stats(),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "samples_ms": {"pytorch": torch_samples, "triton": triton_samples},
            }
        results.append(result)
        for state in states:
            cache.release(state)
        print(
            f"length {length}: PyTorch {torch_time['median_ms']:.3f} ms, Triton {triton_time['median_ms']:.3f} ms",
            flush=True,
        )

    output = {
        "benchmark": "synthetic_cached_decode_attention",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "lengths": lengths,
            "batch": args.batch,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "cache_layout": args.cache_layout,
            "cache_dtype": args.cache_dtype,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
