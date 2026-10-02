"""Benchmark fixed-token custom, HF cached, or HF uncached model execution."""

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def summary(values):
    values = sorted(values)
    return {"median_ms": statistics.median(values), "p95_ms": values[math.ceil(0.95 * len(values)) - 1]}


def repeated_tokens(source, length):
    if not source:
        raise ValueError("Tokenizer returned no tokens")
    return (source * math.ceil(length / len(source)))[:length]


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backend", choices=("custom", "hf", "uncached"), required=True)
    parser.add_argument("--prompt-tokens", default="128,1024,4096", help="One length or comma-separated lengths")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cache-layout", choices=("paged", "contiguous"), default="paged")
    parser.add_argument("--cache-dtype", choices=("auto", "int8"), default="auto")
    parser.add_argument("--attention", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-memory-mb", type=int, default=2048)
    parser.add_argument("--profile", action="store_true", help="Profile one extra run at the first prompt length")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        lengths = [int(value) for value in args.prompt_tokens.split(",")]
    except ValueError:
        parser.error("prompt-tokens must be one integer or comma-separated integers")
    if not lengths or min(*lengths, args.output_tokens, args.repetitions, args.cache_memory_mb) < 1 or args.warmup < 0:
        parser.error("token counts, repetitions and cache-memory-mb must be positive; warmup must be nonnegative")

    import torch
    import transformers
    from minillm.model.hf import HFRunner
    from minillm.tokenization import HFTokenizer

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a functional fallback")
    dtype = getattr(torch, args.dtype) if device.type == "cuda" else torch.float32
    model_path = str(args.model_path.resolve())
    tokenizer = HFTokenizer(model_path)
    prompt_source = tokenizer.encode_messages([{"role": "user", "content":
        "Explain how a small language model generates the next token. " * 16}])
    continuation_source = tokenizer.encode(" A deterministic continuation for the benchmark.")
    forced = repeated_tokens(continuation_source, args.output_tokens)
    max_context = max(lengths) + args.output_tokens
    if args.backend == "custom":
        from minillm.model.smollm import SmolLMRunner
        constructor = lambda: SmolLMRunner(model_path, device=device, dtype=dtype,
            cache_layout=args.cache_layout, cache_dtype=args.cache_dtype,
            attention_backend=args.attention, max_context=max_context, max_requests=1,
            cache_memory_mb=args.cache_memory_mb, prefix_cache=False)
    else:
        constructor = lambda: HFRunner(model_path, device=device, dtype=dtype,
                                      max_context=max_context, multimodal=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loaded_at = time.perf_counter()
    runner = constructor()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loader_seconds = time.perf_counter() - loaded_at
    if max_context > runner.context_limit:
        parser.error(f"Requested {max_context} tokens exceeds the model context limit {runner.context_limit}")
    after_load_bytes = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    loader_peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    loader_peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None

    def clocked(fn):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            value = fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end), value
        started = time.perf_counter()
        value = fn()
        return (time.perf_counter() - started) * 1000, value

    def run_once(prompt_ids):
        state = None
        try:
            if args.backend == "uncached":
                def forward(ids):
                    tensor = torch.tensor([ids], device=device, dtype=torch.long)
                    return runner.model(tensor, use_cache=False, logits_to_keep=1, return_dict=True).logits[0, -1]
                prefill_ms, logits = clocked(lambda: forward(prompt_ids))
                def decode_full_prefix():
                    nonlocal logits
                    for index in range(args.output_tokens):
                        logits = forward(prompt_ids + forced[:index + 1])
                    return logits
                decode_ms, logits = clocked(decode_full_prefix)
                cache = {"cache_bytes": 0, "mode": "full-prefix recomputation"}
            else:
                state = runner.create_state()
                prefill_ms, logits = clocked(lambda: runner.prefill(prompt_ids, state))
                def decode_cached():
                    nonlocal logits
                    for token in forced:
                        logits = runner.decode([token], [state])[0]
                    return logits
                decode_ms, logits = clocked(decode_cached)
                cache = runner.stats()
            # Force logits to be consumed; no argmax or sampling changes the token workload.
            checksum = float(logits.float().sum().item())
            return {"prefill_ms": prefill_ms, "decode_total_ms": decode_ms,
                    "decode_per_token_ms": decode_ms / args.output_tokens,
                    "final_logits_sum": checksum, "cache": cache,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        finally:
            if state is not None:
                runner.release(state)

    results = []
    with torch.inference_mode():
        for length in lengths:
            prompt_ids = repeated_tokens(prompt_source, length)
            for _ in range(args.warmup):
                run_once(prompt_ids)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            samples = [run_once(prompt_ids) for _ in range(args.repetitions)]
            row = {"prompt_tokens": length, "prompt_ids_sha256": ids_hash(prompt_ids),
                   "forced_continuation_ids_sha256": ids_hash(forced),
                   "prefill": summary([sample["prefill_ms"] for sample in samples]),
                   "decode_total": summary([sample["decode_total_ms"] for sample in samples]),
                   "decode_per_token": summary([sample["decode_per_token_ms"] for sample in samples]),
                   "gpu_peak_allocated_bytes": max((sample["gpu_peak_allocated_bytes"] or 0) for sample in samples),
                   "gpu_peak_reserved_bytes": max((sample["gpu_peak_reserved_bytes"] or 0) for sample in samples),
                   "cache": samples[-1]["cache"], "samples": samples}
            results.append(row)
            print(f"{length} prompt tokens: prefill {row['prefill']['median_ms']:.2f} ms, "
                  f"decode {row['decode_per_token']['median_ms']:.2f} ms/token", flush=True)

        profile_table = None
        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profile:
                run_once(repeated_tokens(prompt_source, lengths[0]))
            sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
            profile_table = profile.key_averages().table(sort_by=sort_key, row_limit=20)
            print(profile_table)

    output = {"benchmark": "fixed_token_runner", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "settings": {"model_path": model_path, "backend": args.backend, "prompt_tokens": lengths,
                           "output_tokens": args.output_tokens, "warmup": args.warmup,
                           "repetitions": args.repetitions, "cache_layout": args.cache_layout,
                           "cache_dtype": args.cache_dtype, "attention": args.attention,
                           "dtype": str(dtype), "device": str(device), "cache_memory_mb": args.cache_memory_mb,
                           "token_source": "HF chat template and fixed repeated token IDs"},
              "hardware": {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                           "torch": torch.__version__, "cuda": torch.version.cuda,
                           "transformers": transformers.__version__},
              "loader_seconds": loader_seconds, "gpu_allocated_after_load_bytes": after_load_bytes,
              "loader_peak_allocated_bytes": loader_peak_allocated, "loader_peak_reserved_bytes": loader_peak_reserved,
              "forced_continuation_ids_sha256": ids_hash(forced),
              "results": results, "profiler_top_operations": profile_table}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
