# MiniLLM inference engine roadmap

## Goal and scope

Build a small, understandable single-GPU text inference engine inspired by vLLM. Add an OpenAI-compatible Responses endpoint early and keep it working through later milestones. Start with Hugging Face as the reference implementation, then replace the generation loop, attention, KV cache, scheduling, tokenizer, and eventually the weight loader one piece at a time.

This is a sequence of independently demonstrable milestones, not a promise of full vLLM feature parity. Finish each milestone before starting the next. Keep the current implementation available as an oracle until its replacement passes the relevant comparison checks. No pytest setup is required: each milestone can use a small executable check script and an HTTP smoke request.

Assumptions: one local NVIDIA GPU, one text model, one server process, greedy decoding first, SmolLM2-1.7B-Instruct as the initial model. The current GPU is an RTX 5090 with about 32 GiB. SmolLM2's local config has 24 layers, 32 attention heads, 32 KV heads, head dimension 64, and an 8,192-token context. Thus it uses ordinary multi-head attention, not GQA. At two bytes per element its full KV cache is approximately `2 (K,V) * 24 * 32 * 64 * 8192 * 2 = 1.5 GiB` per request. Use synthetic GQA cases at first; add a GQA model later.

## Keep these boundaries stable

```text
HTTP adapter -> Engine request -> Scheduler -> Model runner -> Attention/cache backend
                    ^                               ^
             tokenizer adapter              HF oracle initially
```

- HTTP handles OpenAI-shaped JSON, validation, error mapping, and streaming. It never owns KV tensors.
- Engine owns active requests and their lifecycle. Each request has token IDs, generated IDs, sampling settings, status, and a result queue/stream.
- Scheduler decides which requests run on the next model step and tracks token and memory budgets.
- Model runner accepts scheduled token IDs, positions, and cache handles; returns logits and updates KV state.
- Cache backend owns physical KV storage and request-to-storage mappings. It has explicit allocate, append, read, and release operations.
- Tokenizer adapter produces exact token IDs from messages and decodes output. Start with `AutoTokenizer`.

Keep `minillm/server.py` and `minillm/schemas.py` as the API boundary. As implementations appear, add focused modules under `minillm/engine/`, `minillm/model/`, `minillm/cache/`, `minillm/kernels/`, and `minillm/tokenization/`. Put repeatable checks and benchmarks in `scripts/`. Avoid moving existing code solely for the sake of a folder structure.

| External component | Replacement milestone | Keep as oracle until |
| --- | --- | --- |
| `model.generate()` | 3: own decode loop | Greedy token IDs and stop behavior agree |
| Hugging Face `DynamicCache` | 5: own contiguous cache in own runner | Per-layer outputs and logits agree |
| Hugging Face model forward/attention | 5: own model runner and PyTorch attention | Model logits agree; retain optional comparison mode |
| PyTorch decode attention | 9: contiguous Triton kernel; 10: paged Triton reader | Attention output and full-model output agree |
| `AutoTokenizer` | 13: exact SmolLM2 tokenizer | Token IDs, special tokens, and decoding agree |
| Hugging Face checkpoint loading | 14: own safetensors loader | Weight names, shapes, dtypes, and values agree |

## Shared completion rule

Every milestone ends with four pieces of evidence:

1. A small deterministic correctness check against the previous implementation, using the same input IDs, weights, dtype, and positions.
2. A normal request through `/v1/responses` still works. Once streaming exists, check that too.
3. A short measurement records latency and peak GPU memory when performance is relevant. Record the setup, warmup, and number of repetitions alongside the result.
4. A short note identifies the external component still in use and the one just replaced.

For numerics, compare tensors with documented dtype-specific tolerances and report maximum absolute/relative error. For generation, compare greedy token IDs and locate the first divergence. A different answer alone does not diagnose an attention bug: close logits can have different argmax values. Make the oracle run in the same dtype when comparing it to your runner.

## Milestones

### 0. Freeze the existing behavior

**Build:** Record a few fixed prompts, their chat-template token IDs, model/tokenizer file hashes, first-step logits (or a small selected logit slice), generated token IDs, EOS behavior, and the current server response. Keep the working dependency set pinned in `uv.lock`.

**Check:** A repeat run produces the same greedy token IDs. A request with a small `max_new_tokens` stops at the budget; an EOS-producing prompt stops early. Capture baseline first-token time, subsequent-token time, and peak GPU memory on the current server.

**Learn:** A reproducible reference is needed before numerical or performance changes can be interpreted.

### 1. Add a real OpenAI-compatible entry point

**Build:** Add `GET /health`, `GET /v1/models`, and `POST /v1/responses`. Start with text-only `input`, one loaded model ID, `max_output_tokens`, `temperature=0`, and `stream=false`. Translate input with the model's chat template. Return a `response` object with typed `output` message items, status, and token usage. Support `previous_response_id` through an in-memory transcript store. Reject unsupported options explicitly rather than silently ignoring them.

**Check:** The official OpenAI Python client can call `client.responses.create(...)` using `base_url="http://127.0.0.1:8123/v1"` and a dummy key. Verify `output_text`, a follow-up via `previous_response_id`, retrieval, and an unknown model ID error. Count usage from token IDs, not whitespace. Check that the response status distinguishes EOS from token-budget exhaustion.

**Learn:** API contracts and stateless chat history. Full `messages` are client-visible history; KV reuse is a server optimization.

### 2. Build reference attention in ordinary PyTorch

**Build:** Evolve `minillm/attention.py` into a pure reference implementation with shape `[batch, sequence, q_heads, head_dim]`. Add causal masking, scaling, RoPE with explicit positions, and GQA head mapping (`q_heads % kv_heads == 0`). Separate prefill and one-token decode functions. Keep this independent of the full model.

**Check:** For batch sizes 1 and 3, sequence lengths 1, 2, 17, and 33, and both MHA and GQA shapes, compare against a simple PyTorch attention equation. Future tokens must not affect earlier outputs; changing a position must change the corresponding RoPE result. Test FP32 first, then the model's execution dtype.

**Learn:** Q/K/V layouts, causal masking, softmax, RoPE, MHA, GQA, and position offsets. SmolLM2 itself only tests MHA.

### 3. Own single-request decoding while using Hugging Face cache

**Build:** Replace `model.generate()` inside the engine with a loop around `model.forward()`. Prefill once, then feed only the newly generated token with the correct position and cache. Implement greedy argmax, EOS detection, output budget, output decoding, and `finish_reason`. Retain `AutoModelForCausalLM`, `AutoTokenizer`, and `DynamicCache` at this milestone.

**Check:** Compare first-step logits and several greedy token sequences to the frozen Hugging Face `generate()` oracle. Check prompt length 1, multi-turn chat templates, EOS on the first step, and exact output-budget exhaustion. The HTTP endpoint must still respond as before.

**Learn:** Prefill versus decode, `cache_position`, how returned logits correspond to the *next* token, and why an inference engine must control each decoding step.

### 4. Build a contiguous KV cache, initially in isolation

**Build:** Preallocate K and V for a fixed number of requests, layers, KV heads, and positions. Expose request-slot allocation, append, read-prefix, length, and release. Do not concatenate a new tensor every token. Use it with the reference attention layer first. Keep this cache backend independent of `DynamicCache`.

**Check:** Cached one-token outputs match full-prefix recomputation for several lengths and batches. Fill a buffer exactly to capacity, reject one more append cleanly, release a slot, reuse it, and confirm no stale keys survive. Confirm the underlying storage pointer stays fixed while appending. Record bytes allocated and append time.

**Learn:** Memory layout, preallocation, mutable state, and cache lifetime. At this point the custom cache exists, but the full model still uses `DynamicCache`.

### 5. Integrate your attention and cache into SmolLM2

**Build:** Implement a minimal Llama-style forward runner for *this model configuration*: embedding, RMSNorm, Q/K/V and output projections, RoPE, attention, gated MLP, final norm, and tied output embedding. Load the existing Hugging Face checkpoint weights initially; continue using `AutoTokenizer`. Your runner uses the contiguous cache and reference attention for prefill and decode. Keep the Hugging Face runner selectable as an oracle while developing this. Do not assume all Llama-family checkpoints have identical configuration details.

**Check:** For fixed token IDs, compare embeddings, each block's output, and final logits against the Hugging Face model. Find the first layer that diverges. Then compare greedy token IDs for short prompts and two-turn chats. Full API request still works when the custom runner is selected. This milestone retires `DynamicCache` from the custom execution path.

**Learn:** Weight mapping, normalization, residuals, dtype effects, model-wide numerical drift, and end-to-end token latency.

### 6. Create the benchmark and profiling harness

**Build:** Add `scripts/bench_engine.py` and a smaller `scripts/bench_attention.py`. Measure prefill latency, time to first token, time per output token, total output tokens/second, peak allocated and reserved GPU memory, and cache bytes. Cover prompt lengths 128, 1,024, and 4,096 and one/few concurrent requests. Use warmup, synchronize around GPU timing, and report medians and setup. Keep the HF and custom runners on the same prompt set.

**Check:** Repeated runs produce plausible, stable figures and report exactly which runner, dtype, prompt length, output length, batch size, and GPU were used. A profile identifies where decode time goes before kernel work starts.

**Learn:** CUDA events, CPU/GPU synchronization, memory bandwidth, launch overhead, and end-to-end versus kernel-only metrics.

### 7. Add a request scheduler and continuous decode batching

**Build:** Give the engine a waiting queue and active set. One engine worker owns the GPU and advances multiple requests one model step at a time. Prefill incoming prompts first, then batch the one-token decode steps of active requests; admit new requests between steps. Add admission limits, per-request cancellation, EOS/max-token completion, and cache release. Begin with greedy decoding and the contiguous cache. Keep mixed prefill/decode and chunked prefill for a later iteration.

**Check:** Submit three concurrent HTTP requests with different prompt and output lengths. Compare each response's token IDs to isolated execution. The short request finishes while the long one continues. A new request joins before existing long requests finish. Cancellation and errors free cache slots. Throughput and per-request latency are measured against the serial engine.

**Learn:** Continuous batching, queueing, lifecycle state, variable lengths, fairness, backpressure, and why a global generation lock limits throughput.

### 8. Add streaming and basic sampling

**Build:** Have each decode step publish tokens to the corresponding request stream. Implement Responses `stream=true` with SSE events. Add seeded temperature/top-p sampling after greedy decoding is stable, keeping all sampling state per request. Handle client disconnect by cancelling the engine request and releasing its cache. Clearly return an error for unsupported OpenAI options.

**Check:** The OpenAI Python client consumes streamed chunks, reconstructs the same text as a non-streaming greedy call, and receives a terminal chunk. Two streams interleave progress without mixing request IDs. Client disconnect frees its active slot. A fixed seed reproduces sampled token IDs under the same execution settings.

**Learn:** SSE, token-to-text boundary handling, sampling, cancellation, and async request coordination.

### 9. Write a Triton decode-attention kernel for contiguous cache

**Build:** Write a forward-only kernel for one query token per active sequence and head. Read K/V in tiles, use online softmax with stable running max/sum, and write the attention output without materializing the full attention matrix. First implement the contiguous layout; support the model's head dimension, then test other head counts including synthetic GQA. Leave prefill on the reference implementation at first.

**Check:** Compare kernel output to the PyTorch reference for short/long contexts, non-power-of-two lengths, partial tiles, batch sizes 1 and 4, and varied Q/KV head ratios. Test finite outputs and masked positions. Benchmark only after correctness, against PyTorch reference and the full model's token latency.

**Learn:** Triton indexing, coalescing, reductions, online softmax, GPU occupancy, and why a faster kernel may have little end-to-end effect.

### 10. Move cache storage to fixed-size pages

**Build:** Introduce a pool of fixed-size KV blocks, per-request block tables, a free list, length tracking, and explicit allocation/release. Start with an easy block size such as 16 tokens and only one model layout. Implement a slow PyTorch gather-based attention reader as a correctness oracle, then adapt the Triton decode kernel to read physical blocks via the request's block table directly. Do not gather the entire prefix into contiguous tensors in the optimized path.

**Check:** Noncontiguous physical block IDs produce the same outputs as contiguous cache. Exercise lengths 15, 16, and 17; repeated allocate/release; multiple requests; and out-of-blocks behavior. Verify no block is shared accidentally and all blocks return to the free list after requests finish or cancel. Compare peak memory and concurrent capacity against the contiguous implementation.

**Learn:** Logical versus physical positions, block-table indirection, internal fragmentation, reference counts, and memory accounting.

### 11. Add block-level prefix reuse and chunked prefill

**Build:** Hash full token blocks together with their parent prefix hash and model identity. Allow new requests to reuse computed full blocks, reference-count them, and use copy-on-write or a fresh block for partial tail writes. Then let the scheduler prefill long prompts in bounded chunks so decode requests keep making progress. Keep a clear policy for cache eviction and memory pressure.

**Check:** Two requests with the same prefix yield identical tokens while the second performs fewer prefill computations. Different prompts never share the wrong block. A cancelled request cannot free a block still referenced by another request. A long prompt does not stall ongoing decode for its entire prefill. Report cache-hit tokens and time to first token.

**Learn:** Prefix hashes, copy-on-write, block ownership, eviction, and mixed prefill/decode scheduling.

### 12. Quantize KV pages only after the paged path is correct

**Build:** Add INT8 KV pages with explicit scale metadata and tilewise dequantization inside the attention kernel. Measure per-tensor versus per-head or per-block scales. Add packed INT4 only if INT8 shows a useful memory/capacity tradeoff. Keep BF16/FP16 pages as the reference backend.

**Check:** Check packing/unpacking against known values and end-to-end logits/text quality against the unquantized path. Report cache bytes, maximum concurrent requests, first-token time, and token latency. Keep quantization selectable so regressions are observable.

**Learn:** Quantization error, scale granularity, bit packing, memory bandwidth, and dequantization cost.

### 13. Replace the tokenizer

**Build:** Implement the exact tokenizer behavior needed by SmolLM2: its BPE vocabulary/merges, byte handling, added/special tokens, decoding, and chat-template formatting. Put it behind the tokenizer adapter. Retain `AutoTokenizer` as a selectable reference until parity is established.

**Check:** Compare token IDs and decoded text with `AutoTokenizer` over ordinary text, Unicode, whitespace, special tokens, multi-turn messages, and assistant-generation prompts. The same HTTP request must produce the same prompt token IDs and greedy output token IDs after the swap.

**Learn:** BPE internals, chat templates, and why exact tokenization is part of model correctness.

### 14. Replace the checkpoint loader

**Build:** Load `safetensors` and map named tensors to your runner without constructing `AutoModelForCausalLM`. Load one model configuration explicitly before attempting general checkpoint support.

**Check:** Compare loaded weights by tensor name, shape, dtype, and sample values. Re-run the same fixed prompt and compare initial logits and greedy token IDs to the Hugging Face-loaded runner.

**Learn:** Checkpoint formats, tensor ownership, tied embeddings, and startup memory use.

## Priority and deferrals

The shortest path to a compelling vLLM-like demo is milestones **0–8**, followed by **9–10** for the kernel and paged-cache core. Prefix caching (11) and KV quantization (12) are advanced optimizations. Replacing the tokenizer and loader (13–14) is valuable for learning but does not directly improve GPU serving throughput.

Do not claim full OpenAI API compatibility: document the supported subset. Do not call the project equivalent to vLLM's performance without workload-matched benchmarks; its current implementation includes many features beyond this single-GPU text scope.

## References

- [vLLM online serving and supported API routes](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [vLLM scheduler interface and token budgeting](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/)
- [vLLM automatic prefix caching design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
- [Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [OpenAI Chat Completions API reference](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions/methods/create)
