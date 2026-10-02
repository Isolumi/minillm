# MiniLLM inference engine roadmap

## Goal and scope

Build a small, understandable single-GPU inference engine inspired by vLLM. Keep an OpenAI-compatible Responses endpoint working while replacing the generation loop, attention, KV cache, scheduling, tokenizer, and eventually the weight loader one piece at a time. Start with text; add multiple model IDs and an explicitly HF-backed multimodal path before attempting custom support for other model architectures.

This is a sequence of independently demonstrable milestones, not a promise of full vLLM feature parity. Finish each milestone before starting the next. Keep the current implementation available as an oracle until its replacement passes the relevant comparison checks. No pytest setup is required: each milestone can use a small executable check script and an HTTP smoke request.

Assumptions: one local NVIDIA GPU, one server process, greedy decoding first, SmolLM2-1.7B-Instruct as the initial text model. The current GPU is an RTX 5090 with about 32 GiB. SmolLM2's local config has 24 layers, 32 attention heads, 32 KV heads, head dimension 64, and an 8,192-token context. Thus it uses ordinary multi-head attention, not GQA. At two bytes per element its full KV cache is approximately `2 (K,V) * 24 * 32 * 64 * 8192 * 2 = 1.5 GiB` per request. Use synthetic GQA cases at first; add a GQA model later. All models listed below are candidate downloads, not claims that their weights are already present locally or can all remain resident together.

| Demo role | Model | Execution path when introduced |
| --- | --- | --- |
| Correctness and kernel development | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Existing HF path, then custom SmolLM2 runner |
| First second model ID | `HuggingFaceTB/SmolLM2-360M-Instruct` | Same-family custom runner after milestone 5 |
| GQA and larger text workload | `Qwen/Qwen2.5-7B-Instruct` (28 Q heads, 4 KV heads) | HF-backed initially; custom architecture support is a separate later extension |
| First multimodal workload | `google/gemma-4-E4B-it` | HF processor and multimodal model; custom Gemma internals are a separate later extension |

Do not compare speed across different models as evidence of an optimization. “Available through the API” is distinct from “simultaneously resident in VRAM”: record the loading/eviction policy and measure load time separately from steady-state inference.

## Keep these boundaries stable

```text
HTTP adapter -> Model registry -> Engine request -> Scheduler -> Model runner -> Attention/cache backend
                       |               ^                               ^
             tokenizer/processor      per-model state             HF oracle initially
```

- HTTP handles OpenAI-shaped JSON, validation, error mapping, and streaming. It never owns KV tensors.
- Engine owns active requests and their lifecycle. Each request has token IDs, generated IDs, sampling settings, status, and a result queue/stream.
- Scheduler decides which requests run on the next model step and tracks token and memory budgets.
- Model runner accepts scheduled token IDs, positions, and cache handles; returns logits and updates KV state.
- Cache backend owns physical KV storage and request-to-storage mappings. It has explicit allocate, append, read, and release operations.
- Model registry resolves the requested model ID to its runner and tokenizer/processor. A request and its stored conversation remain bound to one model ID; caches, block tables, and prefix hashes never cross model IDs. Scheduler batches only compatible requests.
- Tokenizer/processor adapter produces exact model inputs and decodes output. Start with `AutoTokenizer` for text and `AutoProcessor` for HF-backed multimodal input.

Keep `minillm/server.py` and `minillm/schemas.py` as the API boundary. As implementations appear, add focused modules under `minillm/engine/`, `minillm/model/`, `minillm/cache/`, `minillm/kernels/`, and `minillm/tokenization/`. Put repeatable checks and benchmarks in `scripts/`. Add a processor/input adapter only when the multimodal milestone arrives. Avoid moving existing code solely for the sake of a folder structure.

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

1. A small deterministic correctness check against the previous implementation, using the same input IDs, weights, dtype, and positions where applicable. Milestone 0 establishes the reference by repeating itself; new-model stages compare each checkpoint to its own HF oracle.
2. A normal request through `/v1/responses` still works. Once streaming exists, check that too.
3. A reproducible demo artifact shows the stage's mechanism and outcome: a small table, chart, trace, or HTTP transcript under `docs/results/`, with the command used to produce it. Keep a tiny measurement helper from milestone 0; milestone 6 expands it into a full benchmark harness.
4. A short note identifies the external component still in use and the one just replaced.

For performance comparisons, keep model revision, weights, dtype, prompt token IDs, output token IDs, context length, batch/concurrency, and GPU fixed. Warm up, synchronize CUDA measurements, report median and p95 over a stated repetition count, and separate kernel timing from wall-clock HTTP latency. Use fixed/forced continuation tokens when comparing backends so a different generated answer does not change the workload. Report GPU peak allocated/reserved bytes and cache bytes separately. If a step is primarily about correctness or API behavior, its demo can be a trace/transcript rather than a speedup chart. Never label a count of decoder positions as measured FLOPs or wall-clock speedup.

For numerics, compare tensors with documented dtype-specific tolerances and report maximum absolute/relative error. For generation, compare greedy token IDs and locate the first divergence. A different answer alone does not diagnose an attention bug: close logits can have different argmax values. Make the oracle run in the same dtype when comparing it to your runner.

## Milestones

### 0. Freeze the existing behavior

**Build:** Record a few fixed prompts, their chat-template token IDs, model/tokenizer file hashes, first-step logits (or a small selected logit slice), generated token IDs, EOS behavior, and the current server response. Keep the working dependency set pinned in `uv.lock`. Add a minimal repeatable measurement script now; the full profiling harness waits until milestone 6.

**Check:** A repeat run produces the same greedy token IDs. A request with a small `max_new_tokens` stops at the budget; an EOS-producing prompt stops early. Capture baseline first-token time, subsequent-token time, and peak GPU memory on the current server.

**Learn:** A reproducible reference is needed before numerical or performance changes can be interpreted.

**Demo:** Show one reproducible baseline card: model revision, input/output token IDs, first-token latency, per-token latency, and peak VRAM, plus the command that generated it. This is the reference row for later same-model comparisons.

### 1. Add a real OpenAI-compatible entry point

**Build:** Add `GET /health`, `GET /v1/models`, and `POST /v1/responses`. Start with text-only `input`, one loaded model ID, `max_output_tokens`, `temperature=0`, and `stream=false`. Translate input with the model's chat template. Return a `response` object with typed `output` message items, status, and token usage. Support `previous_response_id` through an in-memory transcript store. Reject unsupported options explicitly rather than silently ignoring them.

**Check:** The official OpenAI Python client can call `client.responses.create(...)` using `base_url="http://127.0.0.1:8123/v1"` and a dummy key. Verify `output_text`, a follow-up via `previous_response_id`, retrieval, and an unknown model ID error. Count usage from token IDs, not whitespace. Check that the response status distinguishes EOS from token-budget exhaustion.

**Learn:** API contracts and stateless chat history. Full `messages` are client-visible history; KV reuse is a server optimization.

**Demo:** Use the OpenAI client to create, continue, and retrieve a response; show the request/response transcript and one explicit unsupported-option error. This milestone's result is interoperability, not a GPU speedup.

### 2. Build reference attention in ordinary PyTorch

**Build:** Evolve `minillm/attention.py` into a pure reference implementation with shape `[batch, sequence, q_heads, head_dim]`. Add causal masking, scaling, RoPE with explicit positions, and GQA head mapping (`q_heads % kv_heads == 0`). Separate prefill and one-token decode functions. Keep this independent of the full model.

**Check:** For batch sizes 1 and 3, sequence lengths 1, 2, 17, and 33, and both MHA and GQA shapes, compare against a simple PyTorch attention equation. Future tokens must not affect earlier outputs; changing a position must change the corresponding RoPE result. Test FP32 first, then the model's execution dtype.

**Learn:** Q/K/V layouts, causal masking, softmax, RoPE, MHA, GQA, and position offsets. SmolLM2 itself only tests MHA.

**Demo:** Show tensor shapes, a causal-mask example, an MHA/GQA head-mapping sketch, and a table of maximum error versus the PyTorch reference across lengths and head ratios. Label synthetic GQA as synthetic.

### 3. Own single-request decoding while using Hugging Face cache

**Build:** Replace `model.generate()` inside the engine with a loop around `model.forward()`. Prefill once, then feed only the newly generated token with the correct position and cache. Implement greedy argmax, EOS detection, output budget, output decoding, and `finish_reason`. Retain `AutoModelForCausalLM`, `AutoTokenizer`, and `DynamicCache` at this milestone.

**Check:** Compare first-step logits and several greedy token sequences to the frozen Hugging Face `generate()` oracle. Check prompt length 1, multi-turn chat templates, EOS on the first step, and exact output-budget exhaustion. The HTTP endpoint must still respond as before. Compare cached prefill-plus-decode to a deliberately uncached full-prefix-forward baseline on the same forced continuation tokens.

**Learn:** Prefill versus decode, `cache_position`, how returned logits correspond to the *next* token, and why an inference engine must control each decoding step.

**Demo:** Plot per-step decode latency as context grows for cached versus full-prefix recomputation; include prompt/output token counts and total wall time. For a 1,024-token prompt and 128 output tokens, the uncached baseline processes `128*1024 + 128*127/2 = 139,200` decoder token positions versus `1024+127 = 1,151` for cache use: approximately 121× fewer *positions processed*, not 121× fewer FLOPs or measured time. This is the main KV-cache compute-saving demo; do not re-claim it at milestone 4.

### 4. Build a contiguous KV cache, initially in isolation

**Build:** Preallocate K and V for a fixed number of requests, layers, KV heads, and positions. Expose request-slot allocation, append, read-prefix, length, and release. Do not concatenate a new tensor every token. Use it with the reference attention layer first. Keep this cache backend independent of `DynamicCache`.

**Check:** Cached one-token outputs match full-prefix recomputation for several lengths and batches. Fill a buffer exactly to capacity, reject one more append cleanly, release a slot, reuse it, and confirm no stale keys survive. Confirm the underlying storage pointer stays fixed while appending. Record bytes allocated and append time.

**Learn:** Memory layout, preallocation, mutable state, and cache lifetime. At this point the custom cache exists, but the full model still uses `DynamicCache`.

**Demo:** Show a cache-lifetime trace (allocate, append, read, release, reuse), fixed storage-pointer evidence, allocated-versus-used bytes, append-time curve, and cached-versus-uncached attention error. Compare with `DynamicCache` where feasible, but do not claim another compute reduction from merely changing cache storage.

### 5. Integrate your attention and cache into SmolLM2

**Build:** Implement a minimal Llama-style forward runner for *this model configuration*: embedding, RMSNorm, Q/K/V and output projections, RoPE, attention, gated MLP, final norm, and tied output embedding. Load the existing Hugging Face checkpoint weights initially; continue using `AutoTokenizer`. Your runner uses the contiguous cache and reference attention for prefill and decode. Keep the Hugging Face runner selectable as an oracle while developing this. Do not assume all Llama-family checkpoints have identical configuration details.

**Check:** For fixed token IDs, compare embeddings, each block's output, and final logits against the Hugging Face model. Find the first layer that diverges. Then compare greedy token IDs for short prompts and two-turn chats. Full API request still works when the custom runner is selected. This milestone retires `DynamicCache` from the custom execution path.

**Learn:** Weight mapping, normalization, residuals, dtype effects, model-wide numerical drift, and end-to-end token latency.

**Demo:** Show a per-layer maximum-error table and first-divergence location, then the same fixed prompt through `/v1/responses` with HF and custom SmolLM2 runners. Report end-to-end latency and VRAM for those two paths under the same workload.

### 5a. Serve two models from the same family

**Build:** Introduce a model registry keyed by API model ID and an isolated runner/tokenizer/cache bundle per loaded model. Add `HuggingFaceTB/SmolLM2-360M-Instruct` alongside the 1.7B model. Reuse the custom SmolLM2 runner only after checking the smaller model's config and weight mapping; retain the HF path as its oracle. Start with serial execution or explicit per-model admission; do not silently mix model requests in one decode batch. Advertise only loaded/servable models from `/v1/models`.

**Check:** Both model IDs return responses through the same endpoint; each matches its own HF oracle on fixed token IDs. Unknown IDs and attempts to continue a stored response under another model ID fail explicitly. Cache release for one model does not touch the other. Record cold-load time and per-model peak/resident VRAM.

**Learn:** Model registries, per-model isolation, checkpoint/config validation, and resident-versus-load-on-demand tradeoffs.

**Demo:** Show two client requests differing only in `model`, the `/v1/models` listing, each model's token-parity result, and a VRAM residency table. Do not present the models' raw speed difference as an engine optimization.

### 6. Create the benchmark and profiling harness

**Build:** Add `scripts/bench_engine.py` and a smaller `scripts/bench_attention.py`. Measure prefill latency, time to first token, time per output token, total output tokens/second, peak allocated and reserved GPU memory, and cache bytes. Cover prompt lengths 128, 1,024, and 4,096 and one/few concurrent requests. Use warmup, synchronize around GPU timing, and report medians and setup. Keep the HF and custom runners on the same prompt set.

**Check:** Repeated runs produce plausible, stable figures and report exactly which runner, dtype, prompt length, output length, batch size, and GPU were used. A profile identifies where decode time goes before kernel work starts.

**Learn:** CUDA events, CPU/GPU synchronization, memory bandwidth, launch overhead, and end-to-end versus kernel-only metrics.

**Demo:** Produce a reproducible benchmark table or chart with HF/custom runner, prompt length, batch size, p50/p95 prefill and decode latency, throughput, and peak allocated/reserved VRAM. Highlight one profiler finding that motivates the next optimization. This expands the earlier smoke measurements rather than introducing measurement for the first time.

### 7. Add a request scheduler and continuous decode batching

**Build:** Give the engine a waiting queue and active set. One engine worker owns the GPU and advances multiple requests one model step at a time. Prefill incoming prompts first, then batch the one-token decode steps of compatible active requests; admit new requests between steps. Group by model ID, runner, and cache layout; never put different model IDs into one forward batch. Add admission limits, per-request cancellation, EOS/max-token completion, and cache release. Begin with greedy decoding and the contiguous cache. Keep mixed prefill/decode and chunked prefill for a later iteration.

**Check:** Submit three concurrent HTTP requests with different prompt and output lengths. Compare each response's token IDs to isolated execution. The short request finishes while the long one continues. A new request joins before existing long requests finish. Cancellation and errors free cache slots. Throughput and per-request latency are measured against the serial engine.

**Learn:** Continuous batching, queueing, lifecycle state, variable lengths, fairness, backpressure, and why a global generation lock limits throughput.

**Demo:** Run 1, 4, and 8 simultaneous same-model clients; chart aggregate tokens/s and per-request p50/p95 latency against serial execution. Show a timeline where a short request finishes and a new one joins while a long request is still decoding. Separately show that different model IDs stay in isolated groups.

### 8. Add streaming and basic sampling

**Build:** Have each decode step publish tokens to the corresponding request stream. Implement Responses `stream=true` with SSE events. Add seeded temperature/top-p sampling after greedy decoding is stable, keeping all sampling state per request. Handle client disconnect by cancelling the engine request and releasing its cache. Clearly return an error for unsupported OpenAI options.

**Check:** The OpenAI Python client consumes streamed chunks, reconstructs the same text as a non-streaming greedy call, and receives a terminal chunk. Two streams interleave progress without mixing request IDs. Client disconnect frees its active slot. A fixed seed reproduces sampled token IDs under the same execution settings.

**Learn:** SSE, token-to-text boundary handling, sampling, cancellation, and async request coordination.

**Demo:** Display a live SSE trace with time to first visible chunk and completion time, alongside the non-streaming result. Show two interleaved streams and a reproducible seeded sample; cancellation should visibly free the request's cache slot.

### 8a. Add a GQA text model through the HF backend

**Build:** Register `Qwen/Qwen2.5-7B-Instruct` as a distinct text architecture using its own HF tokenizer/model runner. It provides a real GQA workload (28 query heads, 4 KV heads), but does not yet claim custom Qwen model-forward support. Bound its max context/output and residency according to measured free VRAM; a 32 GiB card does not imply every listed model can be loaded at once.

**Check:** Compare a fixed Qwen response and token IDs to its HF reference. Reject unsupported attempts to route Qwen into SmolLM2-only custom attention/cache code. Confirm mixed-model HTTP requests cannot reuse one another's history, cache handles, or prefix state. Record load time, steady-state memory, and single-model latency.

**Learn:** GQA in a real model, architecture-specific configs, and the difference between API multi-model support and a generic custom runner.

**Demo:** Call the same Responses endpoint with SmolLM2 and Qwen model IDs, show a real 28:4 GQA shape trace, and report each model's resource profile without claiming cross-model performance improvement.

### 8b. Add image-and-text input through an HF-backed multimodal model

**Build:** Add typed input content parts and preserve them in stored conversation history instead of reducing every message to a string. Register `google/gemma-4-E4B-it` with its `AutoProcessor` and HF multimodal model path; start with text plus a single image and text output. Accept only bounded, explicitly supported image sources/sizes and reject unsupported modalities/options. Keep this path separate from SmolLM2's custom text runner and label it HF-backed in the demo/docs. Treat audio and sampled video as later sub-milestones only after image input works end to end; each gets its own processor, limits, checks, and demo.

**Check:** A text-only request and a text-plus-image request both work for Gemma; multi-turn continuation retains the image-bearing history correctly. An invalid image, unsupported part type, and a cross-model continuation fail clearly. Compare preprocessed inputs and greedy output tokens with the direct HF path. Record preprocessing time, input token expansion, first-token time, and peak VRAM.

**Learn:** Typed multimodal messages, preprocessing, vision token expansion, media lifetime, and why a multimodal model is more than a text runner with a different checkpoint.

**Demo:** Send an image question through the same `/v1/responses` endpoint, show the structured request and answer, then an image-aware follow-up. Show a stage-timing breakdown (model load, media acquisition/decode, processor, prefill, token generation) and VRAM. If audio/video are added, demonstrate and measure each separately rather than bundling them into the image result.

### 9. Write a Triton decode-attention kernel for contiguous cache

**Build:** Write a forward-only kernel for one query token per active sequence and head. Read K/V in tiles, use online softmax with stable running max/sum, and write the attention output without materializing the full attention matrix. First implement the contiguous layout; support the model's head dimension, then test other head counts including synthetic GQA. Leave prefill on the reference implementation at first.

**Check:** Compare kernel output to the PyTorch reference for short/long contexts, non-power-of-two lengths, partial tiles, batch sizes 1 and 4, and varied Q/KV head ratios. Test finite outputs and masked positions. Benchmark only after correctness, against PyTorch reference and the full model's token latency.

**Learn:** Triton indexing, coalescing, reductions, online softmax, GPU occupancy, and why a faster kernel may have little end-to-end effect.

**Demo:** Plot Triton versus PyTorch decode-attention latency over context length for the same inputs, including maximum numerical error. Alongside it, report the full-model per-token latency before/after kernel selection so kernel speedup is not mistaken for end-to-end speedup. Use real Qwen GQA shapes as a kernel test only after the kernel supports them; this alone does not integrate the custom Qwen runner.

### 10. Move cache storage to fixed-size pages

**Build:** Introduce a pool of fixed-size KV blocks, per-request block tables, a free list, length tracking, and explicit allocation/release. Start with an easy block size such as 16 tokens and only one model layout. Implement a slow PyTorch gather-based attention reader as a correctness oracle, then adapt the Triton decode kernel to read physical blocks via the request's block table directly. Do not gather the entire prefix into contiguous tensors in the optimized path.

**Check:** Noncontiguous physical block IDs produce the same outputs as contiguous cache. Exercise lengths 15, 16, and 17; repeated allocate/release; multiple requests; and out-of-blocks behavior. Verify no block is shared accidentally and all blocks return to the free list after requests finish or cancel. Compare peak memory and concurrent capacity against the contiguous implementation.

**Learn:** Logical versus physical positions, block-table indirection, internal fragmentation, reference counts, and memory accounting.

**Demo:** Show a visual block-table/free-list trace for requests of 15, 16, and 17 tokens, plus used/reserved KV bytes and the maximum number of admitted requests under an identical memory budget for contiguous versus paged storage. Report attention-output parity and any decode latency cost.

### 11. Add block-level prefix reuse and chunked prefill

**Build:** Hash full token blocks together with their parent prefix hash and model identity. Allow new requests to reuse computed full blocks, reference-count them, and use copy-on-write or a fresh block for partial tail writes. Then let the scheduler prefill long prompts in bounded chunks so decode requests keep making progress. Keep a clear policy for cache eviction and memory pressure.

**Check:** Two requests with the same prefix yield identical tokens while the second performs fewer prefill computations. Different prompts never share the wrong block. A cancelled request cannot free a block still referenced by another request. A long prompt does not stall ongoing decode for its entire prefill. Report cache-hit tokens and time to first token.

**Learn:** Prefix hashes, copy-on-write, block ownership, eviction, and mixed prefill/decode scheduling.

**Demo:** Replay a shared-prefix request cold and warm. Show cached-token count, actual prefill tokens computed, and time to first token; then show a timeline in which chunked prefill lets an existing decode request progress. Hashes and reported hits must include model identity, so no cross-model reuse is possible.

### 12. Quantize KV pages only after the paged path is correct

**Build:** Add INT8 KV pages with explicit scale metadata and tilewise dequantization inside the attention kernel. Measure per-tensor versus per-head or per-block scales. Add packed INT4 only if INT8 shows a useful memory/capacity tradeoff. Keep BF16/FP16 pages as the reference backend.

**Check:** Check packing/unpacking against known values and end-to-end logits/text quality against the unquantized path. Report cache bytes, maximum concurrent requests, first-token time, and token latency. Keep quantization selectable so regressions are observable.

**Learn:** Quantization error, scale granularity, bit packing, memory bandwidth, and dequantization cost.

**Demo:** Present BF16/FP16 versus INT8 (and INT4 only if implemented) for cache bytes per token, concurrency at a fixed VRAM cap, per-token latency, maximum logit error, and any output divergence on fixed prompts. Describe the memory/quality/latency tradeoff, not only the compression ratio.

### 13. Replace the tokenizer

**Build:** Implement the exact tokenizer behavior needed by SmolLM2: its BPE vocabulary/merges, byte handling, added/special tokens, decoding, and chat-template formatting. Put it behind the tokenizer adapter. Retain `AutoTokenizer` as a selectable reference until parity is established.

**Check:** Compare token IDs and decoded text with `AutoTokenizer` over ordinary text, Unicode, whitespace, special tokens, multi-turn messages, and assistant-generation prompts. The same HTTP request must produce the same prompt token IDs and greedy output token IDs after the swap.

**Learn:** BPE internals, chat templates, and why exact tokenization is part of model correctness.

**Demo:** Show exact ID parity over the tokenizer corpus and encode/decode latency for HF versus custom tokenization. Keep the demo scoped to SmolLM2; Qwen and Gemma remain on their own HF tokenizer/processor until separately implemented.

### 14. Replace the checkpoint loader

**Build:** Load `safetensors` and map named tensors to your runner without constructing `AutoModelForCausalLM`. Load one model configuration explicitly before attempting general checkpoint support.

**Check:** Compare loaded weights by tensor name, shape, dtype, and sample values. Re-run the same fixed prompt and compare initial logits and greedy token IDs to the Hugging Face-loaded runner.

**Learn:** Checkpoint formats, tensor ownership, tied embeddings, and startup memory use.

**Demo:** Show weight-by-weight parity summary, first-logit and greedy-token parity, and startup time/peak VRAM for HF loading versus the custom SmolLM2 loader. Do not imply this loader automatically handles Qwen or Gemma checkpoints.

### Later architecture extensions (separate projects, not prerequisites for milestones 9–14)

- **Custom Qwen text path:** After the reference GQA kernel and model-registry boundaries are stable, implement Qwen-specific weight mapping, normalization/RoPE/config behavior, custom prefill/decode, and cache integration. Compare every layer's outputs and greedy tokens with the HF Qwen path before benchmarking. Demo the same Qwen checkpoint through HF and custom paths with layer-error and same-workload latency/memory results.
- **Custom Gemma multimodal path:** Only after the HF-backed image path is correct, replace components one at a time: processor/tokenizer parity, vision encoder, projector/merger, then Gemma text runner and its attention/cache behavior. Use HF intermediate tensors as oracles. Demo image-embedding, per-layer, and final-token parity before performance claims. Audio/video need their own verified processor/encoder paths; they are not obtained automatically by replacing the text decoder.

## Priority and deferrals

The shortest path to a compelling vLLM-like *text* demo is milestones **0–8**, including **5a** for two same-family model IDs, followed by **9–10** for the kernel and paged-cache core. **8a** and **8b** add real GQA and multimodality through clearly labeled HF-backed paths; they may be built after 8 in either order and must not block text-kernel work. Prefix caching (11) and KV quantization (12) are advanced optimizations. Replacing the tokenizer and loader (13–14) is valuable for learning but does not directly improve GPU serving throughput. Custom Qwen and Gemma internals are later separate projects, not hidden requirements of the API demos.

Do not claim full OpenAI API compatibility: document the supported subset. Do not call the project equivalent to vLLM's performance without workload-matched benchmarks; vLLM includes many features beyond this project's single-GPU scope.

## References

- [vLLM online serving and supported API routes](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [vLLM scheduler interface and token budgeting](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/)
- [vLLM automatic prefix caching design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
- [Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [OpenAI Responses create API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
- [OpenAI Responses streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses)
- [SmolLM2-360M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- [Qwen2.5-7B-Instruct model card and GQA head counts](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Gemma 4 E4B instruction model and HF multimodal loading path](https://huggingface.co/google/gemma-4-E4B-it)
