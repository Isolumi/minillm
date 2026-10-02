# minillm

A small single-GPU inference engine in Python and PyTorch, with browser chat and
an OpenAI-compatible Responses API.

- Custom SmolLM2 forward pass, BPE tokenizer, and safetensors loader
- Continuous batching, chunked prefill, and prefix reuse
- Paged KV cache, Triton decode attention, and optional INT8 KV storage
- Streaming, conversation history, cancellation, and image input with Gemma

Runs SmolLM2 360M/1.7B directly, plus Qwen2.5 7B and Gemma 4 E4B through
Transformers. One model stays in VRAM at a time.

## Run

```bash
uv sync --locked
uv run hf download HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --local-dir models/smollm2-1.7b-instruct \
  --include '*.json' --include '*.safetensors' --include '*.jinja'
uv run python main.py
```

Open [localhost:8123](http://127.0.0.1:8123). Requires Python 3.14+ and an NVIDIA
GPU. For other local checkpoints, see [models.example.json](models.example.json)
and set `MINILLM_MODELS_CONFIG` to your config file.

## API

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8123/v1", api_key="local")
response = client.responses.create(
    model="smollm2-1.7b-instruct",
    input="Explain how a KV cache works.",
    max_output_tokens=128,
)
print(response.output_text)
```

## RTX 5090 numbers

SmolLM2 1.7B, FP16, 40-token prompt, 32 output tokens, warm prefix cache.
Medians from three batches after one warmup, measured through the streaming API.

| Concurrent requests | Total output tokens/s | Request latency |
| ---: | ---: | ---: |
| 1 | 146.2 | 219 ms |
| 4 | 425.9 | 284 ms |
| 8 | 618.5 | 387 ms |

INT8 KV storage uses **46.9% less cache memory**, including scales.
Single-request decode is still slower than the Hugging Face cached runner.

To rerun the throughput benchmark with the server running:

```bash
uv run python scripts/bench_engine.py --model smollm2-1.7b-instruct \
  --output-tokens 32 --concurrency 8 --warmup 1 --repetitions 3 \
  --output benchmarks/http.json
```
