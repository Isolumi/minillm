# MiniLLM

A local text-generation server built around a Hugging Face model. The server exposes a text-only subset of the OpenAI Responses API while the inference engine is developed incrementally.

## Run

From the repository root, with the model in `models/smollm2-1.7b-instruct`:

```bash
uv run python main.py
```

The server listens at `http://127.0.0.1:8123`. Set `MINILLM_MODEL_PATH` to use another local model directory; optionally set `MINILLM_MODEL_NAME` to choose the ID exposed by the API.

## Responses API

```bash
curl http://127.0.0.1:8123/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"smollm2-1.7b-instruct","input":"Hello!","max_output_tokens":64}'
```

The same endpoint can be called with an OpenAI client by setting its base URL to `http://127.0.0.1:8123/v1` and providing any nonempty API key:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8123/v1", api_key="unused")
first = client.responses.create(
    model="smollm2-1.7b-instruct",
    input="My name is Avery.",
    max_output_tokens=64,
)
print(first.output_text)

second = client.responses.create(
    model="smollm2-1.7b-instruct",
    previous_response_id=first.id,
    input="What is my name?",
    max_output_tokens=64,
)
print(second.output_text)
```

The endpoint accepts a text `input` or an array of simple `{role, content}` messages. For follow-up turns, use `previous_response_id` or send the full message history. `instructions` provides a system message for the current request; resend it on follow-ups if needed. `store` defaults to `true`; `store: false` prevents later retrieval or continuation by that response ID. The server keeps at most 128 responses in memory, evicts the oldest, and loses them on restart. It does not retain GPU KV cache for these responses.

This text-only subset supports greedy output, `max_output_tokens` (default 128, maximum 1024), `temperature=0`, and `stream=false`. Unsupported request fields are rejected. Streaming, sampling, tools, and multimodal content are planned for later milestones.

`GET /v1/responses/{response_id}` retrieves a stored response. `GET /v1/models` lists the served model, and `GET /health` reports server readiness.
