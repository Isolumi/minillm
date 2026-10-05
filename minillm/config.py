"""Small, explicit environment configuration. Checkpoints stay local."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def integer(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default)
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(sorted(choices))}")
    return value


@dataclass(frozen=True)
class ModelSpec:
    id: str
    path: str
    backend: str = "custom"
    tokenizer: str = "custom"


@dataclass(frozen=True)
class Settings:
    device: str = "cuda"
    dtype: str = "float16"
    max_context: int = 8192
    max_requests: int = 8
    max_queue: int = 64
    prefill_chunk: int = 256
    cache_memory_mb: int = 2048
    cache_layout: str = "paged"
    cache_dtype: str = "auto"
    attention_backend: str = "auto"
    prefix_cache: bool = True
    max_model_turn_requests: int = 16

    @classmethod
    def from_env(cls):
        return cls(
            device=os.getenv("MINILLM_DEVICE", "cuda"),
            dtype=choice(
                "MINILLM_DTYPE", "float16", {"float16", "bfloat16", "float32"}
            ),
            max_context=integer("MINILLM_MAX_CONTEXT", 8192),
            max_requests=integer("MINILLM_MAX_REQUESTS", 8),
            max_queue=integer("MINILLM_MAX_QUEUE", 64),
            prefill_chunk=integer("MINILLM_PREFILL_CHUNK", 256),
            cache_memory_mb=integer("MINILLM_CACHE_MB", 2048),
            cache_layout=choice(
                "MINILLM_CACHE_LAYOUT", "paged", {"paged", "contiguous"}
            ),
            cache_dtype=choice("MINILLM_CACHE_DTYPE", "auto", {"auto", "int8"}),
            attention_backend=choice(
                "MINILLM_ATTENTION", "auto", {"auto", "torch", "triton"}
            ),
            prefix_cache=choice("MINILLM_PREFIX_CACHE", "1", {"0", "1"}) == "1",
            max_model_turn_requests=integer("MINILLM_MODEL_TURN_REQUESTS", 16),
        )


def checkpoint_ready(path: str | Path) -> bool:
    path = Path(path)
    if not (path / "config.json").is_file() or not (path / "tokenizer.json").is_file():
        return False
    index = path / "model.safetensors.index.json"
    if index.exists():
        try:
            names = set(json.loads(index.read_text())["weight_map"].values())
            return bool(names) and all((path / name).is_file() for name in names)
        except ValueError, KeyError:
            return False
    return (path / "model.safetensors").is_file()


def model_specs() -> list[ModelSpec]:
    config = os.getenv("MINILLM_MODELS_CONFIG")
    if config:
        config_path = Path(config).resolve()
        rows = json.loads(config_path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError("MINILLM_MODELS_CONFIG must contain a nonempty JSON array")
        specs = []
        for row in rows:
            spec = ModelSpec(**row)
            path = Path(spec.path)
            if not path.is_absolute():
                path = config_path.parent / path
            specs.append(
                ModelSpec(spec.id, str(path.resolve()), spec.backend, spec.tokenizer)
            )
    else:
        primary = Path(
            os.getenv("MINILLM_MODEL_PATH", str(ROOT / "models/smollm2-1.7b-instruct"))
        ).resolve()
        name = os.getenv("MINILLM_MODEL_NAME", primary.name)
        backend = choice("MINILLM_BACKEND", "custom", {"custom", "hf", "multimodal"})
        tokenizer = choice(
            "MINILLM_TOKENIZER",
            "custom" if backend == "custom" else "hf",
            {"custom", "hf"},
        )
        specs = [ModelSpec(name, str(primary), backend, tokenizer)]
        # Extra checkpoints are advertised only after all their weight shards exist.
        if "MINILLM_MODEL_PATH" not in os.environ:
            candidates = [
                ModelSpec(
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    str(ROOT / "models/smollm2-360m-instruct"),
                ),
                ModelSpec(
                    "Qwen/Qwen2.5-7B-Instruct",
                    str(ROOT / "models/qwen2.5-7b-instruct"),
                    "hf",
                    "hf",
                ),
                ModelSpec(
                    "google/gemma-4-E4B-it",
                    str(ROOT / "models/gemma4-e4b"),
                    "multimodal",
                    "hf",
                ),
            ]
            specs.extend(s for s in candidates if checkpoint_ready(s.path))
    ids = set()
    for spec in specs:
        if not spec.id or spec.id in ids:
            raise ValueError(f"Invalid or duplicate model ID: {spec.id}")
        ids.add(spec.id)
        if spec.backend not in {"custom", "hf", "multimodal"} or spec.tokenizer not in {
            "custom",
            "hf",
        }:
            raise ValueError(f"Unsupported backend/tokenizer for {spec.id}")
        if not checkpoint_ready(spec.path):
            raise ValueError(
                f"Checkpoint incomplete for {spec.id}: {spec.path}; download it first"
            )
    return specs
