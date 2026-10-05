"""One resident model at a time; all GPU ownership stays on the engine worker."""

import gc
import time
from dataclasses import dataclass

import torch

from minillm.config import ModelSpec, Settings
from minillm.model.hf import HFRunner, text_messages


@dataclass
class ModelBundle:
    spec: ModelSpec
    runner: object
    tokenizer: object
    capacity_tokens: int
    load_seconds: float

    def prepare(self, messages):
        if self.spec.backend != "custom":
            return self.runner.encode_messages(messages)
        return self.tokenizer.encode_messages(text_messages(messages)), {}

    def decode(self, ids):
        if self.spec.backend != "custom":
            return self.runner.decode_text(ids)
        return self.tokenizer.decode(ids)


class ModelRegistry:
    def __init__(self, specs: list[ModelSpec], settings: Settings):
        self.specs = {spec.id: spec for spec in specs}
        self.settings = settings
        self.current: ModelBundle | None = None
        self.load_times: dict[str, float] = {}

    def unload(self):
        self.current = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_id: str) -> ModelBundle:
        if self.current and self.current.spec.id == model_id:
            return self.current
        self.unload()
        started = time.perf_counter()
        spec, cfg = self.specs[model_id], self.settings
        device = torch.device(cfg.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use MINILLM_DEVICE=cpu for the PyTorch fallback."
            )
        dtype = getattr(torch, cfg.dtype) if device.type != "cpu" else torch.float32
        if spec.backend == "custom":
            from minillm.model.smollm import SmolLMRunner
            from minillm.tokenization import HFTokenizer, SmolLMTokenizer

            tokenizer = (
                SmolLMTokenizer if spec.tokenizer == "custom" else HFTokenizer
            )(spec.path)
            runner = SmolLMRunner(
                spec.path,
                device=device,
                dtype=dtype,
                cache_layout=cfg.cache_layout,
                cache_dtype=cfg.cache_dtype,
                attention_backend=cfg.attention_backend,
                max_context=cfg.max_context,
                max_requests=cfg.max_requests,
                cache_memory_mb=cfg.cache_memory_mb,
                prefix_cache=cfg.prefix_cache,
            )
            capacity = runner.capacity_tokens
        else:
            runner = HFRunner(
                spec.path,
                device=device,
                dtype=dtype,
                max_context=cfg.max_context,
                multimodal=spec.backend == "multimodal",
            )
            tokenizer = None
            # Gemma has heterogeneous layer dimensions. Use the largest configured
            # dimensions for conservative admission, without global attribute access.
            config = runner.model.config.get_text_config()
            kv_elements = 0
            for i in range(config.num_hidden_layers):
                layer = config.per_layer_config[i]
                head_dim = (
                    getattr(layer, "head_dim", None)
                    or layer.hidden_size // layer.num_attention_heads
                )
                kv_elements += layer.num_key_value_heads * head_dim
            kv_bytes = 2 * kv_elements * runner.model.dtype.itemsize
            capacity = max(1, cfg.cache_memory_mb * 1024 * 1024 // kv_bytes)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        elapsed = time.perf_counter() - started
        self.load_times[model_id] = elapsed
        self.current = ModelBundle(spec, runner, tokenizer, capacity, elapsed)
        return self.current
