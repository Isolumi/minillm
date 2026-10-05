"""Minimal Llama-style SmolLM2 inference, loaded directly from safetensors.

No transformers model or DynamicCache is involved. The scheduler owns request
lifetimes; forward failure invalidates its affected states, which must be released.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from minillm.attention import apply_rope, decode_attention, prefill_attention
from minillm.cache import ContiguousKVCache, PagedKVCache
from minillm.kernels import available as triton_available
from minillm.kernels import paged_decode_attention


def rms_norm(x, weight, epsilon):
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + epsilon
    )
    return normalized.to(x.dtype) * weight


class SmolLMRunner:
    def __init__(
        self,
        model_path,
        device="cuda",
        dtype=torch.float16,
        cache_layout="paged",
        cache_dtype="auto",
        attention_backend="auto",
        max_context=8192,
        max_requests=8,
        cache_memory_mb=2048,
        block_size=16,
        prefix_cache=True,
    ):
        self.model_path = Path(model_path).resolve()
        self.config = json.loads((self.model_path / "config.json").read_text())
        config = self.config
        if (
            config.get("model_type") != "llama"
            or config.get("hidden_act", "silu") != "silu"
            or config.get("rope_scaling")
            or config.get("attention_bias", False)
            or config.get("mlp_bias", False)
            or config.get("pretraining_tp", 1) != 1
        ):
            raise ValueError(
                "Custom runner supports SmolLM2 Llama configs with SiLU, plain RoPE, and bias-free projections"
            )
        if isinstance(dtype, str):
            dtype = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(dtype)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Model dtype must be float16, bfloat16, or float32")
        self.device, self.dtype = torch.device(device), dtype
        self.context_limit = min(max_context, config["max_position_embeddings"])
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        if (
            self.hidden_size != self.num_heads * self.head_dim
            or self.num_heads % self.num_kv_heads
        ):
            raise ValueError("Invalid attention head dimensions")
        self.epsilon = config["rms_norm_eps"]
        self.rope_theta = config.get("rope_theta", 10000.0)
        eos = config.get("eos_token_id", [])
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        generation_path = self.model_path / "generation_config.json"
        if generation_path.exists():
            eos = json.loads(generation_path.read_text()).get("eos_token_id", [])
            self.eos_token_ids.update((eos if isinstance(eos, list) else [eos]))
            self.eos_token_ids.discard(None)
        if attention_backend not in ("auto", "torch", "triton"):
            raise ValueError("attention_backend must be auto, torch, or triton")
        if attention_backend == "triton" and not triton_available(self.device):
            raise ValueError("Triton attention requested without Triton/CUDA support")
        self.attention_backend = (
            ("triton" if triton_available(self.device) else "torch")
            if attention_backend == "auto"
            else attention_backend
        )
        if cache_layout not in ("paged", "contiguous"):
            raise ValueError("cache_layout must be paged or contiguous")
        self.cache_layout = cache_layout
        self.weights = self._load_weights()
        self.embedding = self.weights["model.embed_tokens.weight"]
        self.output_weight = (
            self.embedding
            if config.get("tie_word_embeddings", False)
            else self.weights["lm_head.weight"]
        )
        cache_class = PagedKVCache if cache_layout == "paged" else ContiguousKVCache
        self.cache = cache_class(
            config["num_hidden_layers"],
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=dtype,
            cache_dtype=cache_dtype,
            max_context=self.context_limit,
            max_requests=max_requests,
            memory_mb=cache_memory_mb,
            block_size=block_size,
            prefix_cache=prefix_cache,
            model_id=str(self.model_path),
        )
        self.capacity_tokens = self.cache.num_blocks * self.cache.block_size
        self.max_requests = (
            min(max_requests, self.cache.num_blocks)
            if cache_layout == "contiguous"
            else max_requests
        )
        self.prefill_tokens = 0
        self.decode_tokens = 0

    def _load_weights(self):
        config = self.config
        hidden, intermediate, vocab = (
            self.hidden_size,
            config["intermediate_size"],
            config["vocab_size"],
        )
        shapes = {
            "model.embed_tokens.weight": (vocab, hidden),
            "model.norm.weight": (hidden,),
        }
        if not config.get("tie_word_embeddings", False):
            shapes["lm_head.weight"] = (vocab, hidden)
        for layer in range(config["num_hidden_layers"]):
            prefix = f"model.layers.{layer}."
            shapes.update(
                {
                    prefix + "input_layernorm.weight": (hidden,),
                    prefix + "post_attention_layernorm.weight": (hidden,),
                    prefix + "self_attn.q_proj.weight": (
                        self.num_heads * self.head_dim,
                        hidden,
                    ),
                    prefix + "self_attn.k_proj.weight": (
                        self.num_kv_heads * self.head_dim,
                        hidden,
                    ),
                    prefix + "self_attn.v_proj.weight": (
                        self.num_kv_heads * self.head_dim,
                        hidden,
                    ),
                    prefix + "self_attn.o_proj.weight": (hidden, hidden),
                    prefix + "mlp.gate_proj.weight": (intermediate, hidden),
                    prefix + "mlp.up_proj.weight": (intermediate, hidden),
                    prefix + "mlp.down_proj.weight": (hidden, intermediate),
                }
            )
        index = self.model_path / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            paths = sorted(
                {self.model_path / mapping[name] for name in shapes if name in mapping}
            )
        else:
            paths = [self.model_path / "model.safetensors"]
        weights = {}
        for path in paths:
            if path.resolve().parent != self.model_path:
                raise ValueError("Checkpoint shards must reside in the model directory")
            with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    if name not in shapes:
                        continue
                    tensor = checkpoint.get_tensor(name)
                    if tuple(tensor.shape) != shapes[name]:
                        raise ValueError(
                            f"Unexpected checkpoint shape for {name}: {tuple(tensor.shape)}, expected {shapes[name]}"
                        )
                    if not tensor.is_floating_point():
                        raise ValueError(
                            f"Unsupported checkpoint tensor dtype for {name}"
                        )
                    weights[name] = tensor.to(device=self.device, dtype=self.dtype)
        missing = shapes.keys() - weights.keys()
        if missing:
            raise ValueError(f"Missing checkpoint weights: {sorted(missing)}")
        return weights

    def create_state(self):
        return self.cache.allocate()

    def release(self, state):
        self.cache.release(state)

    def reuse_prefix(self, token_ids, state):
        if self.cache_layout == "contiguous":
            return 0
        return self.cache.reuse_prefix(token_ids, state)

    def publish_prefix(self, token_ids, state):
        self.cache.publish_prefix(token_ids, state)

    def _validate_tokens(self, token_ids):
        if not token_ids or any(
            not isinstance(token, int)
            or token < 0
            or token >= self.config["vocab_size"]
            for token in token_ids
        ):
            raise ValueError("Expected nonempty token IDs within the model vocabulary")

    @torch.inference_mode()
    def prefill(self, token_ids, state):
        self._validate_tokens(token_ids)
        start = state.length
        self.cache.reserve_batch([state], [len(token_ids)])
        ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        positions = torch.arange(start, state.length, device=self.device)[None, :]
        logits = self._forward(ids, positions, [state], [start], decode=False)
        self.prefill_tokens += len(token_ids)
        return logits[0]

    @torch.inference_mode()
    def decode(self, token_ids, states):
        self._validate_tokens(token_ids)
        if len(token_ids) != len(states):
            raise ValueError("Decode requires one token for every cache state")
        starts = [state.length for state in states]
        self.cache.reserve_batch(states, [1] * len(states))
        ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)[:, None]
        positions = torch.tensor(starts, device=self.device, dtype=torch.long)[:, None]
        logits = self._forward(ids, positions, states, starts, decode=True)
        self.decode_tokens += len(states)
        return logits

    def _forward(self, ids, positions, states, starts, *, decode):
        x = F.embedding(ids, self.embedding)
        batch, sequence = ids.shape
        metadata = (
            self.cache.metadata(states)
            if decode and self.attention_backend == "triton"
            else None
        )
        for layer in range(self.config["num_hidden_layers"]):
            prefix = f"model.layers.{layer}."
            weights = self.weights
            residual = x
            normed = rms_norm(
                x, weights[prefix + "input_layernorm.weight"], self.epsilon
            )
            q = F.linear(normed, weights[prefix + "self_attn.q_proj.weight"]).view(
                batch, sequence, self.num_heads, self.head_dim
            )
            k = F.linear(normed, weights[prefix + "self_attn.k_proj.weight"]).view(
                batch, sequence, self.num_kv_heads, self.head_dim
            )
            v = F.linear(normed, weights[prefix + "self_attn.v_proj.weight"]).view(
                batch, sequence, self.num_kv_heads, self.head_dim
            )
            q, k = apply_rope(q, k, positions, self.rope_theta)
            for row, state in enumerate(states):
                self.cache.write(layer, state, starts[row], k[row], v[row])
            if decode and self.attention_backend == "triton":
                attended = paged_decode_attention(
                    q[:, 0], self.cache, layer, states, metadata
                )[:, None]
            else:
                rows = []
                for row, state in enumerate(states):
                    cached_key, cached_value = self.cache.read(layer, state)
                    cached_key, cached_value = (
                        cached_key.to(self.dtype),
                        cached_value.to(self.dtype),
                    )
                    if decode:
                        result = decode_attention(
                            q[row : row + 1], cached_key[None], cached_value[None]
                        )
                    else:
                        result = prefill_attention(
                            q[row : row + 1],
                            cached_key[None],
                            cached_value[None],
                            starts[row],
                        )
                    rows.append(result)
                attended = torch.cat(rows, dim=0)
            x = residual + F.linear(
                attended.reshape(batch, sequence, self.hidden_size),
                weights[prefix + "self_attn.o_proj.weight"],
            )
            residual = x
            normed = rms_norm(
                x, weights[prefix + "post_attention_layernorm.weight"], self.epsilon
            )
            gate = F.silu(F.linear(normed, weights[prefix + "mlp.gate_proj.weight"]))
            up = F.linear(normed, weights[prefix + "mlp.up_proj.weight"])
            x = residual + F.linear(gate * up, weights[prefix + "mlp.down_proj.weight"])
        x = rms_norm(x[:, -1], self.weights["model.norm.weight"], self.epsilon)
        return F.linear(x, self.output_weight).float()

    def stats(self):
        return {
            "backend": "custom",
            "attention_backend": self.attention_backend,
            "capacity_tokens": self.capacity_tokens,
            "context_limit": self.context_limit,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "cache": self.cache.stats(),
        }
