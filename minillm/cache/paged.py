"""A model-owned fixed KV pool with explicit request and prefix ownership."""

import hashlib
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass, field

import torch


@dataclass(eq=False)
class CacheState:
    owner: object
    blocks: list[int] = field(default_factory=list)
    length: int = 0
    released: bool = False


class PagedKVCache:
    def __init__(
        self,
        num_layers,
        num_kv_heads,
        head_dim,
        *,
        device: str | torch.device = "cuda",
        dtype=torch.float16,
        cache_dtype="auto",
        max_context=8192,
        max_requests=8,
        memory_mb=2048,
        block_size=16,
        prefix_cache=True,
        model_id="model",
    ):
        if (
            min(
                num_layers,
                num_kv_heads,
                head_dim,
                max_context,
                max_requests,
                memory_mb,
                block_size,
            )
            <= 0
        ):
            raise ValueError("Cache dimensions and limits must be positive")
        if cache_dtype not in ("auto", "float16", "fp16", "bfloat16", "bf16", "int8"):
            raise ValueError("cache_dtype must be auto, float16, bfloat16, or int8")
        self.dtype, self.device = dtype, torch.device(device)
        self.storage_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "int8": torch.int8,
        }.get(cache_dtype, dtype)
        self.quantized = self.storage_dtype == torch.int8
        self.block_size, self.max_context, self.max_requests = (
            block_size,
            max_context,
            max_requests,
        )
        self.prefix_enabled = prefix_cache
        self.model_id = model_id.encode()
        element_bytes = torch.empty((), dtype=self.storage_dtype).element_size()
        self.bytes_per_block = (
            2
            * num_layers
            * block_size
            * num_kv_heads
            * (head_dim * element_bytes + (4 if self.quantized else 0))
        )
        needed = max_requests * math.ceil(max_context / block_size)
        self.num_blocks = min(needed, int(memory_mb * 1024**2) // self.bytes_per_block)
        if self.num_blocks < 1:
            raise MemoryError("KV memory budget cannot hold even one cache block")
        shape = (num_layers, self.num_blocks, block_size, num_kv_heads, head_dim)
        self.keys = torch.empty(shape, dtype=self.storage_dtype, device=self.device)
        self.values = torch.empty_like(self.keys)
        self.key_scales = self.value_scales = None
        if self.quantized:
            self.key_scales = torch.empty(
                shape[:-1] + (1,), dtype=torch.float32, device=self.device
            )
            self.value_scales = torch.empty_like(self.key_scales)
        self.free = list(reversed(range(self.num_blocks)))
        self.references = [0] * self.num_blocks
        self.states = set()
        self.prefixes = OrderedDict()
        self.prefix_hit_tokens = 0
        self.prefix_evictions = 0

    def _check(self, state):
        if state.owner is not self or state.released or state not in self.states:
            raise ValueError("Cache state is released or belongs to another model")

    def allocate(self):
        if len(self.states) >= self.max_requests:
            raise MemoryError("Maximum active KV cache requests reached")
        state = CacheState(self)
        self.states.add(state)
        return state

    def _drop(self, block):
        self.references[block] -= 1
        if self.references[block] == 0:
            self.free.append(block)

    def _make_room(self, count):
        while len(self.free) < count and self.prefixes:
            _, block = self.prefixes.popitem(last=False)
            self._drop(block)
            self.prefix_evictions += 1
        if len(self.free) < count:
            raise MemoryError(
                f"KV cache exhausted: need {count} blocks, {len(self.free)} free; lower concurrency/context or increase cache_memory_mb"
            )

    def reserve_batch(self, states, counts):
        if len(states) != len(counts) or len(set(states)) != len(states):
            raise ValueError(
                "Cache batch must contain distinct states and matching counts"
            )
        extra = []
        for state, count in zip(states, counts):
            self._check(state)
            if count < 1 or state.length + count > self.max_context:
                raise ValueError(f"Append exceeds context limit {self.max_context}")
            extra.append(
                math.ceil((state.length + count) / self.block_size) - len(state.blocks)
            )
        self._make_room(sum(extra))
        for state, count, blocks in zip(states, counts, extra):
            for _ in range(blocks):
                block = self.free.pop()
                self.references[block] = 1
                state.blocks.append(block)
            state.length += count

    def write(self, layer, state, start, key, value):
        """Write [tokens, KV heads, dim] into space reserved for this append."""
        self._check(state)
        if start < 0 or start + len(key) > state.length or key.shape != value.shape:
            raise ValueError("Invalid KV write range")
        offset = 0
        while offset < len(key):
            position = start + offset
            block = state.blocks[position // self.block_size]
            if self.references[block] != 1:
                raise ValueError("Cannot mutate a shared prefix block")
            within = position % self.block_size
            count = min(len(key) - offset, self.block_size - within)
            target = (layer, block, slice(within, within + count))
            for source, storage, scales in (
                (key, self.keys, self.key_scales),
                (value, self.values, self.value_scales),
            ):
                chunk = source[offset : offset + count]
                if self.quantized:
                    assert scales is not None
                    scale = (
                        chunk.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                        / 127
                    )
                    storage[target] = (
                        (chunk.float() / scale).round().clamp(-127, 127).to(torch.int8)
                    )
                    scales[target] = scale
                else:
                    storage[target] = chunk
            offset += count

    def read(self, layer, state):
        self._check(state)
        blocks = torch.tensor(state.blocks, device=self.device, dtype=torch.long)
        key = self.keys[layer].index_select(0, blocks).flatten(0, 1)[: state.length]
        value = self.values[layer].index_select(0, blocks).flatten(0, 1)[: state.length]
        if self.quantized:
            assert self.key_scales is not None and self.value_scales is not None
            ks = (
                self.key_scales[layer]
                .index_select(0, blocks)
                .flatten(0, 1)[: state.length]
            )
            vs = (
                self.value_scales[layer]
                .index_select(0, blocks)
                .flatten(0, 1)[: state.length]
            )
            key, value = (
                (key.float() * ks).to(self.dtype),
                (value.float() * vs).to(self.dtype),
            )
        return key, value

    def metadata(self, states):
        width = max(len(state.blocks) for state in states)
        tables = [state.blocks + [0] * (width - len(state.blocks)) for state in states]
        return (
            torch.tensor(tables, dtype=torch.int32, device=self.device),
            torch.tensor(
                [state.length for state in states],
                dtype=torch.int32,
                device=self.device,
            ),
        )

    def _prefix_keys(self, token_ids):
        digest = hashlib.sha256(self.model_id).digest()
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            block = token_ids[start : start + self.block_size]
            digest = hashlib.sha256(
                digest + struct.pack(f"<{len(block)}q", *block)
            ).digest()
            yield digest

    def reuse_prefix(self, token_ids, state):
        self._check(state)
        if not self.prefix_enabled:
            return 0
        if state.length or state.blocks:
            raise ValueError("Prefix reuse requires an empty state")
        if len(token_ids) > self.max_context:
            raise ValueError(f"Prompt exceeds context limit {self.max_context}")
        for key in self._prefix_keys(token_ids[:-1]):
            block = self.prefixes.get(key)
            if block is None:
                break
            state.blocks.append(block)
            self.references[block] += 1
            self.prefixes.move_to_end(key)
            state.length += self.block_size
        self.prefix_hit_tokens += state.length
        return state.length

    def publish_prefix(self, token_ids, state):
        self._check(state)
        if not self.prefix_enabled:
            return
        for index, key in enumerate(self._prefix_keys(token_ids[: state.length])):
            if key not in self.prefixes:
                block = state.blocks[index]
                self.references[block] += 1
                self.prefixes[key] = block
            self.prefixes.move_to_end(key)

    def release(self, state):
        if state.owner is not self:
            raise ValueError("Cache state belongs to another model")
        if state.released:
            return
        for block in state.blocks:
            self._drop(block)
        state.blocks.clear()
        state.released = True
        self.states.remove(state)

    def stats(self):
        used = self.num_blocks - len(self.free)
        return {
            "layout": "paged",
            "cache_dtype": str(self.storage_dtype).removeprefix("torch."),
            "allocated_bytes": self.num_blocks * self.bytes_per_block,
            "used_bytes": used * self.bytes_per_block,
            "bytes_per_token": self.bytes_per_block // self.block_size,
            "num_blocks": self.num_blocks,
            "free_blocks": len(self.free),
            "used_blocks": used,
            "block_size": self.block_size,
            "active_requests": len(self.states),
            "capacity_tokens": self.num_blocks * self.block_size,
            "active_tokens": sum(state.length for state in self.states),
            "prefix_blocks": len(self.prefixes),
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "prefix_evictions": self.prefix_evictions,
        }
