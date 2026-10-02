"""Attention primitives use [batch, sequence, heads, head_dim] tensors."""
import torch
from torch.nn import functional as F


def apply_rope(query, key, positions, theta=10000.0):
    """Llama split-half RoPE with explicit [sequence] or [batch, sequence] positions."""
    dim = query.shape[-1]
    if dim % 2 or key.shape[-1] != dim:
        raise ValueError("RoPE requires matching, even head dimensions")
    frequency = 1.0 / (theta ** (torch.arange(0, dim, 2, device=query.device).float() / dim))
    angles = positions.to(device=query.device, dtype=torch.float32)[..., None] * frequency
    if angles.ndim == 2:
        angles = angles[None]
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(-2)
    cosine, sine = angles.cos(), angles.sin()
    def rotate(x):
        a, b = x.chunk(2, dim=-1)
        return x * cosine.to(x.dtype) + torch.cat((-b, a), -1) * sine.to(x.dtype)
    return rotate(query), rotate(key)


def _heads(query, key, value):
    if query.shape[-2] % key.shape[-2] or key.shape != value.shape:
        raise ValueError("Query heads must be divisible by KV heads; K and V must match")
    repeats = query.shape[-2] // key.shape[-2]
    if repeats != 1:
        key = key.repeat_interleave(repeats, dim=2)
        value = value.repeat_interleave(repeats, dim=2)
    return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)


def reference_attention(query, key, value, query_positions=None, key_positions=None, causal=True):
    """Explicit FP32 softmax reference, including cached-prefix position offsets."""
    q, k, v = _heads(query, key, value)
    scores = q.float() @ k.float().transpose(-1, -2) / query.shape[-1] ** 0.5
    if causal:
        if query_positions is None:
            query_positions = torch.arange(key.shape[1] - query.shape[1], key.shape[1], device=q.device)
        if key_positions is None:
            key_positions = torch.arange(key.shape[1], device=q.device)
        mask = key_positions[..., None, :] <= query_positions[..., :, None]
        if mask.ndim == 3:
            mask = mask[:, None]
        scores = scores.masked_fill(~mask, float("-inf"))
    return (scores.softmax(-1) @ v.float()).to(query.dtype).transpose(1, 2)


def prefill_attention(query, key, value, position_offset=0):
    """PyTorch SDPA prefill with causal masking aligned to the cached prefix."""
    q, k, v = _heads(query, key, value)
    if position_offset == 0 and query.shape[1] == key.shape[1]:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    positions = torch.arange(query.shape[1], device=q.device) + position_offset
    mask = torch.arange(key.shape[1], device=q.device)[None, :] <= positions[:, None]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)


def decode_attention(query, key, value):
    if query.shape[1] != 1:
        raise ValueError("Decode attention expects exactly one query token")
    q, k, v = _heads(query, key, value)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
