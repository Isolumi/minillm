import math

import torch

def reference_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor
) -> torch.Tensor:
    # q: [B, Tq, Hq, D]
    # k/v: [B, Tk, Hkv, D]
    
    batch, query_tokens, query_heads, head_dim = q.shape
    key_batch, key_tokens, kv_heads, key_dim = k.shape

    assert batch == key_batch
    assert head_dim == key_dim
    assert v.shape == k.shape
    assert query_heads % kv_heads == 0
    assert len(query_positions) == query_tokens
    assert len(key_positions) == key_tokens

    repeats = query_heads // kv_heads
    k = k.repeat_interleave(repeats, dim=2)
    v = v.repeat_interleave(repeats, dim=2)

    q = q.transpose(1, 2).float()
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()
