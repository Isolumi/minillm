"""One-token online-softmax attention over contiguous or paged KV storage."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _decode(Q, K, V, KS, VS, TABLE, LENGTH, OUT,
                Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, DIM: tl.constexpr,
                PAGE: tl.constexpr, TABLE_WIDTH: tl.constexpr, QUANTIZED: tl.constexpr,
                TILE: tl.constexpr, WIDTH: tl.constexpr):
        batch = tl.program_id(0)
        head = tl.program_id(1)
        kv_head = head // (Q_HEADS // KV_HEADS)
        dims = tl.arange(0, WIDTH)
        offsets = tl.arange(0, TILE)
        length = tl.load(LENGTH + batch)
        query = tl.load(Q + (batch * Q_HEADS + head) * DIM + dims, dims < DIM, 0).to(tl.float32)
        maximum = tl.full((), -float("inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)
        accumulator = tl.zeros((WIDTH,), tl.float32)
        for tile in range(tl.cdiv(length, TILE)):
            tokens = tile * TILE + offsets
            valid = tokens < length
            blocks = tl.load(TABLE + batch * TABLE_WIDTH + tokens // PAGE, valid, 0)
            rows = (blocks * PAGE + tokens % PAGE) * KV_HEADS + kv_head
            key = tl.load(K + rows[:, None] * DIM + dims[None, :], valid[:, None] & (dims[None, :] < DIM), 0).to(tl.float32)
            value = tl.load(V + rows[:, None] * DIM + dims[None, :], valid[:, None] & (dims[None, :] < DIM), 0).to(tl.float32)
            if QUANTIZED:
                key *= tl.load(KS + rows, valid, 0)[:, None]
                value *= tl.load(VS + rows, valid, 0)[:, None]
            score = tl.sum(key * query[None, :], 1) * (DIM ** -0.5)
            score = tl.where(valid, score, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(score, 0))
            correction = tl.exp(maximum - next_maximum)
            probability = tl.exp(score - next_maximum)
            accumulator = accumulator * correction + tl.sum(probability[:, None] * value, 0)
            denominator = denominator * correction + tl.sum(probability, 0)
            maximum = next_maximum
        tl.store(OUT + (batch * Q_HEADS + head) * DIM + dims, accumulator / denominator, dims < DIM)


def available(device):
    return triton is not None and torch.device(device).type == "cuda"


def paged_decode_attention(query, cache, layer, states, metadata=None):
    """Query [batch, heads, dim]; the kernel never gathers the full KV prefix."""
    if not available(query.device):
        raise RuntimeError("Triton decode requires Triton and a CUDA device")
    if query.ndim != 3 or len(states) != query.shape[0]:
        raise ValueError("Expected one query per cache state")
    tables, lengths = cache.metadata(states) if metadata is None else metadata
    query = query.contiguous()
    output = torch.empty_like(query)
    key, value = cache.keys[layer], cache.values[layer]
    key_scales = cache.key_scales[layer] if cache.quantized else key
    value_scales = cache.value_scales[layer] if cache.quantized else value
    _decode[(query.shape[0], query.shape[1])](
        query, key, value, key_scales, value_scales, tables, lengths, output,
        Q_HEADS=query.shape[1], KV_HEADS=key.shape[-2], DIM=query.shape[-1],
        PAGE=cache.block_size, TABLE_WIDTH=tables.shape[1], QUANTIZED=cache.quantized,
        TILE=32, WIDTH=triton.next_power_of_2(query.shape[-1]), num_warps=4)
    return output
