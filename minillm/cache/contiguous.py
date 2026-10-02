"""Contiguous request slots using one context-sized block per request."""
from .paged import PagedKVCache


class ContiguousKVCache(PagedKVCache):
    def __init__(self, *args, max_context=8192, **kwargs):
        kwargs.pop("block_size", None)
        kwargs.pop("prefix_cache", None)
        super().__init__(*args, max_context=max_context, block_size=max_context,
                         prefix_cache=False, **kwargs)

    def allocate(self):
        state = super().allocate()
        try:
            self._make_room(1)
            block = self.free.pop()
            self.references[block] = 1
            state.blocks.append(block)
        except BaseException:
            self.release(state)
            raise
        return state

    def read(self, layer, state):
        self._check(state)
        block = state.blocks[0]
        key, value = self.keys[layer, block, :state.length], self.values[layer, block, :state.length]
        if self.quantized:
            key = (key.float() * self.key_scales[layer, block, :state.length]).to(self.dtype)
            value = (value.float() * self.value_scales[layer, block, :state.length]).to(self.dtype)
        return key, value

    def stats(self):
        return {**super().stats(), "layout": "contiguous"}
