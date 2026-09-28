import math

import numpy as np
import torch
from torch import nn


# TODO: batching, multihead, kv cache
class Attention(nn.Module):
    def __init__(self, hidden_size: int, max_sequence_length: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(-1)

        causal_mask = torch.triu(
            torch.ones(
                max_sequence_length,
                max_sequence_length,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x):
        # x: [B, T, D_model]
        T, _ = x.shape

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        logits = query @ key.T
        logits = logits / math.sqrt(self.hidden_size)

        mask = self.causal_mask[:T, :T]

        logits = logits.masked_fill(
            mask,
            torch.finfo(logits.dtype).min,
        )

        attention_weights = self.softmax(logits)

        return attention_weights @ value


attention_layer = Attention(3, 6)

x = torch.rand(5,3)
torch.no_grad()
print(attention_layer.forward(x))
