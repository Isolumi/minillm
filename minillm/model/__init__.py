"""Common operations implemented by each model runner."""

from typing import Protocol

import torch


class Runner[State](Protocol):
    device: torch.device
    context_limit: int
    eos_token_ids: set[int]

    def create_state(self) -> State: ...

    def prefill(self, token_ids: list[int], state: State) -> torch.Tensor: ...

    def decode(self, token_ids: list[int], states: list[State]) -> torch.Tensor: ...

    def release(self, state: State) -> None: ...

    def stats(self) -> dict: ...
