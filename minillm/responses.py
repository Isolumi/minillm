"""Bounded in-memory Responses history. Stored messages retain typed image parts."""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from minillm.schemas import ModelResponse


@dataclass(frozen=True)
class StoredResponse:
    response: ModelResponse
    messages: tuple[dict[str, Any], ...]


class ResponseStore:
    def __init__(self, max_responses: int = 128, max_bytes: int = 64 * 1024 * 1024) -> None:
        self._responses: OrderedDict[str, tuple[StoredResponse, int]] = OrderedDict()
        self._max_responses = max_responses
        self._max_bytes = max_bytes
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, response_id: str) -> StoredResponse | None:
        with self._lock:
            entry = self._responses.get(response_id)
            return entry[0] if entry is not None else None

    def save(self, stored: StoredResponse) -> bool:
        # Account for duplicated history across follow-up responses, including images.
        size = len(stored.response.model_dump_json().encode("utf-8")) + sum(
            len(str(message).encode("utf-8")) for message in stored.messages
        )
        if size > self._max_bytes:
            return False
        with self._lock:
            old = self._responses.pop(stored.response.id, None)
            if old is not None:
                self._bytes -= old[1]
            self._responses[stored.response.id] = (stored, size)
            self._bytes += size
            while len(self._responses) > self._max_responses or self._bytes > self._max_bytes:
                _, (_, removed_size) = self._responses.popitem(last=False)
                self._bytes -= removed_size
        return True

    def delete(self, response_id: str) -> bool:
        with self._lock:
            old = self._responses.pop(response_id, None)
            if old is None:
                return False
            self._bytes -= old[1]
            return True
