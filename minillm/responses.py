import threading
from collections import OrderedDict
from dataclasses import dataclass

from minillm.schemas import ModelResponse


@dataclass(frozen=True)
class StoredResponse:
    response: ModelResponse
    messages: tuple[tuple[str, str], ...]


class ResponseStore:
    def __init__(self, max_responses: int = 128) -> None:
        self._responses: OrderedDict[str, StoredResponse] = OrderedDict()
        self._max_responses = max_responses
        self._lock = threading.Lock()

    def get(self, response_id: str) -> StoredResponse | None:
        with self._lock:
            return self._responses.get(response_id)

    def save(self, stored: StoredResponse) -> None:
        with self._lock:
            self._responses[stored.response.id] = stored
            if len(self._responses) > self._max_responses:
                self._responses.popitem(last=False)
