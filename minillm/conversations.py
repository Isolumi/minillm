import threading
from dataclasses import dataclass, field
from uuid import uuid4

from transformers import DynamicCache


@dataclass
class Conversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    cache: DynamicCache | None = None
    cached_ids: list[int] = field(default_factory=list)


class ConversationStore:
    def __init__(self, max_conversations: int = 2):
        self._conversations: dict[str, Conversation] = {}
        self._max_conversations = max_conversations
        self._lock = threading.Lock()

    def create(self) -> tuple[str, Conversation]:
        with self._lock:
            if len(self._conversations) >= self._max_conversations:
                raise ConversationLimitError

            conversation_id = str(uuid4())
            conversation = Conversation()
            self._conversations[conversation_id] = conversation

            return conversation_id, conversation

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, conversation: Conversation) -> None:
        with self._lock:
            self._conversations[conversation_id] = conversation

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None


class ConversationLimitError(Exception):
    pass
