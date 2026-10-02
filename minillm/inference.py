"""Small synchronous Python facade over the same engine used by HTTP."""
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from minillm.config import ModelSpec, Settings
from minillm.engine.scheduler import Engine


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class ContextLimitExceeded(ValueError):
    pass


class InferenceEngine:
    def __init__(self, model_path: str, *, backend: str = "custom", settings: Settings | None = None):
        self.model_id = Path(model_path).name
        spec = ModelSpec(self.model_id, str(Path(model_path).resolve()), backend,
                         "custom" if backend == "custom" else "hf")
        self.engine = Engine(settings=settings, specs=[spec])
        self.engine.start()

    def generate(self, messages: list[dict], max_new_tokens: int, **sampling) -> GenerationResult:
        handle = self.engine.submit(uuid4().hex, self.model_id, messages, max_new_tokens, **sampling)
        try:
            while True:
                event = handle.events.get()
                if event["type"] == "error":
                    if "context limit" in event["message"]:
                        raise ContextLimitExceeded(event["message"])
                    raise RuntimeError(event["message"])
                if event["type"] == "done":
                    return GenerationResult(event["text"], event["prompt_tokens"], event["completion_tokens"], event["finish_reason"])
        except BaseException:
            handle.cancel()
            raise

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
