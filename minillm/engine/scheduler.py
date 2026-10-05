"""Bounded continuous batching with a single GPU-owning worker thread."""

import copy
import logging
import math
import queue
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from minillm.config import ModelSpec, Settings, model_specs
from minillm.engine.registry import ModelRegistry

log = logging.getLogger(__name__)


@dataclass
class GenerationHandle:
    request_id: str
    events: queue.Queue
    cancelled: threading.Event = field(default_factory=threading.Event)

    def cancel(self):
        self.cancelled.set()


@dataclass
class Request:
    handle: GenerationHandle
    model: str
    messages: list[dict]
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    submitted: float = field(default_factory=time.perf_counter)
    started: float = 0.0
    first_token: float = 0.0
    input_ids: list[int] | None = None
    initial_inputs: dict = field(default_factory=dict)
    output_ids: list[int] = field(default_factory=list)
    prefill_position: int = 0
    cached_tokens: int = 0
    state: object = None
    generator: object = None
    emitted: str = ""
    finished: bool = False
    prefill_seconds: float = 0.0
    decode_seconds: list[float] = field(default_factory=list)

    @property
    def reservation(self):
        # Include block rounding and the generated terminal token conservatively.
        return math.ceil((len(self.input_ids or []) + self.max_new_tokens) / 16) * 16


class Engine:
    def __init__(
        self, settings: Settings | None = None, specs: list[ModelSpec] | None = None
    ):
        self.settings = settings or Settings.from_env()
        self.registry = ModelRegistry(specs or model_specs(), self.settings)
        self._condition = threading.Condition()
        self._waiting: deque[Request] = deque()
        self._handles: dict[str, GenerationHandle] = {}
        self._active: list[Request] = []
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._turn_requests = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._generated = 0
        self._trace: deque[dict] = deque(maxlen=64)
        self._snapshot = {"status": "starting", "resident_model": None}
        self._created = int(time.time())

    def start(self):
        with self._condition:
            if self._thread:
                return
            self._snapshot["status"] = "ready"
            self._thread = threading.Thread(
                target=self._run, name="minillm-gpu", daemon=True
            )
            self._thread.start()

    def close(self):
        with self._condition:
            self._stopping = True
            for handle in self._handles.values():
                handle.cancel()
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=30)

    def models(self):
        with self._condition:
            resident = self._snapshot.get("resident_model")
        return [
            {
                "id": s.id,
                "object": "model",
                "created": self._created,
                "owned_by": "minillm",
                "backend": s.backend,
                "modalities": ["text", "image"]
                if s.backend == "multimodal"
                else ["text"],
                "resident": s.id == resident,
            }
            for s in self.registry.specs.values()
        ]

    def stats(self):
        with self._condition:
            return copy.deepcopy(self._snapshot)

    def submit(
        self,
        request_id: str,
        model: str,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float = 0,
        top_p: float = 1,
        seed: int | None = None,
    ):
        if model not in self.registry.specs:
            raise KeyError(f"Unknown model: {model}")
        if not 1 <= max_new_tokens <= 8192:
            raise ValueError("max_output_tokens must be between 1 and 8192")
        if (
            not math.isfinite(temperature)
            or not 0 <= temperature <= 2
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError("Invalid temperature or top_p")
        if not messages:
            raise ValueError("Input cannot be empty")
        text_bytes = 0
        for message in messages:
            content = message["content"]
            texts = (
                [content]
                if isinstance(content, str)
                else [p.get("text", "") for p in content]
            )
            text_bytes += sum(len(text.encode("utf-8")) for text in texts)
        if text_bytes > self.settings.max_context * 32:
            raise ValueError(
                f"Text input exceeds the {self.settings.max_context * 32}-byte limit; shorten the conversation"
            )
        handle = GenerationHandle(request_id, queue.Queue(maxsize=max_new_tokens + 8))
        request = Request(
            handle,
            model,
            copy.deepcopy(messages),
            max_new_tokens,
            temperature,
            top_p,
            seed if seed is not None else secrets.randbits(63),
        )
        with self._condition:
            if self._stopping or not self._thread or not self._thread.is_alive():
                raise RuntimeError("Engine is not running")
            if request_id in self._handles:
                raise ValueError("Request ID already exists")
            if (
                len(self._handles)
                >= self.settings.max_queue + self.settings.max_requests
            ):
                raise RuntimeError(
                    "Request queue is full; retry when an active request finishes"
                )
            self._handles[request_id] = handle
            self._waiting.append(request)
            self._condition.notify()
        return handle

    def cancel(self, request_id: str) -> bool:
        with self._condition:
            handle = self._handles.get(request_id)
            if handle:
                handle.cancel()
                self._condition.notify()
            return handle is not None

    def _record(self, kind, request=None, **fields):
        self._trace.append(
            {
                "time": time.time(),
                "event": kind,
                **(
                    {"request_id": request.handle.request_id, "model": request.model}
                    if request
                    else {}
                ),
                **fields,
            }
        )

    def _forget(self, request):
        request.finished = True
        with self._condition:
            self._handles.pop(request.handle.request_id, None)
        if request.state is not None:
            try:
                self.registry.current.runner.release(request.state)
            except Exception:
                log.exception("Failed to release request cache")
            request.state = None
        request.initial_inputs.clear()
        request.messages.clear()

    def _error(self, request, exc, code="generation_error"):
        if request.finished:
            return
        self._failed += 1
        self._record("error", request, code=code)
        self._forget(request)
        request.handle.events.put_nowait(
            {"type": "error", "message": str(exc), "code": code}
        )

    def _finish(self, request, reason):
        if request.finished:
            return
        bundle = self.registry.current
        text = bundle.decode(request.output_ids) if request.output_ids else ""
        if not text.startswith(request.emitted):
            self._error(
                request, "Tokenizer revised already streamed text", "decoding_error"
            )
            return
        delta = text[len(request.emitted) :]
        if delta:
            request.handle.events.put_nowait({"type": "delta", "text": delta})
        now = time.perf_counter()
        event = {
            "type": "done",
            "text": text,
            "prompt_tokens": len(request.input_ids or []),
            "completion_tokens": len(request.output_ids),
            "finish_reason": reason,
            "cached_tokens": request.cached_tokens,
            "token_ids": request.output_ids,
            "cache_write_tokens": max(
                0, request.prefill_position - request.cached_tokens
            ),
            "timing": {
                "queue_seconds": max(0, (request.started or now) - request.submitted),
                "ttft_seconds": request.first_token - request.submitted
                if request.first_token
                else None,
                "prefill_seconds": request.prefill_seconds,
                "decode_step_seconds": request.decode_seconds,
                "total_seconds": now - request.submitted,
            },
        }
        self._cancelled += reason == "cancelled"
        self._completed += reason != "cancelled"
        self._record(
            "finish", request, reason=reason, output_tokens=len(request.output_ids)
        )
        self._forget(request)
        request.handle.events.put_nowait(event)

    def _sample(self, request, logits):
        if request.handle.cancelled.is_set():
            self._finish(request, "cancelled")
            return
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model returned non-finite logits")
        if request.temperature == 0:
            token = int(logits.argmax())
        else:
            probabilities = torch.softmax(logits / request.temperature, dim=-1)
            if request.top_p < 1:
                probabilities, indices = probabilities.sort(descending=True)
                remove = probabilities.cumsum(-1) - probabilities >= request.top_p
                probabilities.masked_fill_(remove, 0)
                index = torch.multinomial(probabilities, 1, generator=request.generator)
                token = int(indices[index].item())
            else:
                token = int(
                    torch.multinomial(
                        probabilities, 1, generator=request.generator
                    ).item()
                )
        request.output_ids.append(token)
        self._generated += 1
        if not request.first_token:
            request.first_token = time.perf_counter()
            self._record("first_token", request)
        if token in self.registry.current.runner.eos_token_ids:
            self._finish(request, "stop")
        elif len(request.output_ids) >= request.max_new_tokens:
            self._finish(request, "length")
        else:
            text = self.registry.current.decode(request.output_ids)
            # Flush complete words/lines. Incomplete byte tokens may decode as U+FFFD;
            # retaining the current word prevents sending text that later changes.
            boundary = max(text.rfind(" "), text.rfind("\n")) + 1
            stable = text[:boundary]
            if stable.startswith(request.emitted) and len(stable) > len(
                request.emitted
            ):
                request.handle.events.put_nowait(
                    {"type": "delta", "text": stable[len(request.emitted) :]}
                )
                request.emitted = stable

    def _prepare(self, request, bundle):
        if request.input_ids is None:
            request.input_ids, request.initial_inputs = bundle.prepare(request.messages)
            if not request.input_ids:
                raise ValueError("Prompt produced no tokens")
            if (
                len(request.input_ids) + request.max_new_tokens
                > bundle.runner.context_limit
            ):
                raise ValueError(
                    f"Prompt ({len(request.input_ids)}) plus output budget ({request.max_new_tokens}) exceeds the {bundle.runner.context_limit}-token context limit"
                )
            if request.reservation > bundle.capacity_tokens:
                raise ValueError(
                    f"Request exceeds the configured KV memory budget ({bundle.capacity_tokens} tokens); reduce input/output or increase MINILLM_CACHE_MB"
                )

    def _admit(self, bundle):
        with self._condition:
            candidates = list(self._waiting)
        other_waiting = any(
            r.model != bundle.spec.id and not r.handle.cancelled.is_set()
            for r in candidates
        )
        reserved = sum(r.reservation for r in self._active if not r.finished)
        for request in candidates:
            if len(self._active) >= min(
                self.settings.max_requests,
                getattr(bundle.runner, "max_requests", self.settings.max_requests),
            ):
                break
            if (
                other_waiting
                and self._turn_requests >= self.settings.max_model_turn_requests
            ):
                break
            if request.model != bundle.spec.id:
                continue
            try:
                self._prepare(request, bundle)
                if reserved + request.reservation > bundle.capacity_tokens:
                    continue
                request.state = bundle.runner.create_state()
                if request.initial_inputs:
                    request.state.initial_inputs = request.initial_inputs
                request.generator = torch.Generator(
                    device=bundle.runner.device
                ).manual_seed(request.seed)
                request.started = time.perf_counter()
                if hasattr(bundle.runner, "reuse_prefix"):
                    request.cached_tokens = bundle.runner.reuse_prefix(
                        request.input_ids, request.state
                    )
                    request.prefill_position = request.cached_tokens
                reserved += request.reservation
                self._active.append(request)
                self._turn_requests += 1
                self._record(
                    "admit",
                    request,
                    prompt_tokens=len(request.input_ids),
                    cached_tokens=request.cached_tokens,
                )
            except (ValueError, MemoryError) as exc:
                self._error(
                    request,
                    exc,
                    "invalid_request"
                    if isinstance(exc, ValueError)
                    else "capacity_exceeded",
                )
            except Exception as exc:
                log.exception("Request preparation failed")
                self._error(request, exc)
            with self._condition:
                self._waiting.remove(request)

    def _iteration(self):
        with self._condition:
            cancelled = [r for r in self._waiting if r.handle.cancelled.is_set()]
            for request in cancelled:
                self._waiting.remove(request)
        for request in cancelled:
            self._finish(request, "cancelled")
        for request in self._active:
            if not request.finished and request.handle.cancelled.is_set():
                self._finish(request, "cancelled")
        self._active = [r for r in self._active if not r.finished]
        if not self._active:
            with self._condition:
                if not self._waiting:
                    return
                current_id = (
                    self.registry.current.spec.id if self.registry.current else None
                )
                next_request = self._waiting[0]
                if (
                    current_id
                    and self._turn_requests >= self.settings.max_model_turn_requests
                ):
                    next_request = next(
                        (r for r in self._waiting if r.model != current_id),
                        next_request,
                    )
                model_id = next_request.model
            self._turn_requests = 0
            try:
                with self._condition:
                    self._snapshot.update(status="loading", loading_model=model_id)
                self.registry.load(model_id)
            except Exception as exc:
                log.exception("Model loading failed")
                with self._condition:
                    failures = [r for r in self._waiting if r.model == model_id]
                    self._waiting = deque(
                        r for r in self._waiting if r.model != model_id
                    )
                for request in failures:
                    self._error(
                        request,
                        f"Could not load {model_id}: {exc}",
                        "model_load_failed",
                    )
                return
        bundle = self.registry.current
        self._admit(bundle)
        # Decode every ready request before admitting one chunk of prefill work.
        ready = [r for r in self._active if not r.finished and r.output_ids]
        if ready:
            try:
                started = time.perf_counter()
                logits = bundle.runner.decode(
                    [r.output_ids[-1] for r in ready], [r.state for r in ready]
                )
                if bundle.runner.device.type == "cuda":
                    torch.cuda.synchronize(bundle.runner.device)
                elapsed = time.perf_counter() - started
                self._record(
                    "decode_batch", batch_size=len(ready), model=bundle.spec.id
                )
                for request, row in zip(ready, logits, strict=True):
                    request.decode_seconds.append(elapsed)
                    self._sample(request, row)
            except Exception as exc:
                log.exception("Decode batch failed")
                for request in ready:
                    self._error(
                        request,
                        exc,
                        "capacity_exceeded"
                        if isinstance(exc, MemoryError)
                        else "generation_error",
                    )
        prefilling = next(
            (r for r in self._active if not r.finished and not r.output_ids), None
        )
        if prefilling:
            request = prefilling
            try:
                start = request.prefill_position
                # Multimodal processor tensors align to the entire image-bearing prompt.
                chunk = (
                    len(request.input_ids)
                    if bundle.spec.backend == "multimodal"
                    else self.settings.prefill_chunk
                )
                end = min(start + chunk, len(request.input_ids))
                started = time.perf_counter()
                logits = bundle.runner.prefill(
                    request.input_ids[start:end], request.state
                )
                if bundle.runner.device.type == "cuda":
                    torch.cuda.synchronize(bundle.runner.device)
                request.prefill_seconds += time.perf_counter() - started
                request.prefill_position = end
                self._record("prefill_chunk", request, start=start, end=end)
                if end == len(request.input_ids):
                    if hasattr(bundle.runner, "publish_prefix"):
                        bundle.runner.publish_prefix(request.input_ids, request.state)
                    self._sample(request, logits)
            except Exception as exc:
                log.exception("Prefill failed")
                self._error(
                    request,
                    exc,
                    "capacity_exceeded"
                    if isinstance(exc, MemoryError)
                    else "generation_error",
                )
        self._active = [r for r in self._active if not r.finished]

    def _update_snapshot(self):
        bundle = self.registry.current
        device = torch.device(self.settings.device)
        gpu = {}
        if device.type == "cuda" and torch.cuda.is_available():
            gpu = {
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        with self._condition:
            self._snapshot = {
                "status": "stopped" if self._stopping else "ready",
                "resident_model": bundle.spec.id if bundle else None,
                "residency_policy": "one model; switch after active requests drain",
                "active_requests": len(self._active),
                "queued_requests": len(self._waiting),
                "completed_requests": self._completed,
                "failed_requests": self._failed,
                "cancelled_requests": self._cancelled,
                "generated_tokens": self._generated,
                "model_load_seconds": dict(self.registry.load_times),
                "cache": bundle.runner.stats() if bundle else {},
                "gpu": gpu,
                "trace": list(self._trace),
            }

    def _run(self):
        try:
            with torch.inference_mode():
                while True:
                    with self._condition:
                        if self._stopping and not self._handles:
                            break
                        if not self._waiting and not self._active:
                            self._condition.wait(timeout=0.2)
                    try:
                        self._iteration()
                        self._update_snapshot()
                    except Exception as exc:
                        log.exception("Engine iteration failed")
                        with self._condition:
                            requests = list(self._waiting) + self._active
                            self._waiting.clear()
                        for request in requests:
                            self._error(request, exc, "engine_error")
                        self._active.clear()
        finally:
            self.registry.unload()
            with self._condition:
                self._snapshot["status"] = "stopped"
