"""FastAPI adapter for the bounded, asynchronous MiniLLM engine."""

import asyncio
import base64
import binascii
import json
import queue
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from minillm.responses import ResponseStore, StoredResponse
from minillm.schemas import (
    CreateResponseRequest,
    ModelResponse,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)

_IMAGE_URI = re.compile(r"^data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)$", re.I)
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_HISTORY_MESSAGES = 64
_MAX_HISTORY_BYTES = 16 * 1024 * 1024


class APIError(Exception):
    def __init__(self, status: int, message: str, code: str, param: str | None = None):
        self.status = status
        self.message = message
        self.code = code
        self.param = param


def _error(message: str, code: str, param: str | None = None) -> dict[str, Any]:
    return {
        "message": message,
        "type": "server_error"
        if code
        in {
            "server_error",
            "service_unavailable",
            "generation_error",
            "engine_error",
            "model_load_failed",
            "decoding_error",
            "capacity_exceeded",
        }
        else "invalid_request_error",
        "param": param,
        "code": code,
    }


@dataclass
class ActiveResponse:
    response: ModelResponse
    messages: tuple[dict[str, Any], ...]
    handle: Any
    store: bool
    output_id: str
    cancel_requested: bool = False

    def cancel(self) -> None:
        if not self.cancel_requested:
            self.cancel_requested = True
            self.handle.cancel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Neither the engine nor any model is constructed while importing this module.
    from minillm.engine.scheduler import Engine

    app.state.engine = Engine()
    app.state.responses = ResponseStore()
    app.state.active = {}
    app.state.drainers = set()
    app.state.engine.start()
    try:
        yield
    finally:
        for active in app.state.active.values():
            active.cancel()
        await asyncio.to_thread(app.state.engine.close)
        if app.state.drainers:
            done, pending = await asyncio.wait(app.state.drainers, timeout=2)
            for task in pending:
                task.cancel()


app = FastAPI(lifespan=lifespan)


class BodyLimitMiddleware:
    """Bound JSON before parsing, including chunked uploads without Content-Length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > _MAX_HISTORY_BYTES:
                response = JSONResponse(
                    {
                        "error": _error(
                            "Request body exceeds 16 MiB", "request_too_large"
                        )
                    },
                    status_code=413,
                )
                return await response(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app.add_middleware(BodyLimitMiddleware)


@app.exception_handler(APIError)
async def api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        {"error": _error(exc.message, exc.code, exc.param)}, status_code=exc.status
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    first = exc.errors()[0]
    location = first.get("loc", ())
    param = ".".join(str(part) for part in location[1:]) if len(location) > 1 else None
    message = first.get("msg", "Invalid request")
    return JSONResponse(
        {"error": _error(message, "invalid_request", param)}, status_code=400
    )


def _check_image(image_url: str) -> None:
    match = _IMAGE_URI.fullmatch(image_url)
    if match is None:
        raise APIError(
            400,
            "Only PNG, JPEG, and WebP data URI images are supported",
            "invalid_image",
            "input",
        )
    encoded = match.group(2)
    if len(encoded) > (_MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise APIError(400, "Image exceeds the 4 MiB limit", "invalid_image", "input")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error:
        raise APIError(
            400, "Invalid base64 image data", "invalid_image", "input"
        ) from None
    if len(raw) > _MAX_IMAGE_BYTES:
        raise APIError(400, "Image exceeds the 4 MiB limit", "invalid_image", "input")
    image_type = match.group(1).lower()
    valid_header = (
        raw.startswith(b"\x89PNG\r\n\x1a\n")
        if image_type == "png"
        else raw.startswith(b"\xff\xd8")
        if image_type == "jpeg"
        else raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"
    )
    if not valid_header:
        raise APIError(
            400, "Image data does not match its declared type", "invalid_image", "input"
        )


def _new_messages(value: str | list[Any]) -> tuple[dict[str, Any], ...]:
    if isinstance(value, str):
        if not value:
            raise APIError(400, "Input cannot be empty", "invalid_input", "input")
        return ({"role": "user", "content": value},)
    if not value:
        raise APIError(400, "Input cannot be empty", "invalid_input", "input")
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        content = message.content
        if isinstance(content, list):
            if not content:
                raise APIError(
                    400,
                    "Message content cannot be empty",
                    "invalid_input",
                    f"input.{index}.content",
                )
            parts = [part.model_dump(exclude_unset=True) for part in content]
            for part in parts:
                if part["type"] == "input_image":
                    if message.role != "user":
                        raise APIError(
                            400,
                            "Images are only supported in user messages",
                            "invalid_input",
                            f"input.{index}.content",
                        )
                    _check_image(part["image_url"])
                elif part["type"] == "output_text" and message.role != "assistant":
                    raise APIError(
                        400,
                        "output_text is only supported in assistant messages",
                        "invalid_input",
                        f"input.{index}.content",
                    )
                elif part["type"] == "input_text" and message.role == "assistant":
                    raise APIError(
                        400,
                        "Use output_text in assistant messages",
                        "invalid_input",
                        f"input.{index}.content",
                    )
            content = parts
        elif not content:
            raise APIError(
                400,
                "Message content cannot be empty",
                "invalid_input",
                f"input.{index}.content",
            )
        messages.append({"role": message.role, "content": content})
    if messages[-1]["role"] != "user":
        raise APIError(
            400, "Last input message must be from user", "invalid_input", "input"
        )
    return tuple(messages)


def _prepare_messages(
    request: CreateResponseRequest, store: ResponseStore
) -> tuple[dict[str, Any], ...]:
    history: tuple[dict[str, Any], ...] = ()
    if request.previous_response_id:
        previous = store.get(request.previous_response_id)
        if previous is None:
            raise APIError(
                404,
                "Previous response not found",
                "response_not_found",
                "previous_response_id",
            )
        if previous.response.model != request.model:
            raise APIError(
                400,
                "Previous response belongs to another model",
                "model_mismatch",
                "previous_response_id",
            )
        if previous.response.status not in {"completed", "incomplete"}:
            raise APIError(
                400,
                "Previous response is not complete",
                "invalid_previous_response",
                "previous_response_id",
            )
        history = previous.messages
    messages = history + _new_messages(request.input)
    if len(messages) > _MAX_HISTORY_MESSAGES:
        raise APIError(
            400, "Conversation exceeds 64 messages", "history_limit_exceeded", "input"
        )
    if (
        sum(len(str(message).encode("utf-8")) for message in messages)
        + len((request.instructions or "").encode("utf-8"))
        > _MAX_HISTORY_BYTES
    ):
        raise APIError(
            400,
            "Conversation exceeds the 16 MiB history limit",
            "history_limit_exceeded",
            "input",
        )
    return messages


def _initial_response(
    request: CreateResponseRequest, response_id: str
) -> ModelResponse:
    return ModelResponse(
        id=response_id,
        created_at=int(time.time()),
        status="queued",
        model=request.model,
        instructions=request.instructions,
        previous_response_id=request.previous_response_id,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        seed=request.seed,
    )


def _runtime_status(code: str) -> int:
    if any(term in code for term in ("context", "invalid", "unsupported", "image")):
        return 400
    return 503


def _finish(active: ActiveResponse, event: dict[str, Any]) -> ModelResponse:
    response = active.response
    if event["type"] == "error":
        engine_code = str(event.get("code", "server_error"))
        code = (
            "invalid_prompt" if _runtime_status(engine_code) == 400 else "server_error"
        )
        response = response.model_copy(
            update={
                "status": "failed",
                "completed_at": int(time.time()),
                "error": _error(str(event.get("message", "Generation failed")), code),
            }
        )
    else:
        reason = event.get("finish_reason", "stop")
        status = (
            "cancelled"
            if active.cancel_requested or reason == "cancelled"
            else "incomplete"
            if reason == "length"
            else "completed"
        )
        output_status = (
            "incomplete" if status in {"incomplete", "cancelled"} else "completed"
        )
        text = str(event.get("text", ""))
        prompt_tokens = int(event.get("prompt_tokens", 0))
        completion_tokens = int(event.get("completion_tokens", 0))
        response = response.model_copy(
            update={
                "status": status,
                "completed_at": int(time.time()),
                "incomplete_details": {"reason": "max_output_tokens"}
                if status == "incomplete"
                else None,
                "output": [
                    ResponseOutputMessage(
                        id=active.output_id,
                        status=output_status,
                        content=[ResponseOutputText(text=text)],
                    )
                ],
                "usage": ResponseUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    input_tokens_details={
                        "cached_tokens": int(event.get("cached_tokens", 0)),
                        "cache_write_tokens": int(event.get("cache_write_tokens", 0)),
                    },
                ),
            }
        )
    active.response = response
    return response


def _retain(app: FastAPI, active: ActiveResponse) -> None:
    app.state.active.pop(active.response.id, None)
    if active.store and active.response.status in {
        "completed",
        "incomplete",
        "cancelled",
        "failed",
    }:
        messages = active.messages
        if active.response.output:
            messages += (
                {
                    "role": "assistant",
                    "content": active.response.output[0].content[0].text,
                },
            )
        app.state.responses.save(StoredResponse(active.response, messages))


async def _drain_cancelled(app: FastAPI, active: ActiveResponse) -> None:
    """Keep the cancelled ID retrievable until the worker releases its cache."""
    try:
        while True:
            try:
                event = active.handle.events.get_nowait()
            except queue.Empty:
                if app.state.engine.stats().get("status") == "stopped":
                    break
                await asyncio.sleep(0.01)
                continue
            if event.get("type") in {"done", "error"}:
                _finish(active, event)
                break
    finally:
        _retain(app, active)


def _cleanup(app: FastAPI, active: ActiveResponse) -> None:
    if active.response.status in {"completed", "incomplete", "failed"} or (
        active.response.status == "cancelled" and active.response.usage is not None
    ):
        _retain(app, active)
        return
    active.cancel()
    active.response = active.response.model_copy(
        update={
            "status": "cancelled",
            "completed_at": int(time.time()),
            "incomplete_details": None,
        }
    )
    task = asyncio.create_task(_drain_cancelled(app, active))
    app.state.drainers.add(task)
    task.add_done_callback(app.state.drainers.discard)


async def _next_event(
    request: Request, active: ActiveResponse
) -> dict[str, Any] | None:
    while True:
        if await request.is_disconnected():
            active.cancel()
            return None
        try:
            return active.handle.events.get_nowait()
        except queue.Empty:
            # No background queue.get may outlive this coroutine and steal the
            # terminal event from the disconnect cleanup task.
            await asyncio.sleep(0.01)


def _sse(name: str, data: dict[str, Any], sequence_number: int) -> str:
    if name in {"response.output_text.delta", "response.output_text.done"}:
        data.setdefault("logprobs", [])
    payload = {"type": name, "sequence_number": sequence_number, **data}
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def _stream_events(
    request: Request, active: ActiveResponse
) -> AsyncIterator[str]:
    sequence = 0
    terminal = False

    def emit(name: str, **data: Any) -> str:
        nonlocal sequence
        sequence += 1
        return _sse(name, data, sequence)

    try:
        yield emit("response.created", response=active.response.model_dump(mode="json"))
        active.response = active.response.model_copy(update={"status": "in_progress"})
        yield emit(
            "response.in_progress", response=active.response.model_dump(mode="json")
        )
        item = ResponseOutputMessage(
            id=active.output_id, status="in_progress", content=[]
        )
        yield emit(
            "response.output_item.added",
            output_index=0,
            item=item.model_dump(mode="json"),
        )
        yield emit(
            "response.content_part.added",
            item_id=active.output_id,
            output_index=0,
            content_index=0,
            part=ResponseOutputText(text="").model_dump(mode="json"),
        )
        emitted = ""
        while True:
            event = await _next_event(request, active)
            if event is None:
                return
            if event.get("type") == "delta":
                delta = str(event.get("text", ""))
                if delta:
                    emitted += delta
                    yield emit(
                        "response.output_text.delta",
                        item_id=active.output_id,
                        output_index=0,
                        content_index=0,
                        delta=delta,
                    )
                continue
            if event.get("type") not in {"done", "error"}:
                continue
            response = _finish(active, event)
            if event["type"] == "done":
                text = response.output[0].content[0].text
                # The worker emits text deltas. A final suffix can arise from decoder flushing.
                if text.startswith(emitted) and len(text) > len(emitted):
                    yield emit(
                        "response.output_text.delta",
                        item_id=active.output_id,
                        output_index=0,
                        content_index=0,
                        delta=text[len(emitted) :],
                    )
                yield emit(
                    "response.output_text.done",
                    item_id=active.output_id,
                    output_index=0,
                    content_index=0,
                    text=text,
                )
                yield emit(
                    "response.content_part.done",
                    item_id=active.output_id,
                    output_index=0,
                    content_index=0,
                    part=response.output[0].content[0].model_dump(mode="json"),
                )
                yield emit(
                    "response.output_item.done",
                    output_index=0,
                    item=response.output[0].model_dump(mode="json"),
                )
            name = (
                "response.failed"
                if response.status == "failed"
                else "response.incomplete"
                if response.status in {"incomplete", "cancelled"}
                else "response.completed"
            )
            # A client may continue immediately on receipt of the terminal event.
            terminal = True
            _retain(request.app, active)
            yield emit(name, response=response.model_dump(mode="json"))
            yield "data: [DONE]\n\n"
            return
    except asyncio.CancelledError:
        active.cancel()
        raise
    finally:
        if not terminal:
            _cleanup(request.app, active)


@app.get("/")
async def index() -> HTMLResponse:
    from pathlib import Path

    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text())


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    status = request.app.state.engine.stats().get("status")
    return JSONResponse(
        {"status": "ok" if status in {"ready", "loading"} else status},
        status_code=200 if status in {"ready", "loading"} else 503,
    )


@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    return {"object": "list", "data": request.app.state.engine.models()}


@app.get("/metrics")
@app.get("/v1/metrics")
async def metrics(request: Request) -> dict[str, Any]:
    return request.app.state.engine.stats()


@app.post("/v1/responses", response_model=ModelResponse)
async def create_response(
    request: Request, body: CreateResponseRequest
) -> ModelResponse | StreamingResponse:
    messages = _prepare_messages(body, request.app.state.responses)
    response_id = f"resp_{uuid4().hex}"
    prompt_messages = list(messages)
    if body.instructions:
        prompt_messages = [
            {"role": "system", "content": body.instructions},
            *prompt_messages,
        ]
    try:
        handle = request.app.state.engine.submit(
            request_id=response_id,
            model=body.model,
            messages=prompt_messages,
            max_new_tokens=body.max_output_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            seed=body.seed,
        )
    except KeyError:
        raise APIError(
            404, f"Model '{body.model}' not found", "model_not_found", "model"
        ) from None
    except ValueError as exc:
        raise APIError(400, str(exc), "invalid_request") from None
    except RuntimeError as exc:
        raise APIError(503, str(exc), "service_unavailable") from None
    active = ActiveResponse(
        _initial_response(body, response_id),
        messages,
        handle,
        body.store,
        f"msg_{uuid4().hex}",
    )
    request.app.state.active[response_id] = active
    if body.stream:
        return StreamingResponse(
            _stream_events(request, active),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        active.response = active.response.model_copy(update={"status": "in_progress"})
        while True:
            event = await _next_event(request, active)
            if event is None:
                raise APIError(499, "Client disconnected", "cancelled")
            if event.get("type") == "delta":
                continue
            if event.get("type") in {"done", "error"}:
                response = _finish(active, event)
                if response.status == "failed":
                    error = response.error or {}
                    raise APIError(
                        _runtime_status(str(error.get("code", "server_error"))),
                        str(error.get("message", "Generation failed")),
                        str(error.get("code", "server_error")),
                    )
                return response
    except asyncio.CancelledError:
        active.cancel()
        raise
    finally:
        if (
            active.response.status in {"queued", "in_progress"}
            or active.cancel_requested
        ):
            # A terminal cancellation has already consumed its worker event.
            if active.response.usage is not None:
                _retain(request.app, active)
            else:
                _cleanup(request.app, active)
        else:
            _retain(request.app, active)


@app.get("/v1/responses/{response_id}", response_model=ModelResponse)
async def get_response(request: Request, response_id: str) -> ModelResponse:
    stored = request.app.state.responses.get(response_id)
    if stored is not None:
        return stored.response
    active = request.app.state.active.get(response_id)
    if active is not None and active.store:
        return active.response
    raise APIError(404, "Response not found", "response_not_found")


@app.delete("/v1/responses/{response_id}")
async def delete_response(request: Request, response_id: str) -> dict[str, Any]:
    active = request.app.state.active.get(response_id)
    if active is not None and active.store:
        active.store = False
        active.cancel()
        return {"id": response_id, "object": "response.deleted", "deleted": True}
    if not request.app.state.responses.delete(response_id):
        raise APIError(404, "Response not found", "response_not_found")
    return {"id": response_id, "object": "response.deleted", "deleted": True}


@app.post("/v1/responses/{response_id}/cancel", response_model=ModelResponse)
async def cancel_response(request: Request, response_id: str) -> ModelResponse:
    active = request.app.state.active.get(response_id)
    if active is None:
        stored = request.app.state.responses.get(response_id)
        if stored is None:
            raise APIError(404, "Response not found", "response_not_found")
        return stored.response
    active.cancel()
    active.response = active.response.model_copy(
        update={
            "status": "cancelled",
            "completed_at": int(time.time()),
            "incomplete_details": None,
        }
    )
    return active.response
