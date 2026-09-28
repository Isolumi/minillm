import os
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from minillm.inference import (
    ContextLimitExceeded,
    InferenceEngine,
)
from minillm.responses import ResponseStore, StoredResponse
from minillm.schemas import (
    CreateResponseRequest,
    ModelResponse,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)

MODEL_PATH = os.environ.get(
    "MINILLM_MODEL_PATH",
    "models/smollm2-1.7b-instruct",
)
MODEL_NAME = os.environ.get("MINILLM_MODEL_NAME", Path(MODEL_PATH).name)
MODEL_CREATED = int(time.time())

app = FastAPI()
response_store = ResponseStore(max_responses=128)
engine = InferenceEngine(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": "minillm",
            }
        ],
    }


@app.post("/v1/responses", response_model=ModelResponse)
def create_response(request: CreateResponseRequest) -> ModelResponse:
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=404, detail="Model not found")
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet")
    if request.temperature != 0:
        raise HTTPException(status_code=400, detail="Only temperature=0 is supported")

    history: tuple[tuple[str, str], ...] = ()
    if request.previous_response_id is not None:
        previous = response_store.get(request.previous_response_id)
        if previous is None:
            raise HTTPException(status_code=404, detail="Previous response not found")
        history = previous.messages

    if isinstance(request.input, str):
        if not request.input:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        new_messages = (("user", request.input),)
    else:
        if not request.input:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        new_messages = tuple(
            (message.role, message.content) for message in request.input
        )

    messages = history + new_messages
    if messages[-1][0] != "user":
        raise HTTPException(status_code=400, detail="Last input must be from user")

    prompt_messages = [
        {"role": role, "content": content} for role, content in messages
    ]
    if request.instructions:
        prompt_messages.insert(
            0, {"role": "system", "content": request.instructions}
        )
    started_at = int(time.time())
    try:
        result = engine.generate(
            messages=prompt_messages,
            max_new_tokens=request.max_output_tokens,
        )
    except ContextLimitExceeded:
        raise HTTPException(
            status_code=400,
            detail="Conversation is too long",
        ) from None

    complete = result.finish_reason == "stop"
    response = ModelResponse(
        id=f"resp_{uuid4().hex}",
        created_at=started_at,
        completed_at=int(time.time()),
        status="completed" if complete else "incomplete",
        incomplete_details=None if complete else {"reason": "max_output_tokens"},
        model=MODEL_NAME,
        output=[
            ResponseOutputMessage(
                id=f"msg_{uuid4().hex}",
                status="completed" if complete else "incomplete",
                content=[ResponseOutputText(text=result.text)],
            )
        ],
        instructions=request.instructions,
        previous_response_id=request.previous_response_id,
        max_output_tokens=request.max_output_tokens,
        usage=ResponseUsage(
            input_tokens=result.prompt_tokens,
            output_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )

    if request.store:
        response_store.save(
            StoredResponse(
                response=response,
                messages=messages + (("assistant", result.text),),
            )
        )

    return response


@app.get("/v1/responses/{response_id}", response_model=ModelResponse)
def get_response(response_id: str) -> ModelResponse:
    stored = response_store.get(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Response not found")
    return stored.response
