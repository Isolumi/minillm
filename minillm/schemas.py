"""The deliberately small, strict subset of the OpenAI Responses schema we serve."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputText(StrictModel):
    type: Literal["input_text"]
    text: str = Field(min_length=1)


class InputImage(StrictModel):
    type: Literal["input_image"]
    image_url: str = Field(min_length=1)


class AssistantOutputText(StrictModel):
    type: Literal["output_text"]
    text: str


InputPart = InputText | InputImage | AssistantOutputText


class ResponseInputMessage(StrictModel):
    type: Literal["message"] | None = None
    role: Literal["system", "user", "assistant"]
    content: str | list[InputPart]


class CreateResponseRequest(StrictModel):
    model: str = Field(min_length=1)
    input: str | list[ResponseInputMessage]
    instructions: str | None = None
    previous_response_id: str | None = Field(default=None, min_length=1)
    max_output_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stream: bool = False
    store: bool = True


class ResponseOutputText(StrictModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponseOutputMessage(StrictModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    content: list[ResponseOutputText]


class ResponseUsage(StrictModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, int] = Field(
        default_factory=lambda: {"cached_tokens": 0, "cache_write_tokens": 0}
    )
    output_tokens_details: dict[str, int] = Field(
        default_factory=lambda: {"reasoning_tokens": 0}
    )


class ModelResponse(StrictModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    completed_at: int | None = None
    status: Literal[
        "queued", "in_progress", "completed", "incomplete", "failed", "cancelled"
    ]
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, str] | None = None
    model: str
    output: list[ResponseOutputMessage] = Field(default_factory=list)
    instructions: str | None = None
    previous_response_id: str | None = None
    max_output_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    parallel_tool_calls: bool = False
    tool_choice: Literal["none"] = "none"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    usage: ResponseUsage | None = None
