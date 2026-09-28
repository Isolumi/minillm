from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ResponseInputMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class CreateResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | list[ResponseInputMessage]
    instructions: str | None = None
    previous_response_id: str | None = None
    max_output_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0, le=2)
    stream: bool = False
    store: bool = True


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, object]] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["completed", "incomplete"]
    content: list[ResponseOutputText]


class ResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, int] = Field(
        default_factory=lambda: {"cached_tokens": 0}
    )
    output_tokens_details: dict[str, int] = Field(
        default_factory=lambda: {"reasoning_tokens": 0}
    )


class ModelResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    completed_at: int
    status: Literal["completed", "incomplete"]
    error: None = None
    incomplete_details: dict[str, str] | None = None
    model: str
    output: list[ResponseOutputMessage]
    instructions: str | None = None
    previous_response_id: str | None = None
    max_output_tokens: int
    temperature: float = 0.0
    parallel_tool_calls: bool = False
    tool_choice: Literal["none"] = "none"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    usage: ResponseUsage
