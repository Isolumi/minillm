from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    conversation_id: str | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=1024)

class GenerateResponse(BaseModel):
    conversation_id: str
    text: str
    reused_tokens: int

class DeleteConversationResponse(BaseModel):
    deleted: bool