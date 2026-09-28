import os

from fastapi import FastAPI, HTTPException

from minillm.conversations import (
    ConversationLimitError,
    ConversationStore,
)
from minillm.inference import (
    ContextLimitExceeded,
    InferenceEngine,
)
from minillm.schemas import (
    DeleteConversationResponse,
    GenerateRequest,
    GenerateResponse,
)

MODEL_PATH = os.environ.get(
    "MINILLM_MODEL_PATH",
    "models/smollm2-1.7b-instruct",
)

app = FastAPI()
store = ConversationStore(max_conversations=2)
engine = InferenceEngine(MODEL_PATH)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    created = request.conversation_id is None

    if created:
        try:
            conversation_id, conversation = store.create()
        except ConversationLimitError:
            raise HTTPException(status_code=429, detail="too many convos :(") from None
    else:
        conversation_id = request.conversation_id
        conversation = store.get(conversation_id)

        if conversation is None:
            raise HTTPException(status_code=404, detail="convo not found")

    try:
        result = engine.generate(
            conversation=conversation,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
    except ContextLimitExceeded:
        if created:
            store.delete(conversation_id)
        raise HTTPException(
            status_code=400,
            detail="Conversation is too long",
        ) from None
    except Exception:
        if created:
            store.delete(conversation_id)
        raise

    store.save(conversation_id, conversation)

    return GenerateResponse(
        conversation_id=conversation_id,
        text=result.text,
        reused_tokens=result.reused_tokens,
    )


@app.delete(
    "/conversations/{conversation_id}",
    response_model=DeleteConversationResponse,
)
def delete_conversation(
    conversation_id: str,
) -> DeleteConversationResponse:
    if not store.delete(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return DeleteConversationResponse(deleted=True)
