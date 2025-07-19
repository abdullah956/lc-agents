from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse
from llm.memory import get_memory
from llm.chain import get_chain
from utils.executor import run_in_executor

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_input = request.message.strip()
    session_id = request.session_id.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    try:
        memory = get_memory(session_id)
        chain = get_chain(memory)

        response = await run_in_executor(lambda: chain.predict(input=user_input))

        messages = memory.chat_memory.messages
        messages_list = [msg.dict() for msg in messages]
        return ChatResponse(response=response, messages=messages_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
