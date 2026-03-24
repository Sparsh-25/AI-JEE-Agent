import warnings
warnings.filterwarnings("ignore")

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agent import ask_agent
import time


# Set up logging — writes to terminal with timestamp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

sessions = {}


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    session_id: str



@app.get("/")
def root():
    return {"status": "JEE Agent is running"}



@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()

    # Reject empty questions
    if not request.question.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Question cannot be empty"}
        )


    # Reject questions that are too long
    if len(request.question) > 1000:
        return JSONResponse(
            status_code=400,
            content={"error": "Question too long. Keep it under 1000 characters."}
        )

    logger.info(f"Session: {request.session_id} | Question: {request.question}")


    try:
        if request.session_id not in sessions:
            sessions[request.session_id] = []

        history = sessions[request.session_id]
        answer, updated_history = ask_agent(request.question, history)
        sessions[request.session_id] = updated_history

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Session: {request.session_id} | Response time: {elapsed}s")

        return ChatResponse(answer=answer, session_id=request.session_id)


    except Exception as e:
        
        elapsed = round(time.time() - start_time, 2)
        logger.error(f"Session: {request.session_id} | Error after {elapsed}s | {str(e)}")

        return JSONResponse(
            status_code=500,
            content={"error": "Something went wrong. Please try again."}
        )

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Session cleared: {session_id}")
        return {"cleared": session_id}
    return JSONResponse(
        status_code=404,
        content={"error": f"Session {session_id} not found"}
    )