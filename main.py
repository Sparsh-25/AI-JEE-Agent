import warnings
warnings.filterwarnings("ignore")

import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from agent import ask_agent, search_jee_material

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("jee_agent")

app = FastAPI(
    title="AI JEE Agent API",
    description="RAG-based JEE tutoring assistant powered by GPT-OSS-120B + ChromaDB",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict = {}

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class QueryRequest(BaseModel):
    question: str
    session_id: str = "eval_session"
    include_context: bool = True

class QueryResponse(BaseModel):
    answer: str
    latency_seconds: float
    session_id: str
    retrieved_context: list[str]

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 3)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}s)")
    return response

@app.get("/", tags=["UI"])
def serve_ui():
    return FileResponse("index.html")

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "JEE Agent is running", "version": "1.1.0"}

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    start_time = time.time()

    if not request.question.strip():
        return JSONResponse(status_code=400, content={"error": "Question cannot be empty"})
    if len(request.question) > 1000:
        return JSONResponse(status_code=400, content={"error": "Question too long. Keep it under 1000 characters."})

    logger.info(f"[CHAT] Session={request.session_id} | Q: {request.question[:100]}")

    try:
        if request.session_id not in sessions:
            sessions[request.session_id] = []

        history = sessions[request.session_id]
        answer, updated_history = ask_agent(request.question, history)
        sessions[request.session_id] = updated_history

        elapsed = round(time.time() - start_time, 3)
        logger.info(f"[CHAT] Session={request.session_id} | Latency={elapsed}s | Ans_len={len(answer)}")

        return ChatResponse(answer=answer, session_id=request.session_id)

    except Exception as e:
        elapsed = round(time.time() - start_time, 3)
        logger.error(f"[CHAT] Session={request.session_id} | Error after {elapsed}s | {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Something went wrong. Please try again."})

@app.post("/query", response_model=QueryResponse, tags=["Evaluation"])
def query(request: QueryRequest):
    start_time = time.time()

    if not request.question.strip():
        return JSONResponse(status_code=400, content={"error": "Question cannot be empty"})
    if len(request.question) > 1000:
        return JSONResponse(status_code=400, content={"error": "Question too long."})

    logger.info(f"[QUERY] Q: {request.question[:100]}")

    try:
        if request.session_id not in sessions:
            sessions[request.session_id] = []

        history = sessions[request.session_id]

        retrieved_chunks: list[str] = []
        if request.include_context:
            raw_context = search_jee_material(request.question)
            retrieved_chunks = [
                chunk.strip()
                for chunk in raw_context.split("\n\n")
                if chunk.strip()
            ]
            logger.info(f"[QUERY] Retrieved {len(retrieved_chunks)} chunks from ChromaDB:")
            for idx, chunk in enumerate(retrieved_chunks, 1):
                logger.info(f"[QUERY]   Chunk {idx}: {chunk[:200]}...")

        answer, updated_history = ask_agent(request.question, history)
        sessions[request.session_id] = updated_history

        elapsed = round(time.time() - start_time, 3)

        logger.info(f"[QUERY] Latency={elapsed}s | Answer: {answer[:150]}...")

        return QueryResponse(
            answer=answer,
            latency_seconds=elapsed,
            session_id=request.session_id,
            retrieved_context=retrieved_chunks,
        )

    except Exception as e:
        elapsed = round(time.time() - start_time, 3)
        logger.error(f"[QUERY] Error after {elapsed}s | {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Something went wrong. Please try again."})

@app.delete("/session/{session_id}", tags=["Session"])
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"[SESSION] Cleared: {session_id}")
        return {"cleared": session_id}
    return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})

@app.get("/sessions", tags=["Session"])
def list_sessions():
    return {
        sid: {"message_count": len(hist)}
        for sid, hist in sessions.items()
    }
