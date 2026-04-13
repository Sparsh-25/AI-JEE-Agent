"""
main.py — Production FastAPI Server for AI JEE Agent
=====================================================

KEY ENDPOINTS:
    GET  /           → Health check
    POST /chat       → Original chat endpoint (session-based, no context returned)
    POST /query      → NEW evaluation endpoint (returns latency + retrieved context)
    DELETE /session/{id} → Clear a session

WHAT IS LATENCY IN LLM SYSTEMS?
    Latency = the total wall-clock time from "user sends question" to
    "server sends back the answer". In LLM systems, this includes:
      1. Embedding the query (converting words → numbers) — fast, ~50ms
      2. Vector search in ChromaDB (find relevant chunks) — fast, ~100ms
      3. LLM inference (Groq generates the answer) — slow, ~1–5s
    Most of your latency comes from step 3. This is why fast inference
    APIs like Groq exist — they run optimized hardware (LPUs).

WHAT IS "RETRIEVAL QUALITY"?
    Retrieval quality = how well your vector search finds the RIGHT chunks.
    Even if your LLM is perfect, if ChromaDB returns the wrong PDFchunks,
    the final answer will be wrong. The /query endpoint returns
    "retrieved_context" so you can inspect what was retrieved.
    
    Signs of BAD retrieval quality:
      - retrieved_context is empty or generic
      - The retrieved text is from a different topic than the question
    Signs of GOOD retrieval quality:
      - retrieved_context contains the exact formula or concept asked
      - The LLM answer directly references things in retrieved_context
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agent import ask_agent, search_jee_material

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# Why structured logging? In production, you want to grep logs to debug issues.
# FORMAT: timestamp | level | message
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("jee_agent")

# ──────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI JEE Agent API",
    description="RAG-based JEE tutoring assistant powered by LLaMA 3.3 + ChromaDB",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For production: replace with your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store — maps session_id → conversation history
sessions: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class QueryRequest(BaseModel):
    """Request schema for the /query evaluation endpoint."""
    question: str
    session_id: str = "eval_session"
    include_context: bool = True   # set False to hide retrieved chunks


class QueryResponse(BaseModel):
    """
    Richer response for evaluation — includes latency and retrieved context.
    
    Fields:
        answer           → The LLM's final answer
        latency_seconds  → Total time from request receipt to response send
        session_id       → Which session this belongs to
        retrieved_context → The actual PDF chunks retrieved from ChromaDB
                            (useful for checking if retrieval was relevant)
    """
    answer: str
    latency_seconds: float
    session_id: str
    retrieved_context: list[str]  # list of raw text chunks from ChromaDB


# ──────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE — logs every incoming request
# This is useful for debugging: you can see ALL requests in your terminal
# ──────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs method + path for every request. Helps debug issues in production."""
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 3)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}s)")
    return response


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check. Used by Docker healthcheck and deployment platforms."""
    return {"status": "JEE Agent is running", "version": "1.1.0"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Original chat endpoint — maintains conversation history per session.
    Does NOT return latency or retrieved context (kept for frontend compatibility).
    """
    start_time = time.time()

    # ── Input validation ─────────────────────────────────────────────────────
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
    """
    Evaluation endpoint — designed for the evaluate.py script.
    
    Returns:
        answer           → LLM response
        latency_seconds  → How long the full pipeline took
        retrieved_context → List of actual PDF chunks ChromaDB retrieved
                            (lets you verify retrieval quality)
    
    WHY THIS IS USEFUL:
        The /chat endpoint is for users. The /query endpoint is for YOU — 
        the developer — to understand how well your system works.
        By inspecting retrieved_context, you can see if ChromaDB is finding
        the right study material for each question.
    """
    start_time = time.time()

    # ── Input validation ─────────────────────────────────────────────────────
    if not request.question.strip():
        return JSONResponse(status_code=400, content={"error": "Question cannot be empty"})
    if len(request.question) > 1000:
        return JSONResponse(status_code=400, content={"error": "Question too long."})

    logger.info(f"[QUERY] Q: {request.question[:100]}")

    try:
        if request.session_id not in sessions:
            sessions[request.session_id] = []

        history = sessions[request.session_id]

        # ── Step 1: Retrieve context from ChromaDB (for logging + return) ──────
        retrieved_chunks: list[str] = []
        if request.include_context:
            raw_context = search_jee_material(request.question)
            # The function returns a single joined string — split back into chunks
            retrieved_chunks = [
                chunk.strip()
                for chunk in raw_context.split("\n\n")
                if chunk.strip()
            ]
            # ── Log retrieved chunks for debugging ────────────────────────────
            logger.info(f"[QUERY] Retrieved {len(retrieved_chunks)} chunks from ChromaDB:")
            for idx, chunk in enumerate(retrieved_chunks, 1):
                # Log first 200 chars of each chunk — enough to see if it's relevant
                logger.info(f"[QUERY]   Chunk {idx}: {chunk[:200]}...")

        # ── Step 2: Get full agent answer ─────────────────────────────────────
        answer, updated_history = ask_agent(request.question, history)
        sessions[request.session_id] = updated_history

        elapsed = round(time.time() - start_time, 3)

        # ── Log the final answer ──────────────────────────────────────────────
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
    """Clears the conversation history for a given session."""
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"[SESSION] Cleared: {session_id}")
        return {"cleared": session_id}
    return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})


@app.get("/sessions", tags=["Session"])
def list_sessions():
    """Lists all active session IDs and their history length. Useful for debugging."""
    return {
        sid: {"message_count": len(hist)}
        for sid, hist in sessions.items()
    }