# JEE AI Agent

A full-stack AI-powered JEE tutor with a LangGraph agent, RAG pipeline, 
REST API, and chat interface. Ask questions from your NCERT notes in natural 
language — the agent retrieves relevant content and answers with context.

![JEE AI Tutor Chat UI](screenshot.png)

## How it works

1. NCERT PDFs are chunked, embedded, and stored in ChromaDB
2. A LangGraph ReAct agent receives questions via FastAPI
3. Agent decides: theory → search vector DB, math → calculator
4. Each browser session gets independent conversation memory
5. Clean chat UI served via index.html

## Tech Stack
- Python 3.12
- LangChain + LangGraph
- Groq (LLaMA 3.1 — free LLM API)
- HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- ChromaDB (local vector database)
- FastAPI + Uvicorn
- Vanilla HTML/CSS/JS (no frameworks)

## Setup

1. Clone the repo
```bash
   git clone https://github.com/Sparsh-25/AI-JEE-Agent.git
   cd AI-JEE-Agent
```

2. Create and activate virtual environment
```bash
   python -m venv venv
   source venv/bin/activate
```

3. Install dependencies
```bash
   pip install langchain langchain-community langchain-groq langchain-chroma
   pip install chromadb sentence-transformers groq python-dotenv pypdf
   pip install langgraph numexpr fastapi uvicorn python-multipart
```

4. Add your Groq API key
```bash
   # Create a .env file
   GROQ_API_KEY=your_key_here
```

5. Add JEE PDFs to data/ and build vector database
```bash
   mkdir data
   # Download NCERT chapters from ncert.nic.in and place here
   python rag_pipeline.py
```

6. Start the server
```bash
   python -m uvicorn main:app --reload
```

7. Open the chat UI
```
   Open index.html in your browser
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Health check |
| POST | /chat | Send a question, get an answer |
| DELETE | /session/{id} | Clear a session's memory |

**POST /chat request:**
```json
{
  "question": "What is Newton's second law?",
  "session_id": "your-session-id"
}
```

**POST /chat response:**
```json
{
  "answer": "Newton's second law states that F = ma...",
  "session_id": "your-session-id"
}
```

## Agent Tools

| Tool | Trigger | Description |
|---|---|---|
| search_jee_material | Theory questions | Searches ChromaDB for relevant chunks |
| calculator | Math problems | Evaluates mathematical expressions |

## Project Structure
```
JEE-AI-Agent/
├── data/               # NCERT PDF chapters
├── chroma_db/          # Auto-generated vector database
├── rag_pipeline.py     # Loads PDFs, creates vector DB (run once)
├── rag_chain.py        # Simple RAG without agent
├── agent.py            # LangGraph agent with tools and memory
├── main.py             # FastAPI server with session management
├── index.html          # Chat UI
└── .env                # API keys (never commit this)
```

## Status
- [x] LLM connected via Groq
- [x] RAG pipeline with ChromaDB
- [x] LangGraph ReAct agent
- [x] Tool calling — search + calculator
- [x] Conversation memory per session
- [x] FastAPI REST API with error handling and logging
- [x] Chat UI with session management
- [ ] Multilingual support