# JEE AI Agent

A full-stack multilingual AI-powered JEE tutor with a LangGraph agent, RAG 
pipeline, REST API, and chat interface. Supports Hindi and English — 
automatically detects language, translates queries for retrieval, and 
responds in the user's language.

![JEE AI Tutor Chat UI](screenshot.png)

## How it works

1. NCERT PDFs are chunked, embedded, and stored in ChromaDB
2. User sends a question in Hindi or English via the chat UI
3. Language is detected automatically using langdetect
4. Hindi queries are translated to English before ChromaDB search
5. LangGraph agent retrieves relevant content and answers
6. Response is returned in the same language the user asked in

## Tech Stack
- Python 3.12
- LangChain + LangGraph
- Groq (LLaMA 3.1 — free LLM API)
- HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- ChromaDB (local vector database)
- FastAPI + Uvicorn
- langdetect (language detection)
- Vanilla HTML/CSS/JS

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
   pip install -r requirements.txt
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

## Multilingual Support

The agent automatically detects Hindi and English input. Hindi queries are 
translated to English before searching the knowledge base, with responses 
returned in the user's original language.

| Input | Language | Behaviour |
|---|---|---|
| "What is electric field?" | English | Direct ChromaDB search |
| "Vidyut kshetra kya hai?" | Hindi | Translate → search → reply in Hindi |

**Example:**
```
User:  Vidyut kshetra ki SI unit kya hai?
Agent: Vidyut kshetra ki SI unit Newton per Coulomb (N/C) hai.
```

Extensible to other Indian languages (Bengali, Tamil, Telugu etc.) 
via AI4Bharat models.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Health check |
| POST | /chat | Send a question, get an answer |
| DELETE | /session/{id} | Clear a session's memory |

**POST /chat request:**
```json
{
  "question": "Vidyut kshetra ki SI unit kya hai?",
  "session_id": "your-session-id"
}
```

**POST /chat response:**
```json
{
  "answer": "Vidyut kshetra ki SI unit Newton per Coulomb (N/C) hai.",
  "session_id": "your-session-id"
}
```

## Agent Tools

| Tool | Trigger | Description |
|---|---|---|
| search_jee_material | Theory questions | Searches ChromaDB, supports Hindi + English |
| calculator | Math problems | Evaluates mathematical expressions |

## Project Structure
```
JEE-AI-Agent/
├── data/               # NCERT PDF chapters
├── chroma_db/          # Auto-generated vector database
├── rag_pipeline.py     # Loads PDFs, creates vector DB (run once)
├── rag_chain.py        # Simple RAG without agent
├── agent.py            # LangGraph agent with tools, memory, multilingual
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
- [x] Hindi language detection and translation
- [x] Multilingual responses