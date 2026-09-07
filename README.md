---
title: AI JEE Tutor
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# JEE AI Tutor

A chatbot that answers JEE questions from NCERT textbooks. It looks up the
relevant pages before answering, does the arithmetic when a question needs it,
and replies in Hindi if you ask in Hindi.

Live: https://sparsh-25-ai-jee-tutor.hf.space

## What it does

You ask a question. The app searches an NCERT text database for the pages that
match, then writes an answer using those pages. If the question needs a number
crunched, it works out the expression and evaluates it in Python instead of
letting the model do mental math.

It handles Physics, Chemistry and Maths for Class 11 and 12.

## How it works

The textbooks are processed once, ahead of time. Each PDF is split into ~500
character chunks, each chunk is turned into a vector, and the vectors are stored
in ChromaDB. That gives 18,481 chunks to search through.

Then, for every question:

1. **Check the language.** Hindi script is caught by a Unicode range. Romanised
   Hindi ("vidyut kshetra kya hai") is caught by a small word list. Everything
   else is treated as English.

2. **Decide what to do.** The model is asked to reply in a fixed format saying
   whether the question needs a lookup, a calculation, or both:

   ```
   ACTION: SEARCH or CALCULATE or BOTH
   SEARCH_QUERY: kinetic energy definition formula
   MATH_EXPRESSION: 0.5 * 10 * 5**2
   ```

   The reply is parsed line by line. If nothing parses, it falls back to a plain
   search, so a bad reply degrades into normal RAG rather than an error.

3. **Run the tools.** `SEARCH_QUERY` gets embedded and matched against ChromaDB,
   which returns the 3 closest chunks. `MATH_EXPRESSION` gets evaluated in
   Python. Hindi queries are translated to English first, because the textbooks
   are in English and a Hindi query won't land near English text in vector space.

4. **Write the answer.** The chunks and the calculation get passed back to the
   model along with the last few messages, and it writes the reply in whatever
   language the question came in.

An English question takes two model calls. A Hindi one takes three.

### A note on the design

This is a router, not a ReAct agent. The model picks its tools once, the tools
run once, and then it answers. It never sees the tool output and re-plans.

That's on purpose. Every question costs a predictable 2-3 model calls and
2-5 seconds. A proper agent loop would retry bad lookups, but it could also
spend 15 seconds and six calls on a question a student wants answered now.

## Tech stack

- Python 3.11
- FastAPI + Uvicorn
- Groq running `openai/gpt-oss-120b`
- ChromaDB for the vectors
- sentence-transformers `all-MiniLM-L6-v2` for embeddings, run locally
- LangChain for the Groq and Chroma wrappers
- Plain HTML/CSS/JS for the chat page, with marked and KaTeX for formatting
- Docker, deployed on HuggingFace Spaces

## Setup

Clone it and make a virtual environment:

```bash
git clone https://github.com/Sparsh-25/AI-JEE-Agent.git
cd AI-JEE-Agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add a Groq API key in a `.env` file:

```
GROQ_API_KEY=your_key_here
```

The vector database is already in the repo, so you can start straight away:

```bash
python -m uvicorn main:app --reload
```

Open http://localhost:8000.

### Rebuilding the database

Only needed if you want to add or replace textbooks. The PDFs are not in the
repo, so download the chapters you want from ncert.nic.in into a `data/` folder
first:

```bash
mkdir data
python rag_pipeline.py
```

This takes a few minutes and overwrites `chroma_db/`.

## API

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/` | The chat page |
| GET | `/health` | Returns ok, used by the Docker healthcheck |
| POST | `/chat` | Ask a question |
| POST | `/query` | Same, but also returns timing and the chunks that were retrieved |
| GET | `/sessions` | Lists active sessions and message counts |
| DELETE | `/session/{id}` | Clears one session's history |

Ask something:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Coulombs law?", "session_id": "abc"}'
```

```json
{
  "answer": "Coulomb's law gives the electrostatic force between two point charges...",
  "session_id": "abc"
}
```

`/query` is the same call but returns `latency_seconds` and `retrieved_context`
too, so you can see what the search actually found. That's useful because a
right answer built on the wrong chunks means the model knew it already and the
search didn't help.

Conversation history lives in a dictionary in memory. Restarting the server
clears every chat.

## Evaluation

`evaluate.py` sends 30 questions across all three subjects to `/query`, records
the timing and the retrieved chunks, and writes a CSV you can grade by hand.

```bash
python -m uvicorn main:app --reload    # in one terminal
python evaluate.py                      # in another
```

Results from the last full run:

| | |
|---|---|
| Answered without error | 30 / 30 |
| Average response time | 3.4 s |
| Answers graded correct | 29 / 30 |
| Answers actually backed by the retrieved text | 18 / 30 |

The gap between those last two rows is the interesting part. The model gets
most answers right, but only about 60% of the time is the retrieved text
actually what it needed. The rest of the time it already knew the answer and
the lookup didn't contribute.

These numbers are from before the switch to `gpt-oss-120b` and haven't been
re-measured since.

## Known limitations

- **Some of the text is unreadable.** NCERT PDFs use custom font encodings that
  pypdf can't decode, so roughly a quarter of the chunks come out as symbols.
  They still get searched and still crowd out good results. Switching the
  loader to PyMuPDF and rebuilding would fix this, and it's the main thing
  holding retrieval quality back.
- **The calculator strips letters** before evaluating, to stop anyone running
  Python through it. That also breaks real maths: `sin(30)` becomes `(30)`, and
  you get a wrong number with no error.
- **No rate limiting or auth.** Anyone with the URL can use it, and every
  request costs a Groq call.
- **One worker only.** ChromaDB and the session dictionary aren't shared across
  processes, so the container runs a single worker.

## Files

```
main.py           FastAPI server, endpoints and sessions
agent.py          Language detection, routing, tools, answer generation
rag_pipeline.py   Builds the vector database from PDFs (run once)
rag_chain.py      An earlier plain-RAG version, kept for reference
evaluate.py       Runs 30 test questions and writes results to CSV/JSON
index.html        The chat page
chroma_db/        The vector database, tracked with Git LFS
Dockerfile        Container setup, serves on port 7860
```

## Deploying

The Dockerfile works as-is on HuggingFace Spaces. Set `GROQ_API_KEY` as a Space
secret. The vector database is baked into the image, so there's nothing to build
on startup.

The Space is pushed from an orphan branch with no history, because the earlier
commits contain the source PDFs and HuggingFace scans the whole history for
large files.
