# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — AI JEE Agent
# Compatible with: HuggingFace Spaces (port 7860), Render (port 8000), local
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IS DOCKER?
#   Docker is a tool that packages your app + all its dependencies (Python,
#   libraries, ChromaDB) into a single "container" — a self-contained box
#   that runs the same on your laptop, your friend's PC, and any cloud server.
#   Think of it like shipping your entire Python environment in a box.
#
# WHY USE DOCKER IN AI/ML SYSTEMS?
#   • "Works on my machine" is the #1 problem in ML deployment.
#     Docker solves this by freezing the exact environment.
#   • Cloud platforms (Render, Railway, GCP) all understand Docker.
#     You just hand them the Dockerfile and they handle the rest.
#   • Easy rollback: if v2 breaks, just run the v1 image.
#   • Scalable: run 10 copies of your container when traffic spikes.
#
# ─────────────────────────────────────────────────────────────────────────────

# Step 1: Pick a base image
#   python:3.11-slim = Python 3.11, minimal OS (Debian), no extra bloat.
#   "slim" is lighter than the full image — faster to build and deploy.
FROM python:3.11-slim

# Step 2: Set working directory inside the container
#   All commands from here run relative to /app
WORKDIR /app

# Step 3: Install system-level libraries needed by some Python packages
#   - build-essential: gcc compiler (needed by chromadb, some ML libs)
#   - curl: used for healthchecks
#   We do this BEFORE copying code so Docker can cache this layer.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements first (for caching optimization)
#   Docker builds in layers. If requirements.txt hasn't changed,
#   Docker skips this slow "pip install" step on the next build.
COPY requirements.txt .

# Step 5: Install Python dependencies
#   --no-cache-dir: don't store pip's download cache (saves disk space)
#   --upgrade pip: always use the latest pip to avoid install errors
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your code into the container
#   The .dockerignore file (which we also create) excludes venv/, __pycache__, etc.
COPY . .

# Step 7: Expose port 8000
#   This tells Docker "this container listens on port 8000".
#   Cloud platforms like Render will map this to their public port.
# HuggingFace Spaces uses 7860, Render/local uses 8000
EXPOSE 7860

# Step 8: Healthcheck
#   Render and other platforms use this to know if your container is alive.
#   Every 30s, it curls the / endpoint. If it fails 3 times in a row,
#   the container is marked "unhealthy" and restarted.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Step 9: Start the server
#   uvicorn = the ASGI server that runs FastAPI
#   main:app = "in main.py, find the 'app' object"
#   --host 0.0.0.0 = listen on ALL network interfaces (required for Docker)
#   --port 8000 = must match EXPOSE above
#   --workers 1 = use 1 process (ChromaDB doesn't safe to share across processes)
# PORT env var: HuggingFace sets it to 7860 automatically
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
