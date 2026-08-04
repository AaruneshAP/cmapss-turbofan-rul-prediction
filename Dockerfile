# Stage 5 — Dockerfile
# Slim Python image matching the training environment.
# Python 3.13-slim keeps the base layer small; matches the Colab/Kaggle
# training environment used in Stages 1-3.
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first so Docker can cache the install layers
# independently of the application code.
COPY serving/requirements.txt .

# ── Step 1: Install PyTorch (CPU-only wheel) ─────────────────────────────────
# We use the dedicated CPU index URL so pip downloads the ~200 MB CPU wheel
# instead of the ~2 GB CUDA wheel that PyPI's default index points to.
# This is the primary reason generic "pip install torch" can OOM / EOF during
# a Docker build on resource-constrained hosts (Render free tier, CI runners).
# If you ever need GPU inference, swap the index URL for the matching CUDA one,
# e.g.: https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install remaining dependencies from requirements.txt ──────────────
# torch is intentionally absent from requirements.txt to prevent pip from
# re-resolving it against PyPI and accidentally pulling the CUDA variant.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the serving application and model artifacts
COPY serving/ .

# Expose the API port
EXPOSE 8000

# Health check for container orchestration.
# Uses ${PORT:-8000} so it targets the same port uvicorn is actually bound to.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8000') + '/health')" || exit 1

# Run with uvicorn.
# Shell form (not exec/JSON-array form) is required so the shell can expand
# the $PORT environment variable that Render injects at runtime.
# Falls back to 8000 for local docker run without -e PORT=....
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
