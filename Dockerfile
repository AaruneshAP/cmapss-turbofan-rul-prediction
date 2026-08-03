# Stage 5 — Dockerfile
# Slim Python image matching the training environment
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY serving/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the serving application and model artifacts
COPY serving/ .

# Expose the API port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
