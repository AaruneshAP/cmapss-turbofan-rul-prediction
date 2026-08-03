# Deployment Notes — CMAPSS RUL Prediction API

## Render Free Tier Deployment

### Prerequisites
1. GitHub repository with the full project pushed (data/ excluded via .gitignore)
2. Render account (free at render.com)

### Deployment Steps

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: CMAPSS RUL prediction system"
   git remote add origin https://github.com/YOUR_USERNAME/cmapss-predictive-maintenance.git
   git push -u origin main
   ```

2. **Create Render Web Service**:
   - Go to https://dashboard.render.com/
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: cmapss-rul-api
     - **Runtime**: Docker
     - **Plan**: Free
     - **Branch**: main
   - Click "Create Web Service"

3. **Verify Deployment**:
   ```bash
   # Health check
   curl https://cmapss-rul-api.onrender.com/health

   # Prediction
   curl -X POST https://cmapss-rul-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d @sample_payload.json
   ```

### Cold-Start Behavior (IMPORTANT)

The Render free tier spins down the service after 15 minutes of inactivity. This means:

- **First request after sleep**: Takes 30–60 seconds as the container restarts, the Python process boots, and the model loads into memory.
- **Subsequent requests**: Fast (typically < 200ms) as long as the service stays warm.
- **This is NOT a bug** — it's expected behavior on the free tier.

For a live demo or interview:
1. Send a health check request 1–2 minutes before the demo starts to "wake up" the service
2. Mention the cold-start behavior proactively — it demonstrates understanding of cloud infrastructure constraints

### Upgrading for Production
To eliminate cold starts, upgrade to Render's paid tier ($7/month for Starter), which keeps the service running continuously. For true production use, consider:
- Adding a `/docs` endpoint (FastAPI auto-generates Swagger UI)
- Rate limiting to prevent abuse
- Request logging for monitoring
- Multiple workers via `uvicorn --workers N`

## Local Docker Verification

```bash
# Build
docker build -t cmapss-rul-api .

# Run
docker run -p 8000:8000 cmapss-rul-api

# Test health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```
