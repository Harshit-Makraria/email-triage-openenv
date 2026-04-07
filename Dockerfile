FROM python:3.11-slim

# ── system deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer) ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────
COPY openenv.yaml .
COPY email_triage_env.py .
COPY pyproject.toml .
COPY server/ ./server/
COPY tasks/ ./tasks/

# ── Runtime config ───────────────────────────────────────────
ENV PORT=7860
EXPOSE 7860

# Health-check (used by from_docker_image polling loop)
HEALTHCHECK --interval=5s --timeout=5s --start-period=15s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --log-level info"]
