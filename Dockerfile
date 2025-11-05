# =========================================================
# 🧠 AVCS DNA-MATRIX SPIRIT v7.x — Production Dockerfile
# =========================================================
# Multi-stage build: lightweight, fast, GPU-ready
# =========================================================

# ---------- 1️⃣ Base Image (Python + CUDA optional) ----------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base
# Для CPU-версии можно заменить на: python:3.11-slim-bookworm

LABEL maintainer="AVCS Systems <support@avcs.ai>"
LABEL version="7.x"
LABEL description="Adaptive Industrial Intelligence System — AVCS DNA-MATRIX SPIRIT"

# ---------- 2️⃣ System & Python Setup ----------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    APP_HOME=/app

WORKDIR $APP_HOME

# Системные пакеты и зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential git wget curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- 3️⃣ Copy dependencies & install ----------
COPY requirements.txt .

# Используем кэш pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- 4️⃣ Copy application files ----------
COPY . .

# ---------- 5️⃣ Environment Variables ----------
ENV STREAMLIT_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    PATH="$APP_HOME:$PATH"

# ---------- 6️⃣ Streamlit Config ----------
RUN mkdir -p ~/.streamlit && \
    echo "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = true\nport = ${STREAMLIT_PORT}" > ~/.streamlit/config.toml

# ---------- 7️⃣ Healthcheck ----------
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s CMD curl -f http://localhost:${STREAMLIT_PORT}/_stcore/health || exit 1

# ---------- 8️⃣ Expose Port & Run ----------
EXPOSE ${STREAMLIT_PORT}

CMD ["streamlit", "run", "avcs_dna_matrix_spirit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
