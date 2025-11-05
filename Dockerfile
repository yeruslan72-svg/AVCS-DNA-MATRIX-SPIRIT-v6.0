# --------------------------------------------------------
# 🧬 AVCS DNA-MATRIX SPIRIT — Dockerfile
# Production-ready container for Streamlit orchestration app
# --------------------------------------------------------

FROM python:3.11-slim AS base

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.local/bin:$PATH"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Default Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Entrypoint
CMD ["streamlit", "run", "avcs_dna_matrix_spirit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
