# =========================================================
# AVCS DNA-MATRIX SPIRIT v7.0 — Dockerfile
# =========================================================

# ---- Base image ----
FROM python:3.10-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Copy project files ----
COPY . /app

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    git \
 && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies ----
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Environment variables ----
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ---- Expose Streamlit port ----
EXPOSE 8501

# ---- Run Streamlit app ----
CMD ["streamlit", "run", "avcs_dna_matrix_spirit_app.py"]
