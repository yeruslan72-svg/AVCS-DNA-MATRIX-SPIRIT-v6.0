# Dockerfile
FROM python:3.10-slim

# System dependencies for industrial libraries
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Start application
CMD ["streamlit", "run", "avcs_dna_matrix_spirit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
