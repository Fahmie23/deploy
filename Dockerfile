# CUDA-enabled PyTorch base — matches torch + GPU requirements
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Switch to a reliable mirror and install ffmpeg + libsndfile1
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.sg.gs/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# torch is already in the base image, so skip it to avoid re-downloading
COPY requirements.txt .
RUN grep -v "^torch$" requirements.txt | grep -v "^#" | grep -v "^$" > /tmp/reqs.txt && \
    /opt/conda/bin/pip install --no-cache-dir -r /tmp/reqs.txt

# Copy application files
COPY app/ ./

# Run as non-root
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache
USER appuser

# Redirect numba cache to a writable location for non-root user
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
