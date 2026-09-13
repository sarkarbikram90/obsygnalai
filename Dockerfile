FROM debian:bookworm-slim

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ENV STREAMLIT_SERVER_CORS_ALLOW_ALL=false \
    STREAMLIT_SERVER_ENABLE_CORS=true \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    OLLAMA_HOST=0.0.0.0:11434 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/nvidia/bin:${PATH}" \
    NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility"

# Install base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    tar \
    zstd \
    bash \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Ollama binary directly into /usr
RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | zstd -d | tar -xf - -C /usr && \
    chmod +x /usr/bin/ollama && \
    mkdir -p /etc/ld.so.conf.d && \
    echo "/usr/lib/ollama" > /etc/ld.so.conf.d/ollama.conf && \
    echo "/usr/local/nvidia/lib64" > /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig

# Pre-bake qwen2.5:1.5b model weights and verify live inference execution during build
RUN (ollama serve &) && \
    sleep 5 && \
    ollama pull qwen2.5:1.5b && \
    ollama run qwen2.5:1.5b "hi"; \
    pkill -f ollama || true

WORKDIR /app

# Copy and install python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy source tree
COPY src/ ./src/
RUN chmod +x ./src/start.sh

EXPOSE 8080

ENTRYPOINT ["/bin/bash", "/app/src/start.sh"]

