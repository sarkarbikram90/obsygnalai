FROM debian:bookworm-slim AS glibc-provider

FROM alpine:3.21

SHELL ["/bin/ash", "-eo", "pipefail", "-c"]

ENV STREAMLIT_SERVER_CORS_ALLOW_ALL=false \
    STREAMLIT_SERVER_ENABLE_CORS=true \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    OLLAMA_HOST=0.0.0.0:11434 \
    PYTHONUNBUFFERED=1

# Install base system dependencies on Alpine
RUN apk update && apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    tar \
    zstd \
    bash \
    procps \
    ca-certificates

# Copy genuine, matched GNU C/C++ runtime (glibc, libstdc++, libgcc_s) from Debian
COPY --from=glibc-provider /lib/x86_64-linux-gnu/ /usr/glibc-compat/lib/
COPY --from=glibc-provider /usr/lib/x86_64-linux-gnu/libstdc++.so.6* /usr/glibc-compat/lib/
COPY --from=glibc-provider /lib64/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2
COPY --from=glibc-provider /sbin/ldconfig /usr/glibc-compat/sbin/ldconfig

# Download and extract Ollama binary directly into /usr/bin/ollama, wire symlinks and glibc ldconfig
RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | zstd -d | tar -xf - -C /usr && \
    chmod +x /usr/bin/ollama && \
    ln -sf /usr/lib/ollama /lib/ollama && \
    ln -sf /usr/lib/ollama/llama-server /lib/llama-server && \
    ln -sf /usr/bin/ollama /bin/ollama && \
    ln -sf /lib64/ld-linux-x86-64.so.2 /lib/ld-linux-x86-64.so.2 && \
    echo "/usr/glibc-compat/lib" > /etc/ld.so.conf && \
    echo "/lib/ollama" >> /etc/ld.so.conf && \
    echo "/usr/lib/ollama" >> /etc/ld.so.conf && \
    /usr/glibc-compat/sbin/ldconfig

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

ENTRYPOINT ["/bin/sh", "/app/src/start.sh"]
