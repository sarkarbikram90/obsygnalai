#!/bin/sh

export OLLAMA_KEEP_ALIVE=24h
export OLLAMA_NUM_PARALLEL=1

echo "[Obsygnal] Launching Ollama inference daemon in background..."
ollama serve &
OLLAMA_PID=$!

echo "[Obsygnal] Waiting 5 seconds for Ollama daemon to initialize..."
sleep 5

echo "[Obsygnal] Pulling qwen2.5:1.5b model profile in background..."
ollama pull qwen2.5:1.5b &

echo "[Obsygnal] Starting Streamlit application on port 8080..."
exec streamlit run src/app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
