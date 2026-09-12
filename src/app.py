import os
import uuid
import json
import logging
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import requests
import streamlit as st
from google.cloud import firestore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("obsygnal-ai")

# Configuration
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434/api/generate")
MODEL_PROFILE = os.getenv("MODEL_PROFILE", "qwen2.5:1.5b")
GCP_PROJECT = os.getenv("GCP_PROJECT", None)
FIRESTORE_COLLECTION = "chat_histories"

# Inference Tuning (CPU & KV-Cache optimization)
OLLAMA_NUM_THREADS = int(os.getenv("OLLAMA_NUM_THREADS", "2"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))

# Asynchronous worker pool for non-blocking Firestore syncs
db_executor = ThreadPoolExecutor(max_workers=4)

@st.cache_resource
def get_firestore_client():
    """Initializes and caches Google Cloud Firestore client."""
    try:
        if GCP_PROJECT:
            return firestore.Client(project=GCP_PROJECT)
        return firestore.Client()
    except Exception as exc:
        logger.warning("Firestore client initialization skipped or deferred: %s", exc)
        return None

def async_sync_firestore(user_id: str, role: str, content: str):
    """Stores message envelope asynchronously to Firestore with TTL timestamps."""
    def _task():
        try:
            client = get_firestore_client()
            if not client:
                return

            now_utc = datetime.now(timezone.utc)
            expire_utc = now_utc + timedelta(days=7)

            doc_ref = client.collection(FIRESTORE_COLLECTION).document()
            doc_ref.set({
                "user_id": user_id,
                "role": role,
                "content": content,
                "timestamp": now_utc.isoformat(),
                "updated_at": now_utc,
                "expire_at": expire_utc,
            })
            logger.info("Asynchronously synced message envelope for user %s to Firestore", user_id)
        except Exception as err:
            logger.error("Firestore async write error: %s", err)

    db_executor.submit(_task)

def route_inference_stream(prompt: str, endpoint: str = None, model: str = None):
    """Streams neural inference tokens in real-time from local Ollama service."""
    target_url = endpoint or OLLAMA_ENDPOINT
    target_model = model or MODEL_PROFILE

    # Normalize url if port/path omitted
    if target_url in ("http://127.0.0.1", "http://127.0.0"):
        target_url = f"{target_url}:11434/api/generate"
    elif target_url.endswith(":11434"):
        target_url = f"{target_url}/api/generate"

    payload = {
        "model": target_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_thread": OLLAMA_NUM_THREADS,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT
        }
    }

    try:
        response = requests.post(target_url, json=payload, stream=True, timeout=(10, 600))
        if response.status_code != 200:
            try:
                err_data = response.json()
                err_msg = err_data.get("error", response.text)
            except Exception:
                err_msg = response.text
            logger.warning("Ollama inference non-200: %s", err_msg)
            yield f"Model initializing ({err_msg}). Please retry in a moment."
            return

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                except Exception:
                    continue
    except requests.exceptions.RequestException as err:
        logger.error("Inference query routing failure: %s", err)
        yield f"Inference engine routing failure: {err}"

def route_inference(prompt: str, endpoint: str = None, model: str = None) -> str:
    """Non-streaming collector for testing and evaluation."""
    return "".join(route_inference_stream(prompt, endpoint=endpoint, model=model))

# Streamlit Page Setup
st.set_page_config(
    page_title="Obsygnal AI",
    page_icon="⚡",
    layout="centered"
)

# Persistent Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to **Obsygnal AI**"
        }
    ]

# Header UI
st.markdown("## ⚡ Obsygnal AI")
st.caption(f"Session ID: `{st.session_state.user_id}` | Profile: `{MODEL_PROFILE}` | Engine: `Cloud Run Managed`")

# Render historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Ask Obsygnal AI anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Sync user input to Firestore asynchronously
    async_sync_firestore(st.session_state.user_id, "user", prompt)

    # Stream neural inference output live
    with st.chat_message("assistant"):
        response_text = st.write_stream(route_inference_stream(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Sync assistant response to Firestore asynchronously
    async_sync_firestore(st.session_state.user_id, "assistant", response_text)
