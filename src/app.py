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
FIRESTORE_META_COLLECTION = "conversations_meta"

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

def async_sync_conversation_meta(session_id: str, title: str):
    """Stores conversation metadata asynchronously to Firestore."""
    def _task():
        try:
            client = get_firestore_client()
            if not client:
                return

            now_utc = datetime.now(timezone.utc)
            expire_utc = now_utc + timedelta(days=7)

            doc_ref = client.collection(FIRESTORE_META_COLLECTION).document(session_id)
            doc_ref.set({
                "session_id": session_id,
                "title": title,
                "updated_at": now_utc,
                "expire_at": expire_utc,
            }, merge=True)
            logger.info("Synced conversation metadata for %s", session_id)
        except Exception as err:
            logger.error("Firestore conversation meta write error: %s", err)

    db_executor.submit(_task)

def fetch_recent_conversations(limit: int = 15):
    """Fetches list of recent conversation summaries from Firestore."""
    try:
        client = get_firestore_client()
        if not client:
            return []
        docs = client.collection(FIRESTORE_META_COLLECTION)\
            .order_by("updated_at", direction=firestore.Query.DESCENDING)\
            .limit(limit)\
            .stream()
        results = []
        for d in docs:
            data = d.to_dict()
            results.append({
                "id": data.get("session_id", d.id),
                "title": data.get("title", "Previous Chat"),
                "updated_at": data.get("updated_at", datetime.now(timezone.utc)),
                "messages": []
            })
        return results
    except Exception as exc:
        logger.warning("Could not fetch remote conversations: %s", exc)
        return []

def load_conversation_messages(session_id: str):
    """Loads all messages for a specific conversation from Firestore."""
    try:
        client = get_firestore_client()
        if not client:
            return []
        docs = client.collection(FIRESTORE_COLLECTION)\
            .where("user_id", "==", session_id)\
            .order_by("updated_at", direction=firestore.Query.ASCENDING)\
            .stream()
        messages = []
        for d in docs:
            data = d.to_dict()
            if "role" in data and "content" in data:
                messages.append({"role": data["role"], "content": data["content"]})
        return messages if messages else None
    except Exception as exc:
        logger.warning("Could not fetch messages for session %s: %s", session_id, exc)
        return None

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
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize Conversations Session State
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    
    # Seed with existing conversations from Firestore if available
    recent_remote = fetch_recent_conversations(limit=15)
    for r in recent_remote:
        st.session_state.conversations[r["id"]] = r

# If empty, seed with initial conversation
if not st.session_state.conversations:
    initial_id = str(uuid.uuid4())
    st.session_state.conversations[initial_id] = {
        "id": initial_id,
        "title": "New Chat",
        "messages": [
            {
                "role": "assistant",
                "content": "Welcome to **Obsygnal AI**"
            }
        ],
        "updated_at": datetime.now(timezone.utc)
    }
    st.session_state.active_session_id = initial_id

if "active_session_id" not in st.session_state or st.session_state.active_session_id not in st.session_state.conversations:
    latest_conv = max(
        st.session_state.conversations.values(),
        key=lambda c: c.get("updated_at", datetime.min.replace(tzinfo=timezone.utc))
    )
    st.session_state.active_session_id = latest_conv["id"]

active_conv = st.session_state.conversations[st.session_state.active_session_id]

# Lazy-load conversation messages from Firestore if needed
if not active_conv.get("messages"):
    loaded_msgs = load_conversation_messages(st.session_state.active_session_id)
    if loaded_msgs:
        active_conv["messages"] = loaded_msgs
    else:
        active_conv["messages"] = [
            {
                "role": "assistant",
                "content": "Welcome to **Obsygnal AI**"
            }
        ]

# Synchronize legacy variables for backward compatibility
st.session_state.user_id = st.session_state.active_session_id
st.session_state.messages = active_conv["messages"]

# --- Persistent Sidebar (ChatGPT Style) ---
with st.sidebar:
    st.markdown("### ⚡ Obsygnal AI")
    
    # "+ New Chat" Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.conversations[new_id] = {
            "id": new_id,
            "title": "New Chat",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Welcome to **Obsygnal AI**"
                }
            ],
            "updated_at": datetime.now(timezone.utc)
        }
        st.session_state.active_session_id = new_id
        st.session_state.user_id = new_id
        st.session_state.messages = st.session_state.conversations[new_id]["messages"]
        st.rerun()

    st.markdown("---")
    st.caption("RECENT CONVERSATIONS")

    # Sort conversations by updated_at descending
    sorted_convs = sorted(
        st.session_state.conversations.values(),
        key=lambda c: c.get("updated_at", datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True
    )

    for conv in sorted_convs:
        cid = conv["id"]
        title = conv.get("title", "New Chat")
        is_active = (cid == st.session_state.active_session_id)
        
        button_label = f"▶ {title}" if is_active else f"💬 {title}"
        if st.button(button_label, key=f"nav_{cid}", use_container_width=True, disabled=is_active):
            st.session_state.active_session_id = cid
            st.session_state.user_id = cid
            if not conv.get("messages"):
                loaded = load_conversation_messages(cid)
                conv["messages"] = loaded if loaded else [{"role": "assistant", "content": "Welcome to **Obsygnal AI**"}]
            st.session_state.messages = conv["messages"]
            st.rerun()

    st.markdown("---")
    # Quick Markdown export for current conversation
    curr_chat_export = "\n\n".join(
        [f"**{m['role'].capitalize()}**: {m['content']}" for m in active_conv.get("messages", [])]
    )
    st.download_button(
        label="📥 Export Chat (.md)",
        data=curr_chat_export,
        file_name=f"obsygnal_chat_{st.session_state.active_session_id[:8]}.md",
        mime="text/markdown",
        use_container_width=True
    )

# --- Main Chat Area ---
st.markdown("## ⚡ Obsygnal AI")
st.caption(f"Chat: **{active_conv.get('title', 'New Chat')}** | Profile: `{MODEL_PROFILE}` | Engine: `Cloud Run Managed`")

# Render historical messages
for msg in active_conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Ask Obsygnal AI anything..."):
    # Append user prompt
    active_conv["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # If first prompt, auto-generate conversation title
    if active_conv.get("title") in ("New Chat", "Conversation", None):
        summary_title = prompt[:32].strip() + ("..." if len(prompt) > 32 else "")
        active_conv["title"] = summary_title
        async_sync_conversation_meta(st.session_state.active_session_id, summary_title)

    # Sync user input to Firestore asynchronously
    async_sync_firestore(st.session_state.active_session_id, "user", prompt)

    # Stream neural inference output live
    with st.chat_message("assistant"):
        response_text = st.write_stream(route_inference_stream(prompt))

    active_conv["messages"].append({"role": "assistant", "content": response_text})
    active_conv["updated_at"] = datetime.now(timezone.utc)

    # Sync assistant response and metadata to Firestore asynchronously
    async_sync_firestore(st.session_state.active_session_id, "assistant", response_text)
    async_sync_conversation_meta(st.session_state.active_session_id, active_conv["title"])

    # Update backward-compatible session_state
    st.session_state.messages = active_conv["messages"]
