# demo_app.py
# Simple Streamlit UI that calls FastAPI /similar and shows results.
# All comments are in English as requested.

import os
import json
import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Config: API URL resolution
# -----------------------------
# Priority: Streamlit secrets -> env var -> default localhost:8010
DEFAULT_API = "http://127.0.0.1:8010"
_api_from_secrets = None
try:
    # Avoid hard-failing when secrets.toml is missing
    _api_from_secrets = st.secrets.get("API_URL", None)  # type: ignore[attr-defined]
except Exception:
    _api_from_secrets = None

API_URL = (
    _api_from_secrets
    or os.environ.get("AI_REVIEW_API_URL")
    or DEFAULT_API
)

st.set_page_config(page_title="AI Review Moderation Demo", layout="wide")
st.title("🛡️ AI Review Moderation Demo")

# -----------------------------
# Sidebar: API controls
# -----------------------------
with st.sidebar:
    st.subheader("API Settings")
    api_url = st.text_input("API base URL", value=API_URL, help="Your FastAPI base URL.")
    colA, colB = st.columns(2)
    with colA:
        do_health = st.button("Check /health")
    with colB:
        do_openapi = st.button("Check /openapi.json")

    if do_health:
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            st.success(f"/health OK: {r.status_code}")
            st.json(r.json())
        except Exception as e:
            st.error(f"/health error: {e}")

    if do_openapi:
        try:
            r = requests.get(f"{api_url}/openapi.json", timeout=10)
            st.success(f"/openapi.json OK: {r.status_code}")
            st.json(r.json())
        except Exception as e:
            st.error(f"/openapi.json error: {e}")

st.markdown("---")

# -----------------------------
# Main: Similar search UI
# -----------------------------
st.subheader("Similar Examples (FAISS)")

query = st.text_area(
    "Input text",
    value="This post contains spam links http://spam.example",
    help="Paste a review/post text to search for similar items.",
    height=120,
)

k = st.slider("Top-k", min_value=1, max_value=20, value=5, step=1)

col1, col2 = st.columns([1, 4])

with col1:
    run_btn = st.button("Analyze")
with col2:
    show_raw = st.checkbox("Show raw response JSON", value=False)

if run_btn:
    try:
        payload = {"text": query, "k": int(k)}
        r = requests.post(f"{api_url}/similar", json=payload, timeout=30)
        if r.status_code != 200:
            st.error(f"API returned status {r.status_code}")
        else:
            data = r.json()
            if show_raw:
                st.code(json.dumps(data, indent=2)[:4000], language="json")

            items = data.get("items", [])
            vb = data.get("vector_backend", "none")

            if vb == "none":
                st.warning(
                    "Vector backend is not loaded. Make sure you built the FAISS index and the API can find 'vector_index/index.faiss' and 'vector_index/id_map.json'."
                )

            if not items:
                st.info("No results.")
            else:
                # Normalize items into a flat table for display
                rows = []
                for it in items:
                    meta = it.get("meta", {}) or {}
                    rows.append(
                        {
                            "rank": len(rows) + 1,
                            "distance": it.get("distance", None),
                            "object_id": meta.get("object_id", None),
                            "reason": meta.get("reason", None),
                            "date_created": meta.get("date_created", None),
                            "complex_id": meta.get("complex_id", None),
                            "index": it.get("index", None),
                        }
                    )
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Request failed: {e}")

st.markdown("---")
st.caption(
    "Tip: If you see 'vector_backend: none' in /health, re-check your index files or restart the API."
)
