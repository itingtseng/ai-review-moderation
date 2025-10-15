import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Loading & caching utilities
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the baseline sklearn pipeline (model_baseline.pkl)."""
    import joblib
    p = Path("model_baseline.pkl")
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)
def load_faiss_index():
    """Try loading FAISS index and id_map. Return (index, id_map, msg)."""
    try:
        import faiss  # type: ignore
    except Exception:
        return None, None, "FAISS not available (faiss-cpu not installed)."

    vec_dir = Path("vector_index")
    index_path = vec_dir / "index.faiss"
    idmap_path = vec_dir / "id_map.json"
    if not index_path.exists() or not idmap_path.exists():
        return None, None, "vector_index/ files not found."

    index = faiss.read_index(str(index_path))
    with open(idmap_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return index, id_map, None

@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load the same ST model used to build FAISS."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_corpus_for_fallback(expected_len: int | None):
    """
    Load the text corpus for TF-IDF fallback.
    Tries processed CSVs and aligns with id_map length when possible.
    Returns (texts, meta_df, msg).
    """
    # Try common paths
    candidates = [
        Path("data/processed/reported_reviews_clean.csv"),
        Path("data/processed/reviews_clean.csv"),
    ]
    df = None
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            break
    if df is None:
        return None, None, "No processed CSV found under data/processed/."

    if "review_text" not in df.columns:
        return None, None, "CSV has no 'review_text' column."

    # If expected_len (from id_map) is provided, check alignment quickly
    if expected_len is not None and len(df) != expected_len:
        # Not fatal: we can still build TF-IDF; metadata may not align 1:1 with id_map
        # We'll still return df for generic metadata.
        pass

    return df["review_text"].astype(str).tolist(), df, None

@st.cache_resource(show_spinner=False)
def build_tfidf_index(texts: list[str]):
    """Build TF-IDF matrix for cosine fallback."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)
    X = vec.fit_transform(texts)
    return {"vectorizer": vec, "matrix": X}  # store in resource cache

def tfidf_search(query: str, tfidf_store: dict, top_k: int = 5):
    """Return (scores, indices) using cosine similarity on TF-IDF."""
    from sklearn.metrics.pairwise import linear_kernel
    vec = tfidf_store["vectorizer"]
    X = tfidf_store["matrix"]
    q = vec.transform([query])
    sims = linear_kernel(q, X)[0]  # shape (N,)
    # Get top-k indices
    if top_k >= len(sims):
        order = np.argsort(-sims)
    else:
        # partial top-k for speed on large N
        part = np.argpartition(-sims, top_k)[:top_k]
        order = part[np.argsort(-sims[part])]
    scores = sims[order]
    return scores, order

def faiss_search(query_text: str, index, id_map, embedder, top_k: int = 5):
    """Encode query and search FAISS index, return list of {distance, meta}."""
    qv = embedder.encode([query_text], convert_to_numpy=True).astype("float32")
    if qv.ndim == 1:
        qv = qv[None, :]
    D, I = index.search(qv, top_k)
    items = []
    seen = set()
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = id_map[idx] if 0 <= idx < len(id_map) else {}
        oid = meta.get("object_id")
        if oid and oid in seen:
            continue
        if oid:
            seen.add(oid)
        items.append({"score": float(dist), "meta": meta, "backend": "faiss"})
    return items

def tfidf_items_from_indices(scores, indices, meta_df: pd.DataFrame | None):
    """Build display rows for TF-IDF fallback using available metadata."""
    rows = []
    for s, i in zip(scores, indices):
        meta = {}
        if meta_df is not None:
            # Try to attach helpful fields if present
            for k in ["object_id", "reason", "date_created", "complex_id"]:
                if k in meta_df.columns:
                    val = meta_df.iloc[int(i)][k]
                    meta[k] = None if (pd.isna(val) if hasattr(val, "__float__") else False) else val
        rows.append({"score": float(s), "meta": meta, "backend": "tfidf"})
    return rows

# ------------
# Streamlit UI
# ------------
st.set_page_config(page_title="AI Review Moderation Demo", layout="wide")
st.title("🛡️ AI Review Moderation Demo")

st.caption(
    "This demo runs fully on Streamlit. It predicts a moderation reason with the baseline "
    "model (if present) and retrieves similar examples via FAISS; if FAISS is not available, "
    "it falls back to TF-IDF cosine neighbors."
)

# Load artifacts
model = load_model()
faiss_index, id_map, faiss_msg = load_faiss_index()
embedder = load_embedder() if (faiss_index is not None and id_map is not None and faiss_msg is None) else None

# Prepare TF-IDF fallback only if FAISS is not ready
tfidf_store = None
tfidf_meta_df = None
tfidf_msg = None
if embedder is None:
    texts, df_meta, tfidf_msg = load_corpus_for_fallback(
        expected_len=(len(id_map) if id_map is not None else None)
    )
    if texts is not None:
        tfidf_store = build_tfidf_index(texts)
        tfidf_meta_df = df_meta

# Sidebar status
with st.sidebar:
    st.header("Artifacts Status")
    st.write("**Baseline model (model_baseline.pkl):** ", "✅ loaded" if model else "⚠️ not found")
    if faiss_index is not None and id_map is not None and faiss_msg is None:
        st.write("**Vector index:** ✅ FAISS")
    else:
        st.write("**Vector index:** ",
                 "⚠️ TF-IDF fallback ready" if tfidf_store is not None else "⚠️ unavailable")
        if faiss_msg:
            st.caption(f"Note: {faiss_msg}")
        if tfidf_msg and tfidf_store is None:
            st.caption(f"Fallback note: {tfidf_msg}")
    st.divider()
    topk = st.slider("Top-k similar examples", 1, 10, 5)

# Input
user_text = st.text_area("Input review text", height=160, placeholder="Paste a review here...")

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Analyze", type="primary", use_container_width=True):
        st.session_state["last_query"] = user_text

query = st.session_state.get("last_query")
if query:
    st.subheader("Results")

    # --- 1) Prediction ---
    with st.container(border=True):
        st.markdown("### Predicted Reason / Confidence")
        if model is None:
            st.info("Baseline model not found. Add `model_baseline.pkl` to enable predictions.")
        else:
            try:
                pred = model.predict([query])[0]
                proba_tbl = None
                # Best-effort: show top-5 class probabilities if available
                final_est = getattr(model, "steps", [("final", model)])[-1][1]
                if hasattr(final_est, "predict_proba"):
                    probs = final_est.predict_proba([query])[0]
                    classes = list(final_est.classes_)
                    order = np.argsort(probs)[::-1][:5]
                    proba_tbl = pd.DataFrame({
                        "reason": [classes[i] for i in order],
                        "prob": [float(probs[i]) for i in order]
                    })
                st.write(f"**Predicted reason:** `{pred}`")
                if proba_tbl is not None:
                    st.dataframe(proba_tbl, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # --- 2) Similar examples ---
    with st.container(border=True):
        st.markdown("### Similar Examples")
        try:
            if embedder is not None and faiss_index is not None and id_map is not None:
                rows = faiss_search(query, faiss_index, id_map, embedder, top_k=topk)
                backend = "faiss"
            elif tfidf_store is not None:
                scores, idxs = tfidf_search(query, tfidf_store, top_k=topk)
                rows = tfidf_items_from_indices(scores, idxs, tfidf_meta_df)
                backend = "tfidf"
            else:
                rows = []
                backend = None

            if not rows:
                st.info("No similar examples available. Provide FAISS index or processed CSV to enable this section.")
            else:
                st.caption(f"Backend: **{backend}**")
                for i, item in enumerate(rows, start=1):
                    meta = item["meta"]
                    distance_or_score = item["score"]
                    label = "distance" if backend == "faiss" else "cosine"
                    st.markdown(
                        f"**#{i}** — {label}: `{distance_or_score:.4f}`  "
                        f"• reason: `{meta.get('reason', 'N/A')}`  "
                        f"• object_id: `{meta.get('object_id', 'N/A')}`  "
                        f"• date: `{meta.get('date_created', 'N/A')}`"
                    )
        except Exception as e:
            st.error(f"Similarity search failed: {e}")
