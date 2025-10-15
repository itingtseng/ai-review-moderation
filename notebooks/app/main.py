# app/main.py
# Minimal FastAPI app with a FAISS-backed /similar endpoint (English comments only).

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
import json
import typing as t

# Try to import faiss; allow booting without it (endpoint will return "none")
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # gracefully handle environments without faiss

# -------------------------
# App setup
# -------------------------
app = FastAPI(title="AI Review Moderation API")

# CORS: allow local Streamlit or other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Paths and global state
# -------------------------
# This file is expected under: .../notebooks/app/main.py
_BASE_DIR = Path(__file__).resolve().parent              # .../notebooks/app
_VEC_DIR = _BASE_DIR.parent / "vector_index"             # .../notebooks/vector_index
_INDEX_PATH = _VEC_DIR / "index.faiss"
_IDMAP_PATH = _VEC_DIR / "id_map.json"

_faiss_ready: bool = False
_faiss_index = None
_id_map: list[dict] = []
_embedder = None  # sentence-transformers model


# -------------------------
# Models & schemas
# -------------------------
class SimilarReq(BaseModel):
    text: str
    k: int = 5  # top-k results


# -------------------------
# Startup loader
# -------------------------
@app.on_event("startup")
def _load_vector_backend() -> None:
    """
    Load sentence-transformers model and FAISS index + id_map on startup.
    If FAISS files are missing, we still boot and /similar returns vector_backend="none".
    """
    global _embedder, _faiss_ready, _faiss_index, _id_map

    # Lazy import here to avoid heavy import on module import
    from sentence_transformers import SentenceTransformer

    # Load the same embedder that was used to build the index
    _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index and id_map if present
    if faiss is not None and _INDEX_PATH.exists() and _IDMAP_PATH.exists():
        _faiss_index = faiss.read_index(str(_INDEX_PATH))
        with open(_IDMAP_PATH, "r", encoding="utf-8") as f:
            _id_map = json.load(f)
        _faiss_ready = True
    else:
        _faiss_ready = False  # index not available; endpoint will be a no-op


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    """Basic root endpoint."""
    return {"message": "Hello from AI Review Moderation API"}

@app.get("/health")
def health():
    """Health check with vector backend status."""
    return {
        "status": "ok",
        "model": "loaded" if _embedder is not None else "none",
        "vector_backend": "faiss" if _faiss_ready else "none",
        "index_path": str(_INDEX_PATH),
        "id_map_path": str(_IDMAP_PATH),
    }

@app.post("/similar")
def similar(req: SimilarReq):
    """
    Return Top-k similar items using the FAISS index.
    When the index is not loaded, returns an empty list and vector_backend="none".
    """
    if not _faiss_ready or _faiss_index is None or _embedder is None:
        return {"items": [], "vector_backend": "none"}

    # Encode the query into the same embedding space
    qv = _embedder.encode([req.text], convert_to_numpy=True).astype("float32")
    if qv.ndim == 1:
        qv = qv[None, :]
    D, I = _faiss_index.search(qv, int(max(1, req.k)))

    items: list[dict[str, t.t.Any]] = []
    seen = set()
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = _id_map[idx] if 0 <= idx < len(_id_map) else {}
        # Optional: deduplicate by object_id if available in metadata
        oid = meta.get("object_id")
        if oid and oid in seen:
            continue
        if oid:
            seen.add(oid)
        items.append(
            {
                "index": int(idx),
                "distance": float(dist),
                "meta": meta,
            }
        )

    return {"items": items, "vector_backend": "faiss"}


# -------------------------
# Local dev entry point
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
