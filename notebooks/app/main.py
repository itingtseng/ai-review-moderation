from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from app.utils.loaders import LazyEmbedder, load_model, get_vector_backends
from app import config

app = FastAPI(title="AI Review Moderation API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

MODEL, MODEL_TAG = load_model()
EMBEDDER = LazyEmbedder.get()
search_with_vector, search_with_text, BACKEND = get_vector_backends()

class PredictIn(BaseModel):
    text: str
    top_k: Optional[int] = config.TOP_K

class PredictOut(BaseModel):
    reason: Optional[str] = None
    proba: Optional[float] = None
    model: str
    similar: Optional[List[dict]] = None
    note: Optional[str] = None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_TAG if MODEL is not None else "none",
        "vector_backend": BACKEND
    }

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    notes = []

    emb = EMBEDDER.encode([payload.text])
    emb_np = np.array(emb, dtype="float32")

    reason = None
    proba = None
    model_name = "none"
    if MODEL is None:
        notes.append("No model found. Similarity search only.")
    else:
        model_name = MODEL_TAG
        try:
            if hasattr(MODEL, "predict_proba"):
                proba = float(np.max(MODEL.predict_proba(emb_np)[0]))
            reason = str(MODEL.predict(emb_np)[0])
        except Exception as e:
            notes.append(f"Classification error: {e}")

    similar = None
    try:
        k = payload.top_k or config.TOP_K
        if search_with_vector is not None:
            similar = search_with_vector(emb_np, k)
        elif search_with_text is not None:
            similar = search_with_text(payload.text, k)
        else:
            notes.append("No vector index available.")
    except Exception as e:
        notes.append(f"Similarity search error: {e}")

    return PredictOut(
        reason=reason,
        proba=proba,
        model=model_name,
        similar=similar,
        note=" | ".join(notes) if notes else None
    )
