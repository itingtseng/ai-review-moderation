import json
from pathlib import Path
from typing import Optional, Tuple, List

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None

try:
    import chromadb
except Exception:
    chromadb = None

from app import config


class LazyEmbedder:
    """Lazy singleton for SentenceTransformer."""
    _model: Optional[SentenceTransformer] = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(config.EMBED_MODEL_NAME)
        return cls._model


def load_model() -> Tuple[Optional[object], str]:
    """Load final model first, then fallback to baseline."""
    if config.MODEL_FINAL.exists():
        try:
            return joblib.load(config.MODEL_FINAL), "final"
        except Exception as e:
            return None, f"Failed to load model_final.pkl: {e}"

    if config.MODEL_BASELINE.exists():
        try:
            return joblib.load(config.MODEL_BASELINE), "baseline"
        except Exception as e:
            return None, f"Failed to load model_baseline.pkl: {e}"

    return None, "No available model found."


class FaissIndex:
    """FAISS index wrapper."""
    def __init__(self, index_path: Path, idmap_path: Path):
        self.index = None
        self.idmap = []
        self.dim = None
        self.index_path = index_path
        self.idmap_path = idmap_path

    def exists(self) -> bool:
        return self.index_path.exists() and self.idmap_path.exists() and faiss is not None

    def load(self):
        if faiss is None:
            raise RuntimeError("faiss-cpu not installed.")
        self.index = faiss.read_index(str(self.index_path))
        self.dim = self.index.d
        with open(self.idmap_path, "r", encoding="utf-8") as f:
            self.idmap = json.load(f)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[dict]:
        if self.index is None:
            return []
        D, I = self.index.search(query_vec.astype("float32"), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.idmap[idx] if 0 <= idx < len(self.idmap) else {}
            results.append({"score": float(dist), "meta": meta})
        return results


class ChromaIndex:
    """Chroma vector store wrapper."""
    def __init__(self, persist_dir: Path, collection_name: str = "reviews"):
        self.client = None
        self.collection = None
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def exists(self) -> bool:
        return chromadb is not None and self.persist_dir.exists()

    def load(self):
        if chromadb is None:
            raise RuntimeError("chromadb not installed.")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def search(self, query_texts: List[str], top_k: int) -> List[dict]:
        if self.collection is None:
            return []
        res = self.collection.query(query_texts=query_texts, n_results=top_k)
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "score": float(res["distances"][0][i]) if "distances" in res else None,
                "meta": res["metadatas"][0][i] if "metadatas" in res else {},
                "text": res["documents"][0][i] if "documents" in res else None,
            })
        return out


def get_vector_backends():
    """Auto-detect FAISS or Chroma."""
    faiss_idx = FaissIndex(config.FAISS_INDEX, config.FAISS_IDMAP)
    chroma_idx = ChromaIndex(config.CHROMA_DIR)

    if faiss_idx.exists():
        faiss_idx.load()
        def _search_vec(emb, k):
            return faiss_idx.search(emb, k)
        return _search_vec, None, "faiss"

    if chroma_idx.exists():
        chroma_idx.load()
        def _search_txt(text, k):
            return chroma_idx.search([text], k)
        return None, _search_txt, "chroma"

    return None, None, "none"
