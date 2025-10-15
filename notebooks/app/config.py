from pathlib import Path
MODEL_FINAL = Path("model_final.pkl")
MODEL_BASELINE = Path("model_baseline.pkl")
FAISS_INDEX = Path("vector_index/index.faiss")
FAISS_IDMAP = Path("vector_index/id_map.json")
CHROMA_DIR = Path("vector_index/chroma_db")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
