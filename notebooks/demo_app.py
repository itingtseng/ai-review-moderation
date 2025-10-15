# demo_app.py
# ------------------------------------------------------------------------------
# Streamlit UI for AI Review Moderation
# - Tab 1: Inference (call FastAPI /similar, optional local baseline model)
# - Tab 2: Insights (show charts + top TF-IDF terms CSV)
#
# Notes:
# * Comments are in English only (per your request).
# * API URL priority: st.secrets["API_URL"] -> env var API_URL -> default http://127.0.0.1:8010
# * All assets are expected relative to the project root (run from repo root).
# ------------------------------------------------------------------------------

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import streamlit as st
import pandas as pd

# --- Page setup ---
st.set_page_config(page_title="AI Review Moderation Demo", layout="wide")
st.title("🛡️ AI Review Moderation Demo")

# --- Config & helpers ---
ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).name == "demo_app.py") else Path.cwd()
INSIGHTS_DIR = ROOT / "reports" / "insights"
TERMS_CSV = INSIGHTS_DIR / "top_terms_per_reason.csv"
CHART_DIST = INSIGHTS_DIR / "class_distribution.png"
CHART_TREND = INSIGHTS_DIR / "monthly_trend_topN.png"
BASELINE_PATH = ROOT / "notebooks" / "model_baseline.pkl"  # adjust if you saved elsewhere

def get_api_url() -> str:
    # Read from Streamlit secrets if present, else env var, else default
    api_from_secrets = None
    try:
        api_from_secrets = st.secrets.get("API_URL")  # may raise if no secrets file
    except Exception:
        api_from_secrets = None
    return (
        api_from_secrets
        or os.environ.get("API_URL")
        or "http://127.0.0.1:8010"
    )

API_URL = get_api_url().rstrip("/")

@st.cache_data(show_spinner=False)
def load_terms_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            st.warning(f"Failed to read {csv_path.name}: {e}")
    return None

def post_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)

# Optional: try to load a local baseline sklearn pipeline if available
@st.cache_resource(show_spinner=False)
def load_baseline_model() -> Optional[Any]:
    try:
        import joblib  # lazy import
        if BASELINE_PATH.exists():
            return joblib.load(BASELINE_PATH)
    except Exception as e:
        st.info(f"Baseline model not available: {e}")
    return None

def predict_baseline(pipeline, texts: List[str]) -> Dict[str, Any]:
    """Predict using a scikit-learn text pipeline (if available)."""
    out: Dict[str, Any] = {"preds": [], "proba": None}
    try:
        y_pred = pipeline.predict(texts)
        out["preds"] = list(map(lambda x: x if isinstance(x, str) else str(x), y_pred))
        # Try to get probabilities if classifier supports it
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(texts)
            # For a single text, return top-3 classes with their prob
            out["proba"] = proba.tolist()
    except Exception as e:
        out["error"] = f"Baseline prediction failed: {e}"
    return out

# --- UI: tabs ---
tab1, tab2 = st.tabs(["🔍 Model Inference", "📊 Insights"])

# ==============================================================================
# Tab 1: Inference
# ==============================================================================
with tab1:
    st.subheader("Model Inference")

    # Sidebar controls
    with st.sidebar:
        st.markdown("### API Settings")
        st.caption("Change only if your API runs on a different host/port.")
        api_url_input = st.text_input("FastAPI base URL", value=API_URL, help="Example: http://127.0.0.1:8010")
        use_baseline = st.checkbox("Use local baseline model (optional)", value=True, help="Requires model_baseline.pkl")
        show_raw = st.checkbox("Show raw JSON", value=False)
    API_URL = api_url_input.rstrip("/")

    # Input area
    default_text = "This post contains spam links like http://cheap.example"
    text = st.text_area("Enter a review to analyze", value=default_text, height=140)
    k = st.slider("Top-K similar examples", min_value=1, max_value=10, value=5, step=1)

    cols = st.columns([1, 1])
    with cols[0]:
        run_infer = st.button("Analyze")
    with cols[1]:
        st.markdown(f"**API Health:** `{API_URL}/health`")

    # Results
    if run_infer:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            # Call /similar for nearest neighbors
            with st.spinner("Calling /similar ..."):
                try:
                    r = post_json(f"{API_URL}/similar", {"text": text, "k": k}, timeout=60)
                    if r.status_code == 200:
                        sim = r.json()
                        if show_raw:
                            st.code(json.dumps(sim, indent=2)[:5000], language="json")
                        items = sim.get("items", [])
                        st.success(f"Similar examples (k={len(items)}) via {sim.get('vector_backend','?')}")
                        if items:
                            # Pretty table for similar results
                            rows = []
                            for it in items:
                                meta = it.get("meta", {})
                                rows.append({
                                    "index": it.get("index"),
                                    "distance": round(it.get("distance", 0.0), 4),
                                    "reason": meta.get("reason"),
                                    "date_created": meta.get("date_created"),
                                    "object_id": meta.get("object_id"),
                                    "complex_id": meta.get("complex_id"),
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True)
                        else:
                            st.info("No items returned.")
                    else:
                        st.error(f"/similar returned HTTP {r.status_code}: {r.text[:400]}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

            # Optional: local baseline model prediction
            if use_baseline:
                baseline = load_baseline_model()
                if baseline is None:
                    st.info("No local baseline model found (expected notebooks/model_baseline.pkl).")
                else:
                    with st.spinner("Running baseline prediction ..."):
                        res = predict_baseline(baseline, [text])
                        if "error" in res:
                            st.error(res["error"])
                        else:
                            cols2 = st.columns([1, 1])
                            with cols2[0]:
                                st.markdown("**Baseline Predicted Reason**")
                                st.info(res["preds"][0] if res["preds"] else "(no prediction)")
                            with cols2[1]:
                                if res.get("proba"):
                                    st.markdown("**Baseline Probabilities (top classes)**")
                                    # We don't know class order here; show vector length only
                                    st.write(res["proba"][0][:5])
                                else:
                                    st.caption("Classifier does not expose predict_proba()")

# ==============================================================================
# Tab 2: Insights
# ==============================================================================
with tab2:
    st.subheader("Descriptive Insights")

    charts_cols = st.columns(2)
    # Chart: class distribution
    with charts_cols[0]:
        if CHART_DIST.exists():
            st.image(str(CHART_DIST), caption="Class Distribution", use_container_width=True)
        else:
            st.warning(f"Missing: {CHART_DIST.relative_to(ROOT)}")
    # Chart: monthly trend
    with charts_cols[1]:
        if CHART_TREND.exists():
            st.image(str(CHART_TREND), caption="Monthly Trend of Top Categories", use_container_width=True)
        else:
            st.warning(f"Missing: {CHART_TREND.relative_to(ROOT)}")

    st.markdown("---")
    st.markdown("**Top TF-IDF Terms per Reason**")
    df_terms = load_terms_csv(TERMS_CSV)
    if df_terms is not None and not df_terms.empty:
        # Optional small filters
        cols3 = st.columns([2, 2, 1])
        with cols3[0]:
            reasons = sorted(df_terms["reason"].dropna().astype(str).unique().tolist())
            pick = st.multiselect("Filter by reason(s)", reasons, default=reasons[:3])
        with cols3[1]:
            topn = st.number_input("Show top N rows", min_value=5, max_value=200, value=30, step=5)
        out_df = df_terms.copy()
        if pick:
            out_df = out_df[out_df["reason"].astype(str).isin(pick)]
        st.dataframe(out_df.head(int(topn)), use_container_width=True)
        # Download button
        st.download_button(
            label="Download CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="top_terms_per_reason.filtered.csv",
            mime="text/csv",
        )
    else:
        st.warning(f"Missing or empty: {TERMS_CSV.relative_to(ROOT)}")

# --- Footer info ---
st.caption(
    f"API base: `{API_URL}`  •  "
    f"Insights dir: `{INSIGHTS_DIR.relative_to(ROOT)}`  •  "
    f"Baseline model: `{BASELINE_PATH.relative_to(ROOT)}`"
)
