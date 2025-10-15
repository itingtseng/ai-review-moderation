Author: I-Ting (Tiffany) Tseng  
Date: 2025-10-15  
Version: v1.0  

# 📘 Product Requirement Document (PRD)

## Project: AI Review Moderation

### 1. 🎯 Overview
This project aims to detect and explain **problematic user reviews** (spam, fake, promotional, or offensive) using machine learning and semantic search.  
It provides **automated moderation reasoning** and **retrieval of similar historical cases** to support trust & safety teams.

---

### 2. 🧩 Objectives & KPIs

| Objective | KPI |
|------------|-----|
| Detect risky reviews automatically | ≥ 85% precision, ≤ 10% false negatives |
| Provide interpretable moderation | Each flagged review includes reason + similar examples |
| Support scalable deployment | <3s response time per request |
| Enable human review escalation | Confidence threshold configurable |

---

### 3. 👥 Target Users

- Trust & Safety analysts  
- Community managers  
- Product PMs overseeing content moderation pipelines

---

### 4. 💡 User Flow

1. User pastes a review into input box.  
2. Model predicts likely moderation reason and confidence.  
3. Similar examples (with existing moderation labels) are displayed.  
4. Analyst can validate or escalate low-confidence results.

---

### 5. ⚙️ Technical Summary

| Component | Description |
|------------|--------------|
| Model | Logistic Regression + SentenceTransformer embeddings |
| Backend | FastAPI microservice (`/predict`, `/similar`) |
| Vector DB | FAISS index + metadata map |
| Frontend | Streamlit (single-page app) |

---

### 6. 🧭 Success Criteria

- Response latency < 3s  
- Accuracy ≥ baseline F1 = 0.80  
- End-to-end system deployable via Streamlit Cloud  
- Stakeholders (PM / DS) can interpret model decisions

---

### 7. 🚫 Constraints & Risks

- Dataset imbalance (spam class <10%)  
- Latent bias in language (certain adjectives may overflag)  
- Cold-start latency for first inference  
- Large FAISS file may exceed free Streamlit limits

---

### 8. 📆 Milestones

| Phase | Deliverable | Timeline |
|-------|--------------|----------|
| 1 | Data Cleaning + Labeling | Day 1–2 |
| 2 | Model Training + Validation | Day 3–4 |
| 3 | API + Vector Index Setup | Day 5 |
| 4 | Streamlit Demo Deployment | Day 6 |
| 5 | Reports (PRD, Model Card, Ethics) | Day 7 |
