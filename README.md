# 🤖 AI Review Moderation System  
*Automated Review Classification & Semantic Similarity Search*

---

## 🌟 Overview

This project builds an **AI-driven review moderation system** that automatically detects, explains, and retrieves similar problematic user reviews (spam, fake, or inappropriate).  
It combines **ML classification**, **semantic search (FAISS)**, and **Streamlit visualization** into one end-to-end workflow.

> 🧠 Designed & implemented by [I-Ting (Tiffany) Tseng](https://github.com/itingtseng)

---

## 🚀 Live Demo

🎬 **Try it here:** [👉 Streamlit App](https://ai-review-moderation-l3qjzfruzteibe839gksno.streamlit.app)

- 📝 Input a review → get predicted **moderation reason** + confidence  
- 🔍 Retrieve **similar cases** from FAISS vector index  
- 📊 View descriptive **insights** on review patterns and trends  

---

## 🧩 System Architecture

```mermaid
flowchart LR
    A[User Input] --> B[FastAPI Backend]
    B -->|Embedding| C[SentenceTransformer]
    C -->|Vector Search| D[FAISS Index]
    B -->|Classification| E[LogReg Model]
    D --> F[Similar Examples]
    E --> G[Predicted Reason + Confidence]
    F --> H[Streamlit Frontend]
    G --> H
