# 🧾 Model Card — AI Review Moderation

### Model Name
`model_baseline.pkl` — Logistic Regression trained on SentenceTransformer embeddings.

---

### 🧠 Intended Use
- Detect likely **policy-violating** or **low-quality** user reviews.
- Assist trust & safety reviewers with case retrieval.
- Serve as baseline for later fine-tuned transformer models.

---

### ⚙️ Training Configuration
| Item | Description |
|------|-------------|
| Framework | scikit-learn |
| Input features | Sentence embeddings (`all-MiniLM-L6-v2`) |
| Dataset | Internal review corpus, manually labeled |
| Labels | spam, fake, promotional, inappropriate, none |
| Split | 80/10/10 train/val/test |

---

### 📈 Evaluation Metrics
| Metric | Score |
|---------|-------|
| Accuracy | 0.87 |
| Precision | 0.85 |
| Recall | 0.82 |
| F1 | 0.83 |

---

### ⚠️ Limitations
- Underperforms on short reviews (< 10 words)
- May misclassify sarcasm or mixed sentiment
- Trained primarily on English data

---

### 🔐 Ethical Considerations
- Excludes personally identifiable info (PII)
- Manual labels verified by human annotators
- Model decisions not used for final moderation without review

---

### 🧩 Versioning
| Version | Change | Date |
|----------|---------|------|
| v1.0 | Baseline logistic model | 2025-10-15 |
