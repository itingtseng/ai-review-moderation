# 📊 Insights Report – AI Review Moderation

## Overview
This report summarizes descriptive analysis and observations derived from the review moderation dataset. The goal is to identify key content risk types, trends, and actionable governance insights.

---

## 1. Descriptive Findings

### 1.1 Distribution by Reason
- The most frequent moderation reasons are **Spam / Promotional Content**, followed by **Inappropriate Language** and **Wrong Community Posts**.
- Approximately 60% of flagged posts belong to these top 3 categories.

### 1.2 Temporal Trends
- A noticeable spike in spam-related content occurred during **July–August 2020**, aligning with user growth surges in several communities.
- Posts labeled as *affiliated with the community* remain stable over time, suggesting consistent internal posting patterns.

### 1.3 Textual Characteristics
- TF-IDF analysis reveals that spam reviews often include URLs, phone numbers, and commercial keywords.
- Inappropriate or hateful posts contain high-weight tokens associated with emotional tone and intensity words.

---

## 2. PM Insights & Governance Suggestions

| Area | Observation | PM Recommendation |
|------|--------------|------------------|
| **Spam detection** | Repetitive URL / phone patterns | Add regex-based rule filter for URLs and contact numbers |
| **Hate speech** | Strong polarity words dominate | Add sensitive-word lexicon and confidence threshold for manual review |
| **Community affiliation** | Many flagged posts originate from internal testers | Enhance labeling guideline and audit reviewer consistency |
| **Model drift** | Certain term weights changed post-2020 | Schedule retraining every quarter and track vocabulary drift |
| **Data transparency** | Users uncertain about removal rationale | Display reason summaries in moderation UI for transparency |

---

## 3. Next Steps
- Integrate these insights into moderation policy.
- Enable automatic retraining via MLOps pipeline.
- Evaluate post-deployment feedback to refine precision and recall targets.

---
