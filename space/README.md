---
title: Amharic Legal QA
emoji: ⚖️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Ask questions about Ethiopian family law in Amharic
---

# Amharic Legal QA: Ethiopian Family Code

Ask a question in Amharic about Ethiopia's Revised Family Code (Proclamation No. 213/2000) and get the
articles of the law that answer it, in Amharic and English.

- **Retrieval:** hybrid search (BM25 on words and character 3-grams + BGE-M3 embeddings over the
  Amharic and English text), fused with reciprocal rank fusion.
- **Answering (not in this public demo):** Gemma 3 4B answering only from the retrieved articles was
  evaluated in the repository; it needs a GPU, which this free Space does not get.

⚠️ General legal information, not legal advice. The Amharic text was produced by OCR and may contain
errors; the official Negarit Gazeta text prevails.

Code, data pipeline and evaluation: https://github.com/4brhamy/amharic-legal-rag
