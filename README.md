# Amharic Legal QA: RAG over Ethiopia's Family Code

[![CI](https://github.com/4brhamy/amharic-legal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/4brhamy/amharic-legal-rag/actions/workflows/ci.yml)

Ask a question about Ethiopian family law **in Amharic** and get a short answer grounded in the law,
with article citations. The system covers all 323 articles of the **Revised Family Code**
(Proclamation No. 213/2000 · አዋጅ ቁጥር 213/1992): marriage, divorce, property, paternity, adoption,
maintenance and guardianship.

Amharic is spoken by over 50 million people but is poorly served by language technology: there is
little clean digital text, models make basic factual and language mistakes, and the official law exists
only as scanned print. This project builds the full pipeline, from a scanned PDF to an evaluated,
deployed question-answering system, and measures every step.

<!-- DEMO_LINK -->

## Results at a glance

| Step | Result |
|---|---|
| OCR + alignment of the scanned law | 323 articles; 92% aligned 1:1, 97% of those confirmed by embeddings |
| Retrieval (hybrid) | correct article in the top 5 for **78%** of questions (MRR 0.65) |
| Gemma 3 4B without retrieval | **0%** correct citations; wrong facts (e.g. marriage age 16 instead of 18) |
| Gemma 3 4B with RAG (final setup) | right **68%** of the time when it answers; only **16%** of all answers wrong |

Every number is measured on 50 Amharic questions whose gold articles were **checked by hand**.

## How it works

```
Scanned Amharic law (99 pages) ──OCR──▶ Amharic articles ─┐
                                                          ├─ align ─▶ 323 bilingual articles
English text of the same law ──────────▶ English articles ┘                │
                                                                           ▼
Amharic question ──▶ hybrid retrieval (BM25 + BGE-M3, Amharic + English) ──▶ top articles
                                                                           │
                                     Gemma 3 4B, answer only from these ◀──┘
                                                                           │
                                               Amharic answer + [Article N] citations
```

### 1. From scanned PDF to a bilingual corpus

The official Amharic text is a 99-page scan with no usable text layer.

- **OCR:** Tesseract with the `amh` model. Article headings are **underlined**, which made Tesseract
  misread their Ethiopic numerals (፱ → "ሀ") and drop some headings entirely. Two fixes, found by testing on
  the problem pages: erase underlines with OpenCV before OCR, and run two page-layout modes (psm 4 and 6)
  and keep, per page, whichever found more headings.
- **Numbering by alignment, not by numerals:** because most numerals were unreadable, each Amharic block is
  aligned to the 323 English articles with dynamic programming (as in Gale & Church sentence alignment)
  using text length, number of sub-articles, and any numeral that did read correctly.
- **Result:** 297/323 articles (92%) align one-to-one. An independent check with multilingual embeddings
  confirms **97%** of those: the Amharic article is closer to its own English article than to its neighbours.
  The rest are flagged (`alignment` field) instead of silently mislabelled.

Amharic-specific text handling (`amlegal/ethiopic.py`): Ethiopic numeral conversion, folding of
homophone letters that are spelled interchangeably (ሀ/ሐ/ኀ/ሃ, ሰ/ሠ, አ/ዐ/ኣ/ዓ, ጸ/ፀ), Ethiopic punctuation,
and a stopword list.

### 2. Retrieval

Evaluated on 50 Amharic questions about the Family Code
([beimnet777/Family-code-of-Ethiopia-QA-TEST](https://huggingface.co/datasets/beimnet777/Family-code-of-Ethiopia-QA-TEST)).
The dataset has answers but no article numbers, so every question was **labelled by hand** with the
article(s) that answer it (`data/test_labels.json`). Automatic labelling would have been wrong for 13 of
the 50 questions.

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| BM25, words | 0.44 | 0.50 | 0.52 | 0.49 |
| BM25, character 3-grams | 0.42 | 0.62 | 0.66 | 0.53 |
| BGE-M3 dense, Amharic text | 0.48 | 0.62 | 0.64 | 0.57 |
| BGE-M3 dense, cross-lingual (Amharic question → English text) | 0.44 | 0.54 | 0.64 | 0.52 |
| **Hybrid: reciprocal rank fusion of all four** | **0.56** | **0.72** | **0.78** | **0.65** |

Each method fails on different questions: keywords miss paraphrases, embeddings of OCR'd text miss
exact terms, and the cross-lingual search avoids OCR errors entirely. Fusing them gives the best result
on every metric.

### 3. Answer generation

The model is **Gemma 3 4B** (instruction-tuned), which lists Amharic among its languages and fits on a
free Colab T4. Answers are scored with **chrF** against the reference answers (character-level, suited
to Amharic morphology) and by whether the **first cited article** is a correct one.

**Grounding is essential.** Without the law in its context the model answered every question and got
none right: wrong facts (marriage age 16 instead of 18), Amharic mixed with Spanish words, and runaway
repetition. Given the correct article, it answers 98% of questions and cites the right article 80% of
the time (the *oracle* row).

**Context design matters as much as retrieval.** The first RAG setup (5 articles, Amharic + English,
strict "answer only from the articles" prompt) declined 64% of questions, even though the right
article was in its context 82% of the time. A 4B model gets lost in ~15,000 characters of legal text.
Six variants were compared on all 50 questions:

| Setup | Answers | Right when it answers | Right overall | Wrong overall |
|---|---|---|---|---|
| Closed-book (no articles) | 100% | 0% | 0% | 100% |
| RAG: 5 articles, AM + EN, strict | 36% | 22% | 8% | 28% |
| 3 articles, AM + EN, strict | 54% | 41% | 22% | 32% |
| 5 articles, AM only, strict | 34% | 47% | 16% | 18% |
| 3 articles, AM + EN, lenient | 42% | 57% | 24% | 18% |
| **3 articles, AM only, lenient (chosen)** | **50%** | **68%** | **34%** | **16%** |
| Two-step: pick 1 of 5 articles, then answer | 76% | 45% | 34% | 42% |
| Top-1 article only | 82% | 44% | 36% | 46% |
| *Oracle: hand-verified articles* | *98%* | *80%* | *78%* | *20%* |

*Lenient* = "use the most relevant article; decline only if none is about the question". "Right" =
the first cited article is a correct one.

The variants that answer more often are wrong in over 40% of all cases. For legal information a wrong
answer is worse than "not found", so the deployed system uses the **chosen** setup and always shows the
three retrieved articles (Amharic and English), so a declined question still returns the relevant law.

**What would close the gap to the oracle:** a cross-encoder reranker to put the right article first, a
larger model (the oracle shows the 4B model's ceiling at ~80%), and correcting the OCR'd text.

## Repository

```
amlegal/            library: Ethiopic text tools, OCR, alignment, retrieval, prompting, FastAPI app
notebooks/          Colab notebooks: 01 build corpus (OCR), 02 retrieval eval, 03 generation eval (+ 03b ablation)
scripts/            build_corpus.py, build_space.sh
data/               family_code.jsonl (323 bilingual articles), test_labels.json, raw OCR output
results/            evaluation outputs
space/              Hugging Face demo (Gradio)
tests/              41 unit and API tests
```

## Run it

```bash
pip install -r requirements-dev.txt
pytest                                   # 41 tests
uvicorn amlegal.api:app --port 8000      # API: POST /search, POST /ask, GET /health
```

Docker (keyword retrieval only, CPU):

```bash
docker build -t amharic-legal-rag .
docker run -p 8000:8000 amharic-legal-rag
curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
     -d '{"question": "በጋብቻ ላይ ጋብቻ መፈጸም ይቻላል?", "k": 3}'
```

Set `AMLEGAL_EMBEDDINGS=1` for hybrid retrieval and `AMLEGAL_LLM=unsloth/gemma-3-4b-it` (GPU) for answers.
CI runs lint, the test suite, the Docker build and a container smoke test on every push.

## Limitations

- One law so far. The pipeline (OCR → alignment → retrieval) is reusable for other proclamations.
- The Amharic text comes from OCR: body text is good, but some characters and numerals are wrong, and
  26 articles have uncertain alignment. The English text and citations help users check answers.
- The test set is small (50 questions), so differences of a few points are within noise.
- Not legal advice. The official Negarit Gazeta text prevails.

## Related work

- RAG-C (Knowledge-Based Systems, 2026): Amharic legal QA over Federal Supreme Court cassation decisions.
- AmQA (2023): Amharic reading-comprehension QA from Wikipedia.

## License

Code: MIT. The law text is a public legal document. The test questions belong to their authors and are
not redistributed here; only article labels are.
