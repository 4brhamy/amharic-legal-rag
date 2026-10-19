# %% [markdown]
# # 2. Retrieval: which method finds the right article?
#
# Given an Amharic question, the system must find the article of the Family Code that answers it.
# This notebook compares:
#
# | Method | Searches | Idea |
# |---|---|---|
# | **BM25 (words)** | Amharic text | Classic keyword search on normalised Amharic words |
# | **BM25 (char 3-grams)** | Amharic text | Keyword search on character trigrams, robust to Amharic affixes and OCR errors |
# | **Dense, Amharic** | Amharic text | Multilingual embeddings (BGE-M3) of question and article |
# | **Dense, cross-lingual** | English text | Amharic question vs the clean English article, immune to OCR errors |
# | **Hybrid** | both | Reciprocal rank fusion of the methods above |
#
# **Test set:** 50 Amharic question–answer pairs about the Family Code
# ([beimnet777/Family-code-of-Ethiopia-QA-TEST](https://huggingface.co/datasets/beimnet777/Family-code-of-Ethiopia-QA-TEST)).
# The set gives answers but not article numbers, so the gold article for each question is labelled
# automatically (section 2) and those labels are reported separately as *silver* labels.
#
# **Runtime:** GPU (T4) recommended for the embedding model.

# %%
!pip -q install sentence-transformers pandas pyarrow
!rm -rf amharic-legal-rag && git clone -q https://github.com/4brhamy/amharic-legal-rag.git
import sys, json
sys.path.insert(0, "amharic-legal-rag")
import numpy as np, pandas as pd
from amlegal.retrieval import BM25, reciprocal_rank_fusion, recall_at_k, mrr

corpus = [json.loads(l) for l in open("amharic-legal-rag/data/family_code.jsonl", encoding="utf-8")]
qa = pd.read_parquet("hf://datasets/beimnet777/Family-code-of-Ethiopia-QA-TEST/data/train-00000-of-00001.parquet")
qa.columns = ["question", "answer"]
print(len(corpus), "articles,", len(qa), "questions")
qa.head()

# %% [markdown]
# ## 1. Embed every article (Amharic and English) and the questions
# BGE-M3 is a multilingual embedding model built on XLM-RoBERTa, whose pre-training includes Amharic.

# %%
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
am_docs = [f"{r['title_am']}\n{r['text_am']}" for r in corpus]
en_docs = [f"{r['title_en']}\n{r['text_en']}" for r in corpus]
E_am = model.encode(am_docs, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
E_en = model.encode(en_docs, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
E_q = model.encode(qa.question.tolist(), normalize_embeddings=True)
E_ans = model.encode(qa.answer.tolist(), normalize_embeddings=True)

# %% [markdown]
# ## 2. Check the article alignment, and label the test set
# **Alignment check.** If Amharic article *i* really is the translation of English article *i*, their
# embeddings should be more similar to each other than to the neighbouring articles.

# %%
sim = E_am @ E_en.T
checks = []
for i, r in enumerate(corpus):
    window = range(max(0, i - 3), min(len(corpus), i + 4))
    best = max(window, key=lambda j: sim[i, j])
    checks.append({"article": r["article"], "alignment": r["alignment"], "near_flagged": r["near_flagged"],
                   "own_is_best": best == i})
checks = pd.DataFrame(checks)
print("Amharic article most similar to its own English article (within ±3):")
print(checks.groupby("alignment").own_is_best.agg(["mean", "count"]))
print("1:1 next to a flagged article:", round(checks[checks.near_flagged].own_is_best.mean(), 3))

# %% [markdown]
# **Candidate labels.** Each reference answer is usually a close paraphrase of the law, so the article
# whose Amharic text is most similar to the answer (embeddings + character BM25, fused) is a good
# candidate for the gold article.
#
# ⚠️ These candidates come from the same kinds of signals the retrieval methods use, so evaluating on
# them alone would favour those methods. They are therefore **checked by hand** against the law
# (`data/test_labels.json`, see the README); when that file exists, the verified labels are used.

# %%
bm_char = BM25(am_docs, char_ngrams=3)
gold, uncertain = [], []
for i, ans in enumerate(qa.answer):
    dense_rank = list(np.argsort(-(E_am @ E_ans[i])))[:10]
    bm_rank = bm_char.search(ans, k=10)
    fused = reciprocal_rank_fusion([dense_rank, bm_rank], top=2)
    gold.append({int(fused[0])})
    s = sorted((E_am @ E_ans[i])[fused], reverse=True)
    uncertain.append(len(s) > 1 and s[0] - s[1] < 0.02)
qa["candidate_article"] = [corpus[next(iter(g))]["article"] for g in gold]
qa["uncertain"] = uncertain
print(f"{sum(uncertain)} of {len(qa)} candidate labels are uncertain")

import os
labels_path = "amharic-legal-rag/data/test_labels.json"
if os.path.exists(labels_path):
    verified = json.load(open(labels_path))  # {"question index": [article numbers]}
    index = {r["article"]: i for i, r in enumerate(corpus)}
    gold = [{index[a] for a in verified[str(i)]} for i in range(len(qa))]
    qa["gold_article"] = [sorted(verified[str(i)]) for i in range(len(qa))]
    qa["uncertain"] = False
    print("Using hand-verified labels")
else:
    qa["gold_article"] = qa.candidate_article
    print("No verified labels yet: results below use unverified candidates")
qa[["question", "candidate_article", "uncertain"]].head(10)

# %% [markdown]
# ## 3. Compare retrieval methods

# %%
K = 10
bm_word = BM25(am_docs)
runs = {
    "BM25 (words)": [bm_word.search(q, k=K) for q in qa.question],
    "BM25 (char 3-grams)": [bm_char.search(q, k=K) for q in qa.question],
    "Dense, Amharic": [list(np.argsort(-(E_am @ e)))[:K] for e in E_q],
    "Dense, cross-lingual (EN)": [list(np.argsort(-(E_en @ e)))[:K] for e in E_q],
}
runs["Hybrid (all)"] = [reciprocal_rank_fusion([r[i] for r in runs.values()], top=K) for i in range(len(qa))]
runs["Hybrid (dense AM + EN)"] = [
    reciprocal_rank_fusion([runs["Dense, Amharic"][i], runs["Dense, cross-lingual (EN)"][i]], top=K)
    for i in range(len(qa))]

def evaluate(mask):
    g = [x for x, m in zip(gold, mask) if m]
    rows = []
    for name, ranked in runs.items():
        r = [x for x, m in zip(ranked, mask) if m]
        rows.append({"method": name, "R@1": recall_at_k(r, g, 1), "R@3": recall_at_k(r, g, 3),
                     "R@5": recall_at_k(r, g, 5), "MRR": mrr(r, g)})
    return pd.DataFrame(rows).set_index("method").round(3)

all_q = [True] * len(qa)
confident = [not u for u in qa.uncertain]
print(f"All {len(qa)} questions:")
display(evaluate(all_q))
print(f"\nOnly the {sum(confident)} questions with confident labels:")
display(evaluate(confident))

# %% [markdown]
# ## 4. Save results

# %%
import os
os.makedirs("results", exist_ok=True)
evaluate(all_q).to_csv("results/retrieval_all.csv")
evaluate(confident).to_csv("results/retrieval_confident.csv")
checks.to_csv("results/alignment_check.csv", index=False)
# Candidate labels with article titles and top-5 alternatives, for checking by hand
review = []
for i, ans in enumerate(qa.answer):
    top = reciprocal_rank_fusion([list(np.argsort(-(E_am @ E_ans[i])))[:10], bm_char.search(ans, k=10)], top=5)
    review.append({"i": i, "question": qa.question[i], "answer": ans,
                   "candidates": [{"article": corpus[j]["article"], "title_am": corpus[j]["title_am"],
                                   "title_en": corpus[j]["title_en"]} for j in top]})
json.dump(review, open("results/label_review.json", "w"), ensure_ascii=False, indent=1)
!zip -qr retrieval_results.zip results && ls -la retrieval_results.zip
from google.colab import files
files.download("retrieval_results.zip")
