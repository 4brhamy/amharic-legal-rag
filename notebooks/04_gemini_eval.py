# %% [markdown]
# # 4. Hosted model: Gemini vs Gemma 3 4B
#
# The public demo answers with Google's Gemini API instead of running Gemma in the Space (the free
# Space gets no GPU). This notebook evaluates Gemini on the same 50 hand-labelled questions and the
# same setups as notebook 3, so the two models can be compared directly.
#
# **Before running:** add your API key as a Colab secret named `GEMINI_API_KEY` (🔑 icon in the left
# sidebar, enable notebook access). **Runtime:** CPU is fine. Requests are paced to stay within the
# free-tier rate limit, so the run takes about 30 minutes.
#
# Note: on the free tier, Google may use prompts to improve its products. The prompts here contain
# only public law text and the public test questions.

# %%
!pip -q install sentence-transformers sacrebleu google-genai pandas pyarrow
!rm -rf amharic-legal-rag && git clone -q https://github.com/4brhamy/amharic-legal-rag.git
import sys, json, time, os
sys.path.insert(0, "amharic-legal-rag")
import numpy as np, pandas as pd
from google.colab import userdata
from amlegal.retrieval import HybridRetriever
from amlegal.generate import build_messages, cited_articles, is_refusal
from amlegal.llm import gemini_generator, DEFAULT_GEMINI_MODEL

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
MODEL = DEFAULT_GEMINI_MODEL
generate = gemini_generator(model=MODEL)
print(MODEL, "->", generate([{"role": "user", "content": "ሰላም! በአንድ ቃል መልስ፦ የኢትዮጵያ ዋና ከተማ ማን ናት?"}], 20))

corpus = [json.loads(l) for l in open("amharic-legal-rag/data/family_code.jsonl", encoding="utf-8")]
by_article = {r["article"]: r for r in corpus}
qa = pd.read_parquet("hf://datasets/beimnet777/Family-code-of-Ethiopia-QA-TEST/data/train-00000-of-00001.parquet")
qa.columns = ["question", "answer"]
gold = json.load(open("amharic-legal-rag/data/test_labels.json"))
qa["gold"] = [gold[str(i)] for i in range(len(qa))]

# %% [markdown]
# ## Retrieval (same hybrid retriever as the demo; BGE-M3 on CPU takes a few minutes)

# %%
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
retriever = HybridRetriever(corpus, encode=lambda t: embedder.encode(t, normalize_embeddings=True, batch_size=16))
retrieved = [retriever.search(q, k=5) for q in qa.question]
print("gold article in top 5:", sum(bool({a["article"] for a in r} & set(g)) for r, g in zip(retrieved, qa.gold)), "/ 50")

# %% [markdown]
# ## Answer the 50 questions in four setups

# %%
CLOSED_BOOK = ("አንተ የኢትዮጵያን የቤተሰብ ሕግ የምታውቅ ረዳት ነህ። ጥያቄውን በአማርኛ በአጭሩ መልስ፤ "
               "የምታውቀውን አንቀጽ በ [አንቀጽ ቁጥር] መልክ ጥቀስ።")
SETUPS = {
    "closed_book": lambda q, i: [{"role": "system", "content": CLOSED_BOOK}, {"role": "user", "content": q}],
    "rag_k5_en_strict": lambda q, i: build_messages(q, retrieved[i]),
    "rag_k3_am_lenient": lambda q, i: build_messages(q, retrieved[i][:3], include_english=False, lenient=True),
    "oracle": lambda q, i: build_messages(q, [by_article[a] for a in qa.gold[i]]),
}
PAUSE = 6.5  # seconds between requests: stays under ~10 requests/minute on the free tier
rows, t0 = [], time.time()
for name, make in SETUPS.items():
    for i, q in enumerate(qa.question):
        try:
            ans = generate(make(q, i))
        except Exception as e:
            ans = f"ERROR: {type(e).__name__}: {e}"[:300]
        rows.append({"i": i, "setup": name, "answer": ans})
        time.sleep(PAUSE)
    print(f"{name} done, {time.time() - t0:.0f}s")
res = pd.DataFrame(rows)
print("errors:", res.answer.str.startswith("ERROR").sum())

# %% [markdown]
# ## Score (same metrics as notebook 3)

# %%
from sacrebleu.metrics import CHRF
chrf = CHRF()
res["gold"] = res.i.map(qa.gold)
res["chrF"] = [chrf.sentence_score(a, [qa.answer[i]]).score for a, i in zip(res.answer, res.i)]
res["cited"] = res.answer.map(cited_articles)
res["refusal"] = res.answer.map(is_refusal)
res["citation_correct"] = [bool(c) and c[0] in g for c, g in zip(res.cited, res.gold)]
res["answered"] = ~res.refusal
summary = res.groupby("setup").agg(
    chrF=("chrF", "mean"), answers=("answered", "mean"),
    right_overall=("citation_correct", lambda s: (s & res.loc[s.index, "answered"]).mean()),
    refusals=("refusal", "mean"))
summary["right_when_answering"] = summary.right_overall / summary.answers
summary["wrong_overall"] = summary.answers - summary.right_overall
summary = summary.loc[list(SETUPS)].round(3)
summary

# %%
for i in [7, 37, 42]:
    print(f"\n=== Q{i}: {qa.question[i]}\ngold {qa.gold[i]}")
    for s in SETUPS:
        print(f"--- {s}: {res[(res.i == i) & (res.setup == s)].answer.iloc[0][:300]}")

# %%
os.makedirs("results", exist_ok=True)
summary.to_csv(f"results/gemini_summary.csv")
res[["i", "setup", "answer", "cited", "chrF", "citation_correct", "refusal"]].to_json(
    "results/gemini_answers.jsonl", orient="records", lines=True, force_ascii=False)
with open("results/gemini_model.txt", "w") as f:
    f.write(MODEL + "\n")
!zip -qr gemini_results.zip results && ls -la gemini_results.zip
from google.colab import files
files.download("gemini_results.zip")
