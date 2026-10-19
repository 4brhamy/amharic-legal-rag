# %% [markdown]
# # 3. Answer generation: does retrieval help the model answer in Amharic?
#
# An open LLM answers each test question in three setups:
#
# | Setup | Model sees | Purpose |
# |---|---|---|
# | **Closed-book** | the question only | What the model "knows" about Ethiopian family law by itself |
# | **RAG** | top-5 articles from the hybrid retriever | The real system |
# | **Oracle** | the hand-verified correct articles | Upper bound: answer quality if retrieval were perfect |
#
# **Model:** Gemma 3 4B instruction-tuned (`unsloth/gemma-3-4b-it`, an ungated copy of Google's weights;
# use is subject to the Gemma Terms of Use). Gemma 3 lists Amharic among its supported languages and
# fits on a free T4.
#
# **Metrics**
# - **chrF** against the reference answer (character n-gram F-score; works well for Amharic, where
#   one word carries many affixes).
# - **Citation accuracy:** share of answers whose first cited article is a correct article.
# - **Refusals:** share of answers saying the articles don't contain the answer.
#
# **Runtime:** GPU (T4). About 15–25 minutes.

# %%
!pip -q install sentence-transformers sacrebleu pandas pyarrow "transformers>=4.50" accelerate bitsandbytes
!rm -rf amharic-legal-rag && git clone -q https://github.com/4brhamy/amharic-legal-rag.git
import sys, json, time
sys.path.insert(0, "amharic-legal-rag")
import numpy as np, pandas as pd, torch
from amlegal.retrieval import HybridRetriever
from amlegal.generate import build_messages, cited_articles, is_refusal

corpus = [json.loads(l) for l in open("amharic-legal-rag/data/family_code.jsonl", encoding="utf-8")]
by_article = {r["article"]: r for r in corpus}
qa = pd.read_parquet("hf://datasets/beimnet777/Family-code-of-Ethiopia-QA-TEST/data/train-00000-of-00001.parquet")
qa.columns = ["question", "answer"]
gold = json.load(open("amharic-legal-rag/data/test_labels.json"))
qa["gold"] = [gold[str(i)] for i in range(len(qa))]
print(len(qa), "questions")

# %% [markdown]
# ## Retriever (best method from notebook 2)

# %%
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("BAAI/bge-m3")
encode = lambda texts: embedder.encode(texts, normalize_embeddings=True, batch_size=16)
retriever = HybridRetriever(corpus, encode=encode)
retrieved = [retriever.search(q, k=5) for q in qa.question]
hits = sum(bool({a["article"] for a in r} & set(g)) for r, g in zip(retrieved, qa.gold))
print(f"Correct article in the top 5 for {hits}/{len(qa)} questions")
del embedder; torch.cuda.empty_cache()

# %% [markdown]
# ## Load Gemma 3 4B
# Gemma 3 overflows in float16, and the T4 has no native bfloat16. So on a T4 the weights are loaded
# in 4-bit with float32 computation; on newer GPUs plain bfloat16 is used.

# %%
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
MODEL = "unsloth/gemma-3-4b-it"
processor = AutoProcessor.from_pretrained(MODEL)
if torch.cuda.is_bf16_supported(including_emulation=False):
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
else:
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32)
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, quantization_config=q, device_map="auto",
                                                           torch_dtype=torch.float32)
model.eval()

def generate(messages, max_new_tokens=220):
    msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
    inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

print(generate([{"role": "user", "content": "በኢትዮጵያ የቤተሰብ ሕግ መሠረት ዝቅተኛው የጋብቻ ዕድሜ ስንት ነው?"}]))

# %% [markdown]
# ## Run the three setups

# %%
CLOSED_BOOK = ("አንተ የኢትዮጵያን የቤተሰብ ሕግ የምታውቅ ረዳት ነህ። ጥያቄውን በአማርኛ በአጭሩ መልስ፤ "
               "የምታውቀውን አንቀጽ በ [አንቀጽ ቁጥር] መልክ ጥቀስ።")
rows = []
t0 = time.time()
for i, row in qa.iterrows():
    setups = {
        "closed_book": [{"role": "system", "content": CLOSED_BOOK}, {"role": "user", "content": row.question}],
        "rag": build_messages(row.question, retrieved[i]),
        "oracle": build_messages(row.question, [by_article[a] for a in row.gold]),
    }
    for name, messages in setups.items():
        ans = generate(messages)
        rows.append({"i": i, "setup": name, "answer": ans})
    if i % 10 == 9:
        print(f"{i + 1}/{len(qa)} questions, {time.time() - t0:.0f}s")
res = pd.DataFrame(rows)

# %% [markdown]
# ## Score

# %%
from sacrebleu.metrics import CHRF
chrf = CHRF()
res["reference"] = res.i.map(qa.answer)
res["gold"] = res.i.map(qa.gold)
res["chrF"] = [chrf.sentence_score(a, [r]).score for a, r in zip(res.answer, res.reference)]
res["cited"] = res.answer.map(cited_articles)
res["citation_correct"] = [bool(c) and c[0] in g for c, g in zip(res.cited, res.gold)]
res["cites_any"] = res.cited.map(bool)
res["refusal"] = res.answer.map(is_refusal)
summary = res.groupby("setup").agg(chrF=("chrF", "mean"), citation_accuracy=("citation_correct", "mean"),
                                   answers_with_citation=("cites_any", "mean"), refusals=("refusal", "mean"))
summary = summary.loc[["closed_book", "rag", "oracle"]].round(3)
summary

# %% [markdown]
# ## Examples

# %%
for i in [2, 7, 37, 42]:
    print(f"\n=== Q{i}: {qa.question[i]}\nReference: {qa.answer[i][:200]}\nGold article(s): {qa.gold[i]}")
    for s in ["closed_book", "rag", "oracle"]:
        a = res[(res.i == i) & (res.setup == s)].answer.iloc[0]
        print(f"--- {s}: {a[:400]}")

# %% [markdown]
# ## Save
# Only article numbers, scores and model answers are saved (not the test questions, which have no
# stated licence).

# %%
import os
os.makedirs("results", exist_ok=True)
summary.to_csv("results/generation_summary.csv")
res[["i", "setup", "answer", "cited", "chrF", "citation_correct", "refusal"]].to_json(
    "results/generation_answers.jsonl", orient="records", lines=True, force_ascii=False)
!zip -qr generation_results.zip results && ls -la generation_results.zip
from google.colab import files
files.download("generation_results.zip")
