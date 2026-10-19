"""RAG prompt/context ablation. Run inside the notebook-3 session (uses its globals: qa, retrieved,
by_article, generate, res):

    !cd amharic-legal-rag && git pull -q
    exec(open("amharic-legal-rag/notebooks/03b_rag_ablation.py").read())
"""
import importlib
import json
import time

import amlegal.generate as G
from sacrebleu.metrics import CHRF

importlib.reload(G)
chrf = CHRF()
VARIANTS = {
    "k3_en_strict": dict(k=3, include_english=True, lenient=False),
    "k5_am_strict": dict(k=5, include_english=False, lenient=False),
    "k3_en_lenient": dict(k=3, include_english=True, lenient=True),
    "k3_am_lenient": dict(k=3, include_english=False, lenient=True),
}
abl = []
t0 = time.time()
for name, v in VARIANTS.items():
    for i, row in qa.iterrows():
        msgs = G.build_messages(row.question, retrieved[i][: v["k"]], include_english=v["include_english"],
                                lenient=v["lenient"])
        ans = generate(msgs)
        cited = G.cited_articles(ans)
        abl.append({"i": i, "setup": name, "answer": ans, "cited": cited,
                    "chrF": chrf.sentence_score(ans, [row.answer]).score,
                    "citation_correct": bool(cited) and cited[0] in row.gold,
                    "refusal": G.is_refusal(ans),
                    "gold_in_context": bool({a["article"] for a in retrieved[i][: v["k"]]} & set(row.gold))})
    print(f"{name} done, {time.time() - t0:.0f}s")
abl = pd.DataFrame(abl)
base = res[res.setup.isin(["rag", "oracle"])].copy()
base["gold_in_context"] = [s == "oracle" or bool({a["article"] for a in retrieved[i]} & set(qa.gold[i]))
                           for i, s in zip(base.i, base.setup)]
allres = pd.concat([base[abl.columns], abl])
ablation = allres.groupby("setup").agg(chrF=("chrF", "mean"), citation_accuracy=("citation_correct", "mean"),
                                       refusals=("refusal", "mean"), gold_in_context=("gold_in_context", "mean"))
order = ["rag"] + list(VARIANTS) + ["oracle"]
ablation = ablation.loc[order].round(3)
print(ablation)
os.makedirs("results", exist_ok=True)
ablation.to_csv("results/rag_ablation.csv")
abl[["i", "setup", "answer", "cited", "chrF", "citation_correct", "refusal"]].to_json(
    "results/rag_ablation_answers.jsonl", orient="records", lines=True, force_ascii=False)
os.system("zip -qr ablation_results.zip results/rag_ablation*")
from google.colab import files  # noqa: E402

files.download("ablation_results.zip")
