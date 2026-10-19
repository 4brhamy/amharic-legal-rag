"""Gradio demo: Amharic question answering over Ethiopia's Revised Family Code."""
import json
import os

import gradio as gr
import numpy as np

from amlegal.generate import NOT_FOUND, build_messages, cited_articles, is_refusal
from amlegal.retrieval import HybridRetriever

try:
    import spaces  # Hugging Face ZeroGPU
    gpu = spaces.GPU(duration=60)
except ImportError:  # running locally
    def gpu(fn):
        return fn

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "unsloth/gemma-3-4b-it"
# Chosen by the RAG ablation (results/rag_ablation.csv): 3 articles, Amharic text only, lenient
# prompt. It answers half the questions and is right 68% of the time when it does; variants that
# answer more often are wrong in over 40% of all cases.
K = 3           # articles the model reads
SHOW = 5        # articles shown to the user (a relevant article ranked 4th-5th is still visible)
LENIENT = True
INCLUDE_ENGLISH = False

with open(os.path.join(HERE, "data", "family_code.jsonl"), encoding="utf-8") as f:
    CORPUS = [json.loads(line) for line in f]

# --- retrieval (CPU) ---
from sentence_transformers import SentenceTransformer  # noqa: E402

_embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")


def encode(texts):
    return np.asarray(_embedder.encode(texts, normalize_embeddings=True, batch_size=16))


RETRIEVER = HybridRetriever(CORPUS, encode=encode)

# --- generation ---
# Loaded at startup (download + weights to GPU) so the 60-second ZeroGPU window is only used for
# generating, never for downloading the 8.6 GB model.
import torch  # noqa: E402
from transformers import AutoProcessor, Gemma3ForConditionalGeneration  # noqa: E402

_processor = AutoProcessor.from_pretrained(MODEL_ID)
_model = Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).eval()
# ZeroGPU's documented pattern: move to "cuda" at startup without querying the GPU. Calling
# torch.cuda.is_available() here initialises CUDA in the main process, which breaks the forked GPU
# workers with "No CUDA GPUs are available".
if "spaces" in globals():
    _model.to("cuda")


@gpu
def generate(messages, max_new_tokens=250):
    msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
    inputs = _processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(_model.device)
    with torch.inference_mode():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return _processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def article_card(a, cited, read=True):
    mark = "✅ " if a["article"] in cited else ("" if read else "➕ ")
    flag = "" if a["alignment"] == "1:1" else "  _(OCR alignment uncertain)_"
    return (f"#### {mark}Article {a['article']}: {a['title_am']} / {a['title_en']}{flag}\n\n"
            f"{a['text_am']}\n\n<details><summary>English</summary>\n\n{a['text_en']}\n\n</details>")


def answer(question):
    question = (question or "").strip()
    if len(question) < 3:
        return "እባክዎ ጥያቄ ያስገቡ። (Please enter a question.)", ""
    shown = RETRIEVER.search(question, k=SHOW)
    articles = shown[:K]
    try:
        text = generate(build_messages(question, articles, include_english=INCLUDE_ENGLISH, lenient=LENIENT))
    except Exception as e:  # e.g. ZeroGPU quota exhausted: still show the relevant articles
        text = f"_(The answer model is unavailable right now: {type(e).__name__}. The most relevant articles are below.)_"
    valid = {a["article"] for a in articles}
    cited = [n for n in cited_articles(text) if n in valid]
    if is_refusal(text):
        text += "\n\n_The model did not find the answer in the retrieved articles; check the articles below._"
    sources = "\n\n---\n\n".join(article_card(a, cited, read=i < K) for i, a in enumerate(shown))
    return text, sources


EXAMPLES = [
    "በጋብቻ ላይ ጋብቻ መፈጸም ይቻላል?",
    "ጋብቻ ለመፈጸም ዝቅተኛው ዕድሜ ስንት ነው?",
    "በወንድሞችና በእህቶች መካከል ቀለብ የመስጠት ግዴታ አለ?",
    "ፍቺ እንዴት ሊጠየቅ ይችላል?",
    "ሞግዚት የልጁን ካፒታል በስንት ጊዜ ውስጥ በአትራፊ ሥራ ላይ ማዋል አለበት?",
]

with gr.Blocks(title="Amharic Legal QA", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ⚖️ የቤተሰብ ሕግ ጠያቂ · Amharic Legal QA\n"
        "Ask about Ethiopia's **Revised Family Code** (Proclamation 213/2000) in Amharic. Answers are "
        "generated only from the retrieved articles and cite them.\n\n"
        "_General legal information, not legal advice. The Amharic text comes from OCR and may contain "
        "errors; the official Negarit Gazeta text prevails._")
    q = gr.Textbox(label="ጥያቄ (Question)", placeholder="ለምሳሌ፦ በጋብቻ ላይ ጋብቻ መፈጸም ይቻላል?", lines=2)
    btn = gr.Button("ጠይቅ (Ask)", variant="primary")
    out = gr.Markdown(label="መልስ (Answer)")
    with gr.Accordion("Relevant articles (✅ cited in the answer · ➕ also relevant, not read by the model)", open=True):
        src = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=q)
    btn.click(answer, q, [out, src])
    q.submit(answer, q, [out, src])
    gr.Markdown(f"Model: Gemma 3 4B · Retrieval: BM25 + BGE-M3 hybrid · "
                f"[Code and evaluation](https://github.com/4brhamy/amharic-legal-rag) · "
                f"If no article answers the question the model says: “{NOT_FOUND}”.")

if __name__ == "__main__":
    demo.launch()
