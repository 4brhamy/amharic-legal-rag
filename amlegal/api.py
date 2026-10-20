"""HTTP API for Amharic legal question answering.

Run:  uvicorn amlegal.api:app --port 8000

Configuration (environment variables):
  AMLEGAL_CORPUS     path to the corpus JSONL (default: data/family_code.jsonl)
  AMLEGAL_EMBEDDINGS "1" to add BGE-M3 dense retrieval (needs sentence-transformers; ~2 GB model)
  AMLEGAL_LLM        "gemini" (or a Gemini model id) for hosted answers via GEMINI_API_KEY, no GPU needed;
                     or a Hugging Face model id such as "unsloth/gemma-3-4b-it" (transformers + a GPU).
                     Without it, /ask returns the articles only.
"""
import json
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import build_messages, cited_articles, is_refusal
from .retrieval import HybridRetriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISCLAIMER = ("This is general legal information from the Revised Family Code, not legal advice. "
              "The Amharic text comes from OCR and may contain errors; the official Negarit Gazeta "
              "text prevails.")

app = FastAPI(title="Amharic Legal QA", version="0.1.0",
              description="Ask questions about Ethiopia's Revised Family Code in Amharic.")


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["የጋብቻ ዕድሜ ስንት ነው?"])
    k: int = Field(3, ge=1, le=10)


class Article(BaseModel):
    article: int
    title_am: str
    title_en: str
    text_am: str
    text_en: str
    alignment: str


class SearchResponse(BaseModel):
    articles: list[Article]
    disclaimer: str = DISCLAIMER


class AskResponse(SearchResponse):
    answer: str | None
    cited_articles: list[int]
    refused: bool


@lru_cache
def get_retriever() -> HybridRetriever:
    path = os.environ.get("AMLEGAL_CORPUS", os.path.join(ROOT, "data", "family_code.jsonl"))
    with open(path, encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    encode = None
    if os.environ.get("AMLEGAL_EMBEDDINGS") == "1":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        encode = lambda texts: model.encode(texts, normalize_embeddings=True, batch_size=16)
    return HybridRetriever(corpus, encode=encode)


@lru_cache
def get_generator():
    """Return a function messages -> answer text, or None when no LLM is configured."""
    model_id = os.environ.get("AMLEGAL_LLM")
    if not model_id:
        return None
    if model_id.startswith("gemini"):  # hosted model, no GPU: AMLEGAL_LLM=gemini or gemini-2.5-flash
        from .llm import gemini_generator
        return gemini_generator(model=None if model_id == "gemini" else model_id)
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map="auto").eval()

    def generate(messages, max_new_tokens=250):
        msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    return generate


def _articles(question: Question) -> list[dict]:
    return get_retriever().search(question.question, k=question.k)


def _public(a: dict) -> Article:
    return Article(**{f: a[f] for f in Article.model_fields})


@app.get("/health")
def health():
    return {"status": "ok", "articles": len(get_retriever().corpus), "llm": bool(os.environ.get("AMLEGAL_LLM"))}


@app.post("/search", response_model=SearchResponse)
def search(q: Question):
    """Find the articles most relevant to an Amharic question (hybrid BM25 + embedding search)."""
    return SearchResponse(articles=[_public(a) for a in _articles(q)])


@app.post("/ask", response_model=AskResponse)
def ask(q: Question):
    """Answer an Amharic question from the retrieved articles, with article citations."""
    articles = _articles(q)
    generate = get_generator()
    if generate is None:
        return AskResponse(articles=[_public(a) for a in articles], answer=None, cited_articles=[], refused=False)
    try:
        # configuration chosen by the RAG ablation: Amharic text only, lenient prompt (k=3 by default)
        answer = generate(build_messages(q.question, articles, include_english=False, lenient=True))
    except Exception as e:  # model errors should not crash the service
        raise HTTPException(status_code=503, detail=f"generation failed: {type(e).__name__}") from e
    valid = {a["article"] for a in articles}
    cited = [n for n in cited_articles(answer) if n in valid]  # drop citations of articles not provided
    return AskResponse(articles=[_public(a) for a in articles], answer=answer, cited_articles=cited,
                       refused=is_refusal(answer))
