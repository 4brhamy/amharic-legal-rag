# Amharic legal QA API. Default image: BM25 retrieval only (small, CPU).
# Build with --build-arg EXTRAS="sentence-transformers" and run with -e AMLEGAL_EMBEDDINGS=1 for
# dense retrieval; add transformers/torch and AMLEGAL_LLM for answer generation (GPU recommended).
FROM python:3.11-slim

ARG EXTRAS=""
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt ${EXTRAS}

COPY amlegal ./amlegal
COPY data/family_code.jsonl ./data/family_code.jsonl

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "amlegal.api:app", "--host", "0.0.0.0", "--port", "8000"]
