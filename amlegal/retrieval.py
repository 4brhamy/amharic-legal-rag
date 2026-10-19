"""Retrieval over law articles: BM25 keyword search, dense embeddings, and rank fusion."""
import math
from collections import Counter

from .ethiopic import STOPWORDS, normalize, tokenize


class BM25:
    """Okapi BM25 over pre-tokenised documents (Amharic tokens are normalised by `tokenize`)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75, char_ngrams: int = 0):
        self.k1, self.b, self.char_ngrams = k1, b, char_ngrams
        self.stopwords = {normalize(w) for w in STOPWORDS}
        self.docs = [self._terms(d) for d in docs]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        df = Counter(t for d in self.tf for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _terms(self, text: str) -> list[str]:
        toks = [t for t in tokenize(text) if t not in self.stopwords]
        if not self.char_ngrams:
            return toks
        # Character n-grams make matching robust to Amharic prefixes/suffixes (በ-, የ-, -ን, -ዎች)
        # and to single-letter OCR errors.
        n = self.char_ngrams
        grams = []
        for t in toks:
            padded = f"#{t}#"
            grams += [padded[i:i + n] for i in range(max(1, len(padded) - n + 1))]
        return grams

    def scores(self, query: str) -> list[float]:
        q = self._terms(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def search(self, query: str, k: int = 5) -> list[int]:
        s = self.scores(query)
        return sorted(range(len(s)), key=lambda i: -s[i])[:k]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60, top: int = 5) -> list[int]:
    """Merge several ranked lists of document ids (Cormack et al., 2009)."""
    score = Counter()
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            score[doc] += 1.0 / (k + rank + 1)
    return [doc for doc, _ in score.most_common(top)]


def recall_at_k(ranked: list[list[int]], gold: list[set[int]], k: int) -> float:
    return sum(bool(set(r[:k]) & g) for r, g in zip(ranked, gold)) / max(1, len(gold))


def mrr(ranked: list[list[int]], gold: list[set[int]]) -> float:
    total = 0.0
    for r, g in zip(ranked, gold):
        for i, doc in enumerate(r):
            if doc in g:
                total += 1 / (i + 1)
                break
    return total / max(1, len(gold))


class HybridRetriever:
    """The best method from the retrieval evaluation: rank fusion of BM25 (words and character
    3-grams) and dense BGE-M3 embeddings over both the Amharic and the English text.

    `encode` is any function mapping a list of strings to L2-normalised vectors (numpy array). It is
    passed in so this module has no heavy dependency; with encode=None only BM25 is used.
    """

    def __init__(self, corpus: list[dict], encode=None, depth: int = 20):
        import numpy as np

        self.np = np
        self.corpus = corpus
        self.depth = depth
        am = [f"{r['title_am']}\n{r['text_am']}" for r in corpus]
        self.bm_word = BM25(am)
        self.bm_char = BM25(am, char_ngrams=3)
        self.encode = encode
        if encode is not None:
            self.E_am = encode(am)
            self.E_en = encode([f"{r['title_en']}\n{r['text_en']}" for r in corpus])

    def search(self, question: str, k: int = 5) -> list[dict]:
        rankings = [self.bm_word.search(question, self.depth), self.bm_char.search(question, self.depth)]
        if self.encode is not None:
            q = self.encode([question])[0]
            for E in (self.E_am, self.E_en):
                rankings.append([int(i) for i in self.np.argsort(-(E @ q))[: self.depth]])
        return [self.corpus[i] for i in reciprocal_rank_fusion(rankings, top=k)]
