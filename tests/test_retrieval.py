from amlegal.retrieval import BM25, mrr, recall_at_k, reciprocal_rank_fusion

DOCS = [
    "ጋብቻ ለመፈጸም ወንዱም ሴቷም አሥራ ስምንት ዓመት የሞላቸው መሆን አለባቸው።",   # age of marriage
    "ፍቺ በፍርድ ቤት ውሳኔ ሊፈጸም ይችላል።",                                 # divorce
    "የልጆች ቀለብ የመስጠት ግዴታ በወላጆች ላይ ነው።",                          # maintenance
]


def test_bm25_finds_matching_article():
    bm = BM25(DOCS)
    assert bm.search("የጋብቻ እድሜ ስንት ዓመት ነው?", k=1) == [0]
    assert bm.search("ፍቺ እንዴት ይፈጸማል?", k=1) == [1]


def test_bm25_normalises_homophones():
    # ዓመት and አመት are the same word spelled with different letters
    bm = BM25(DOCS)
    assert bm.search("አመት", k=1) == [0]


def test_char_ngrams_match_affixed_words():
    # "ለልጆች" (for children) shares no whole token with "የልጆች", but shares character n-grams
    bm = BM25(DOCS, char_ngrams=3)
    assert bm.search("ለልጆች ቀለብ", k=1) == [2]


def test_rrf_and_metrics():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]], top=3)
    assert set(fused[:2]) == {1, 2}
    ranked = [[3, 1], [2, 5]]
    gold = [{1}, {9}]
    assert recall_at_k(ranked, gold, 1) == 0
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert mrr(ranked, gold) == 0.25


def test_hybrid_retriever_bm25_only():
    from amlegal.retrieval import HybridRetriever
    corpus = [{"article": i + 1, "title_am": "", "text_am": d, "title_en": "", "text_en": ""}
              for i, d in enumerate(DOCS)]
    r = HybridRetriever(corpus)
    assert r.search("ፍቺ እንዴት ይፈጸማል?", k=1)[0]["article"] == 2


def test_hybrid_retriever_with_encoder():
    import numpy as np
    from amlegal.retrieval import HybridRetriever
    corpus = [{"article": i + 1, "title_am": "", "text_am": d, "title_en": "", "text_en": ""}
              for i, d in enumerate(DOCS)]

    def encode(texts):  # toy encoder: one-hot on which keyword appears
        keys = ["ጋብቻ", "ፍቺ", "ቀለብ"]
        v = np.array([[1.0 if k in t else 0.0 for k in keys] for t in texts]) + 1e-6
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    r = HybridRetriever(corpus, encode=encode)
    assert r.search("ቀለብ", k=1)[0]["article"] == 3
