# retrieval_bm25.py
from rank_bm25 import BM25Okapi

def build_bm25_index(docs):  # docs: List[str]
    tokenized = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def topk(bm25, tokenized_corpus, query_text, k=10):
    scores = bm25.get_scores(query_text.split())
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return idxs, [scores[i] for i in idxs]