"""
Issue 4c retrieval modes.

Three new retrieval functions alongside the existing pipeline.retrieve():
  - retrieve_reranked: two-stage dense → cross-encoder
  - retrieve_hybrid: BM25 + dense with RRF (recursive only)
  - retrieve_hybrid_reranked: hybrid → cross-encoder
  - retrieve_hierarchical_parent_scored: hierarchical with parent-paragraph dedup

All return the same chunk dict format as pipeline.retrieve() so the eval
harness can use the same extract_paper_titles / extract_chunk_texts helpers.

Named constants logged in every result row:
  RRF_K = 60          — RRF fusion constant
  RERANKER_MODEL      — cross-encoder model name
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from pipeline import get_chroma_collection, Embedder, TOP_K

RRF_K = 60
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------- Base: dense retrieve with stored embeddings ----------

def _dense_retrieve(
    query: str,
    strategy: str,
    embedder: Embedder,
    n: int,
) -> list[dict]:
    """Dense retrieval that also returns stored ChromaDB embeddings per chunk."""
    collection = get_chroma_collection(strategy, embedder.name)
    count = collection.count()
    if count == 0:
        return []
    query_emb = embedder.encode([query])
    results = collection.query(
        query_embeddings=query_emb,
        n_results=min(n, count),
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    chunks = []
    for doc, meta, dist, emb in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["embeddings"][0],
    ):
        context = meta.get("parent_context", doc) if strategy == "hierarchical" else doc
        chunks.append({
            "context_text": context,
            "matched_text": doc,
            "title": meta["title"],
            "page": meta["page"],
            "similarity": 1.0 - float(dist),
            "strategy": strategy,
            "embedding": list(emb),
        })
    return chunks


# ---------- Cross-encoder reranker ----------

class CrossEncoderReranker:
    """Two-stage reranker using cross-encoder/ms-marco-MiniLM-L-6-v2.

    Load once and reuse across all configs in a harness run.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Score each chunk jointly with the query, return top_k by CE score."""
        if not chunks:
            return []
        pairs = [(query, c["matched_text"]) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = []
        for score, chunk in ranked[:top_k]:
            c = dict(chunk)
            c["reranker_score"] = float(score)
            result.append(c)
        return result


def retrieve_reranked(
    query: str,
    strategy: str,
    embedder: Embedder,
    reranker: CrossEncoderReranker,
    top_k: int = TOP_K,
    n_retrieve: int = 20,
) -> list[dict]:
    """Two-stage: dense top-N → cross-encoder top-k.

    n_retrieve controls how many candidates the cross-encoder sees.
    """
    candidates = _dense_retrieve(query, strategy, embedder, n=n_retrieve)
    return reranker.rerank(query, candidates, top_k=top_k)


# ---------- BM25 + dense hybrid with RRF ----------

def _build_bm25_corpus(
    strategy: str,
    embedder_name: str,
) -> tuple[BM25Okapi, list[dict], dict[str, int]]:
    """Fetch all docs from ChromaDB and build a BM25 index over them.

    Returns (bm25_index, corpus_records, text_to_corpus_idx).
    Embeddings are included in corpus_records for downstream metrics.
    """
    collection = get_chroma_collection(strategy, embedder_name)
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    raw_docs = result["documents"]
    metas = result["metadatas"]
    embeddings = result["embeddings"]

    corpus = []
    for doc, meta, emb in zip(raw_docs, metas, embeddings):
        context = meta.get("parent_context", doc) if strategy == "hierarchical" else doc
        corpus.append({
            "context_text": context,
            "matched_text": doc,
            "title": meta["title"],
            "page": meta["page"],
            "strategy": strategy,
            "embedding": list(emb),
        })

    tokenized = [d["matched_text"].lower().split() for d in corpus]
    bm25 = BM25Okapi(tokenized)
    text_to_idx = {d["matched_text"]: i for i, d in enumerate(corpus)}
    return bm25, corpus, text_to_idx


def _rrf_fuse(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion over lists of corpus indices.

    score(d) = Σ 1 / (k + rank(d))  summed across all ranked lists.
    Returns indices sorted by descending fused score.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def retrieve_hybrid(
    query: str,
    embedder: Embedder,
    top_k: int = TOP_K,
    n_dense: int = 50,
) -> list[dict]:
    """BM25 + dense retrieval fused with RRF. Recursive strategy only.

    Dense retrieves top n_dense by cosine similarity.
    BM25 ranks the full corpus.
    RRF_K=60 fuses the two ranked lists.
    """
    strategy = "recursive"
    bm25, corpus, text_to_idx = _build_bm25_corpus(strategy, embedder.name)

    dense_chunks = _dense_retrieve(query, strategy, embedder, n=n_dense)
    dense_ranked = [
        text_to_idx[c["matched_text"]]
        for c in dense_chunks
        if c["matched_text"] in text_to_idx
    ]

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(range(len(corpus)), key=lambda i: bm25_scores[i], reverse=True)

    fused = _rrf_fuse([dense_ranked, bm25_ranked], k=RRF_K)

    dense_score_map = {c["matched_text"]: c["similarity"] for c in dense_chunks}
    result = []
    for idx in fused[:top_k]:
        chunk = dict(corpus[idx])
        chunk["similarity"] = dense_score_map.get(chunk["matched_text"], 0.0)
        result.append(chunk)
    return result


def retrieve_hybrid_reranked(
    query: str,
    embedder: Embedder,
    reranker: CrossEncoderReranker,
    top_k: int = TOP_K,
    n_hybrid: int = 50,
) -> list[dict]:
    """Hybrid (BM25+dense+RRF) retrieval → cross-encoder reranking."""
    candidates = retrieve_hybrid(query, embedder, top_k=n_hybrid, n_dense=n_hybrid)
    return reranker.rerank(query, candidates, top_k=top_k)


# ---------- Parent-document scoring for hierarchical ----------

def retrieve_hierarchical_parent_scored(
    query: str,
    embedder: Embedder,
    top_k: int = TOP_K,
    n_retrieve: int = 50,
) -> list[dict]:
    """Hierarchical retrieval with parent-paragraph deduplication.

    The 4b harness retrieved top-N leaf sentences and scored them flat, which
    allowed multiple sentences from one paragraph to occupy all top-5 slots.
    This function collapses leaf chunks by their parent paragraph (context_text),
    keeping the highest-similarity representative, then returns top-k unique parents.

    Ground truth remains paper-level; each result still carries its paper title.
    """
    strategy = "hierarchical"
    candidates = _dense_retrieve(query, strategy, embedder, n=n_retrieve)

    best_by_parent: dict[str, dict] = {}
    for chunk in candidates:
        key = chunk["context_text"]
        if key not in best_by_parent or chunk["similarity"] > best_by_parent[key]["similarity"]:
            best_by_parent[key] = chunk

    deduped = sorted(best_by_parent.values(), key=lambda c: c["similarity"], reverse=True)
    return deduped[:top_k]
