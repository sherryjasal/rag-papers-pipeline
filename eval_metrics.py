"""
Retrieval evaluation metrics.

All functions are pure: they take lists of retrieved/relevant items
and return scores. No pipeline imports, no side effects.

Two types of scoring:
1. Paper-level: did the right papers appear in top-k?
2. Keyword-level: do retrieved chunks contain expected terms?

Built for AI Snippets Issue 4b — Retrieval Layer Evaluation.
"""

from __future__ import annotations
import math


# ---------- Paper-Level Metrics ----------

def precision_at_k(retrieved_papers: list[str], relevant_papers: list[str], k: int) -> float:
    """Fraction of top-k retrieved papers that are relevant.

    Precision@k = |relevant ∩ retrieved[:k]| / k
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_papers[:k]
    relevant = set(relevant_papers)
    hits = sum(1 for p in top_k if p in relevant)
    return hits / k


def recall_at_k(retrieved_papers: list[str], relevant_papers: list[str], k: int) -> float:
    """Fraction of relevant papers that appear in top-k.

    Recall@k = |relevant ∩ retrieved[:k]| / |relevant|
    Returns 1.0 when relevant_papers is empty (nothing to miss).
    """
    if not relevant_papers:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = set(retrieved_papers[:k])
    relevant = set(relevant_papers)
    return len(relevant & top_k) / len(relevant)


def mrr(retrieved_papers: list[str], relevant_papers: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant result.

    Returns 0.0 if no relevant paper is found.
    """
    relevant = set(relevant_papers)
    for rank, paper in enumerate(retrieved_papers, start=1):
        if paper in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_papers: list[str], relevant_papers: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Binary relevance: 1 if paper is relevant, 0 otherwise.
    Only the FIRST chunk from each relevant paper scores a hit.
    This prevents duplicate chunks from inflating DCG beyond IDCG.
    NDCG@k = DCG@k / IDCG@k, always in [0, 1].
    """
    if k <= 0:
        return 0.0
    if not relevant_papers:
        return 1.0

    relevant = set(relevant_papers)
    seen_relevant = set()

    # DCG: only first occurrence of each relevant paper scores
    dcg = 0.0
    for i, paper in enumerate(retrieved_papers[:k]):
        if paper in relevant and paper not in seen_relevant:
            dcg += 1.0 / math.log2(i + 2)
            seen_relevant.add(paper)

    # IDCG: best case — all relevant papers ranked at top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate(retrieved_papers: list[str], relevant_papers: list[str], k: int) -> float:
    """Binary: 1.0 if ANY relevant paper appears in top-k, else 0.0.

    Returns 1.0 when relevant_papers is empty.
    """
    if not relevant_papers:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = set(retrieved_papers[:k])
    relevant = set(relevant_papers)
    return 1.0 if (relevant & top_k) else 0.0


# ---------- Keyword-Level Metrics ----------

def keyword_hit_rate(chunk_texts: list[str], expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found anywhere in retrieved chunk texts.

    Case-insensitive substring match.
    Returns 1.0 when expected_keywords is empty.
    """
    if not expected_keywords:
        return 1.0
    combined_text = " ".join(chunk_texts).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return hits / len(expected_keywords)


# ---------- Aggregate Helpers ----------

def score_single_query(
    retrieved_papers: list[str],
    relevant_papers: list[str],
    chunk_texts: list[str],
    expected_keywords: list[str],
    k: int,
) -> dict[str, float]:
    """Compute all metrics for one query. Returns flat dict."""
    return {
        f"precision@{k}": precision_at_k(retrieved_papers, relevant_papers, k),
        f"recall@{k}": recall_at_k(retrieved_papers, relevant_papers, k),
        "mrr": mrr(retrieved_papers, relevant_papers),
        f"ndcg@{k}": ndcg_at_k(retrieved_papers, relevant_papers, k),
        f"hit_rate@{k}": hit_rate(retrieved_papers, relevant_papers, k),
        "keyword_hit_rate": keyword_hit_rate(chunk_texts, expected_keywords),
    }


def score_test_set(results: list[dict[str, float]]) -> dict[str, float]:
    """Average all metrics across a list of per-query score dicts."""
    if not results:
        return {}
    keys = results[0].keys()
    return {
        key: sum(r[key] for r in results) / len(results)
        for key in keys
    }
