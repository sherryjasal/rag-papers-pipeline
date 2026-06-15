"""
Issue 4c retrieval experiment harness.

Runs the full experiment matrix and outputs a comparison table.
Reuses eval_metrics.py and the pipeline.run_config() interface for baselines;
new retrieval modes are in retrieval_4c.py.

Experiment matrix:
  Baselines          recursive × {minilm, bge-small, openai}
  +reranking         recursive × minilm N∈{10,20,50}; bge-small N=20; openai N=20
  +hybrid            recursive × {minilm, bge-small, openai}  [BM25+dense+RRF]
  +both              recursive × {minilm, openai}              [hybrid → reranker]
  Corrected hier.    hierarchical × {minilm, openai}           [parent-paragraph dedup]

Every result row logs N, RRF_K, and reranker model name so each row is
traceable to its exact config.

Usage:
    python eval_harness_4c.py
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table
from rich.rule import Rule

from pipeline import EMBEDDER_REGISTRY, run_config
from eval_metrics import (
    score_single_query,
    score_test_set,
    unique_source_count_at_k,
    mean_pairwise_similarity_at_k,
)
from retrieval_4c import (
    CrossEncoderReranker,
    RERANKER_MODEL,
    RRF_K,
    retrieve_reranked,
    retrieve_hybrid,
    retrieve_hybrid_reranked,
    retrieve_hierarchical_parent_scored,
)

console = Console()
TEST_SET_PATH = Path(__file__).parent / "test_set.json"
TOP_K = 5

# 4-category grouping (0-indexed query positions)
CATEGORIES: dict[str, list[int]] = {
    "architecture": [0, 1, 2, 10],    # attention, BERT, GPT-3, encoder/decoder
    "retrieval":    [3, 4, 13],        # RAG, HyDE, external knowledge
    "efficiency":   [5, 12],           # LoRA, fine-tuning cost
    "reasoning":    [6, 7, 8, 9, 11, 14],  # ConstitutionalAI, ReAct, CoT, DeepSeek, alignment, RL
}

# Queries that were hard in 4b (hit_rate < 1 or mrr < 0.5 in at least one config)
FAILURE_QUERY_INDICES = {11, 12, 13, 14}


@dataclass
class RunConfig:
    label: str
    retrieve: Callable[[str], list[dict]]
    meta: dict = field(default_factory=dict)


def load_test_set() -> list[dict]:
    with open(TEST_SET_PATH) as f:
        return json.load(f)


def _extract_titles(chunks: list[dict]) -> list[str]:
    return [c.get("title", "") for c in chunks]

def _extract_texts(chunks: list[dict]) -> list[str]:
    return [c.get("context_text", "") for c in chunks]

def _extract_embeddings(chunks: list[dict]) -> list[list[float]]:
    return [c["embedding"] for c in chunks if "embedding" in c]


def evaluate_config(config: RunConfig, queries: list[dict]) -> dict:
    per_query = []
    errors = []
    t0 = time.perf_counter()

    for i, q in enumerate(queries):
        try:
            chunks = config.retrieve(q["query"])
        except Exception as e:
            errors.append(f"Q{i}: {e}")
            chunks = []

        titles = _extract_titles(chunks)
        texts = _extract_texts(chunks)
        embs = _extract_embeddings(chunks)

        scores = score_single_query(
            titles, q["expected_papers"],
            texts, q["expected_keywords"],
            TOP_K,
        )
        scores["unique_src@5"] = unique_source_count_at_k(titles, TOP_K)
        scores["mps@5"] = mean_pairwise_similarity_at_k(embs, TOP_K) if len(embs) >= 2 else 0.0

        per_query.append({
            "idx": i,
            "query": q["query"],
            "expected": q["expected_papers"],
            "retrieved": titles[:TOP_K],
            "scores": scores,
        })

    elapsed = time.perf_counter() - t0
    aggregate = score_test_set([pq["scores"] for pq in per_query])
    return {
        "config": config.label,
        "meta": config.meta,
        "aggregate": aggregate,
        "per_query": per_query,
        "elapsed": round(elapsed, 2),
        "errors": errors,
    }


def build_configs(reranker: CrossEncoderReranker) -> list[RunConfig]:
    """Construct all RunConfig objects for the experiment matrix."""
    configs = []
    emb_cache: dict[str, object] = {}

    def get_emb(name: str):
        if name not in emb_cache:
            emb_cache[name] = EMBEDDER_REGISTRY[name]()
        return emb_cache[name]

    # ── Baselines (re-run for reference) ──────────────────────────────────
    for emb_name in ["minilm", "bge-small", "openai"]:
        configs.append(RunConfig(
            label=f"baseline  recursive/{emb_name}",
            retrieve=lambda q, e=emb_name: run_config(q, "recursive", e, TOP_K),
            meta={"mode": "baseline", "strategy": "recursive", "embedder": emb_name},
        ))

    # ── +reranking: sweep N for minilm, N=20 for others ───────────────────
    for n in [10, 20, 50]:
        configs.append(RunConfig(
            label=f"rerank    recursive/minilm N={n} model={RERANKER_MODEL}",
            retrieve=lambda q, n=n: retrieve_reranked(
                q, "recursive", get_emb("minilm"), reranker, top_k=TOP_K, n_retrieve=n,
            ),
            meta={"mode": "rerank", "embedder": "minilm", "N": n, "reranker": RERANKER_MODEL},
        ))
    for emb_name in ["bge-small", "openai"]:
        configs.append(RunConfig(
            label=f"rerank    recursive/{emb_name} N=20 model={RERANKER_MODEL}",
            retrieve=lambda q, e=emb_name: retrieve_reranked(
                q, "recursive", get_emb(e), reranker, top_k=TOP_K, n_retrieve=20,
            ),
            meta={"mode": "rerank", "embedder": emb_name, "N": 20, "reranker": RERANKER_MODEL},
        ))

    # ── +hybrid: BM25+dense+RRF, recursive only ───────────────────────────
    for emb_name in ["minilm", "bge-small", "openai"]:
        configs.append(RunConfig(
            label=f"hybrid    recursive/{emb_name} RRF_K={RRF_K}",
            retrieve=lambda q, e=emb_name: retrieve_hybrid(
                q, get_emb(e), top_k=TOP_K, n_dense=50,
            ),
            meta={"mode": "hybrid", "embedder": emb_name, "RRF_K": RRF_K},
        ))

    # ── +both: hybrid → reranker ───────────────────────────────────────────
    for emb_name in ["minilm", "openai"]:
        configs.append(RunConfig(
            label=f"hybrid+rr recursive/{emb_name} RRF_K={RRF_K} N=50 model={RERANKER_MODEL}",
            retrieve=lambda q, e=emb_name: retrieve_hybrid_reranked(
                q, get_emb(e), reranker, top_k=TOP_K, n_hybrid=50,
            ),
            meta={"mode": "hybrid+rerank", "embedder": emb_name, "RRF_K": RRF_K,
                  "N": 50, "reranker": RERANKER_MODEL},
        ))

    # ── Corrected hierarchical (parent-paragraph dedup) ───────────────────
    for emb_name in ["minilm", "openai"]:
        configs.append(RunConfig(
            label=f"hier_par  hierarchical/{emb_name} N=50 (parent-scored)",
            retrieve=lambda q, e=emb_name: retrieve_hierarchical_parent_scored(
                q, get_emb(e), top_k=TOP_K, n_retrieve=50,
            ),
            meta={"mode": "hierarchical_parent", "embedder": emb_name, "N": 50},
        ))

    return configs


def print_summary(all_results: list[dict]) -> None:
    table = Table(title=f"4c Retrieval Experiment Summary (k={TOP_K}, 15 queries)", show_lines=True)
    table.add_column("Config", style="bold cyan", min_width=20)
    table.add_column(f"P@{TOP_K}", justify="right")
    table.add_column(f"R@{TOP_K}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"NDCG@{TOP_K}", justify="right")
    table.add_column(f"Hit@{TOP_K}", justify="right")
    table.add_column("KW Hit", justify="right")
    table.add_column("UniSrc@5", justify="right")
    table.add_column("MPS@5", justify="right")
    table.add_column("Time", justify="right", style="dim")

    for r in all_results:
        a = r["aggregate"]
        err_flag = " [red]ERR[/red]" if r["errors"] else ""
        table.add_row(
            r["config"] + err_flag,
            f"{a.get(f'precision@{TOP_K}', 0):.3f}",
            f"{a.get(f'recall@{TOP_K}', 0):.3f}",
            f"{a.get('mrr', 0):.3f}",
            f"{a.get(f'ndcg@{TOP_K}', 0):.3f}",
            f"{a.get(f'hit_rate@{TOP_K}', 0):.3f}",
            f"{a.get('keyword_hit_rate', 0):.3f}",
            f"{a.get('unique_src@5', 0):.2f}",
            f"{a.get('mps@5', 0):.3f}",
            f"{r['elapsed']:.1f}s",
        )
    console.print()
    console.print(table)
    console.print()


def print_category_breakdown(all_results: list[dict]) -> None:
    """Per-category NDCG@5 breakdown."""
    console.rule("[bold]Per-Category NDCG@5 Breakdown[/bold]")
    table = Table(show_lines=True)
    table.add_column("Config", style="cyan", min_width=20)
    for cat in CATEGORIES:
        table.add_column(cat.capitalize(), justify="right")

    for r in all_results:
        pq = r["per_query"]
        row = [r["config"]]
        for cat, indices in CATEGORIES.items():
            ndcg_vals = [pq[i]["scores"].get(f"ndcg@{TOP_K}", 0.0) for i in indices if i < len(pq)]
            avg = sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else 0.0
            row.append(f"{avg:.3f}")
        table.add_row(*row)
    console.print(table)
    console.print()


def print_failure_query_breakdown(all_results: list[dict], queries: list[dict]) -> None:
    """Per-query detail for the 4b failure queries."""
    console.rule("[bold]4b Failure Queries — Per-Config Detail[/bold]")
    for qi in sorted(FAILURE_QUERY_INDICES):
        q = queries[qi]
        console.print(f"\n[yellow]Q{qi}[/yellow]: {q['query']}")
        console.print(f"  Expected: {', '.join(q['expected_papers'])}\n")

        table = Table(show_lines=True, show_header=True)
        table.add_column("Config", style="cyan", min_width=20)
        table.add_column("Retrieved", max_width=40)
        table.add_column(f"Hit@{TOP_K}", justify="right", width=6)
        table.add_column("MRR", justify="right", width=5)
        table.add_column(f"NDCG@{TOP_K}", justify="right", width=7)
        table.add_column("UniSrc", justify="right", width=6)

        for r in all_results:
            pq = r["per_query"]
            if qi >= len(pq):
                continue
            entry = pq[qi]
            s = entry["scores"]
            hit = s.get(f"hit_rate@{TOP_K}", 0)
            hit_str = f"[green]1[/green]" if hit == 1.0 else f"[red]0[/red]"
            retrieved_str = "\n".join(set(entry["retrieved"][:5]))
            table.add_row(
                r["config"],
                retrieved_str,
                hit_str,
                f"{s.get('mrr', 0):.2f}",
                f"{s.get(f'ndcg@{TOP_K}', 0):.3f}",
                f"{s.get('unique_src@5', 0):.0f}",
            )
        console.print(table)


def main() -> None:
    queries = load_test_set()

    console.print()
    console.rule("[bold cyan]Issue 4c — Retrieval Architecture Experiment[/bold cyan]")
    console.print(f"  Queries:       {len(queries)}")
    console.print(f"  Top-k:         {TOP_K}")
    console.print(f"  RRF_K:         {RRF_K}")
    console.print(f"  Reranker:      {RERANKER_MODEL}")
    console.print()

    console.print("  Loading cross-encoder reranker...", end=" ")
    t_load = time.perf_counter()
    reranker = CrossEncoderReranker(RERANKER_MODEL)
    console.print(f"[green]done[/green] ({time.perf_counter() - t_load:.1f}s)")
    console.print()

    configs = build_configs(reranker)
    console.print(f"  Configs to run: {len(configs)}\n")

    all_results = []
    total_t0 = time.perf_counter()

    for cfg in configs:
        console.print(f"  Running [cyan]{cfg.label}[/cyan] ...", end=" ")
        result = evaluate_config(cfg, queries)
        status = f"[green]done[/green] ({result['elapsed']}s)"
        if result["errors"]:
            status += f" [red]{len(result['errors'])} error(s)[/red]"
        console.print(status)
        if result["errors"]:
            for err in result["errors"]:
                console.print(f"    [red]↳ {err}[/red]")
        all_results.append(result)

    total_elapsed = time.perf_counter() - total_t0
    console.print(f"\n  Total runtime: {total_elapsed:.1f}s\n")

    print_summary(all_results)
    print_category_breakdown(all_results)
    print_failure_query_breakdown(all_results, queries)

    # Report any errored configs
    errored = [r for r in all_results if r["errors"]]
    if errored:
        console.rule("[bold red]Errored Configs[/bold red]")
        for r in errored:
            console.print(f"  [red]{r['config']}[/red]")
            for err in r["errors"]:
                console.print(f"    {err}")
    else:
        console.print("[green]All configs completed without errors.[/green]")


if __name__ == "__main__":
    main()
