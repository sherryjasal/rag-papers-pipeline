"""
Retrieval evaluation harness.

Loads test_set.json, runs every strategy x embedder config through
pipeline.run_config(), scores with eval_metrics, prints comparison table.

Usage:
    python eval_harness.py                        # all configs, k=5
    python eval_harness.py --top-k 10             # change cutoff
    python eval_harness.py --strategy recursive    # single strategy
    python eval_harness.py --verbose               # per-query scores
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from pipeline import run_config, EMBEDDER_REGISTRY, PAPER_TITLES
from eval_metrics import score_single_query, score_test_set

console = Console()

TEST_SET_PATH = Path(__file__).parent / "test_set.json"
STRATEGIES = ["recursive", "hierarchical"]
DEFAULT_TOP_K = 5

# Reverse lookup: filename stem -> full paper title
FILENAME_TO_TITLE = {
    fname.replace(".pdf", ""): title
    for fname, title in PAPER_TITLES.items()
}


def load_test_set(path: Path = TEST_SET_PATH) -> list[dict]:
    """Load golden queries from JSON."""
    with open(path) as f:
        return json.load(f)


def extract_paper_titles(chunks: list[dict]) -> list[str]:
    """Return paper titles from chunk dicts (in retrieval order)."""
    return [chunk.get("title", "") for chunk in chunks]


def extract_chunk_texts(chunks: list[dict]) -> list[str]:
    """Pull text content from chunk dicts."""
    return [chunk.get("context_text", "") for chunk in chunks]


def evaluate_config(
    queries: list[dict],
    strategy: str,
    embedder_name: str,
    top_k: int,
    verbose: bool = False,
) -> dict:
    """Run retrieval + scoring for every query under one config."""
    config_label = f"{strategy} / {embedder_name}"
    per_query_scores = []
    per_query_details = []

    t0 = time.perf_counter()

    for q in queries:
        query_text = q["query"]
        expected_papers = q["expected_papers"]
        expected_keywords = q["expected_keywords"]

        try:
            chunks = run_config(
                query=query_text,
                strategy=strategy,
                embedder_name=embedder_name,
                top_k=top_k,
            )
        except Exception as e:
            console.print(f"  [red]ERROR[/red] {config_label}: {e}")
            chunks = []

        retrieved_papers = extract_paper_titles(chunks)
        chunk_texts = extract_chunk_texts(chunks)

        scores = score_single_query(
            retrieved_papers, expected_papers,
            chunk_texts, expected_keywords,
            top_k,
        )
        per_query_scores.append(scores)

        if verbose:
            per_query_details.append({
                "query": query_text,
                "expected": expected_papers,
                "retrieved": list(dict.fromkeys(retrieved_papers))[:5],
                "scores": scores,
            })

    elapsed = time.perf_counter() - t0
    aggregate = score_test_set(per_query_scores)

    if verbose:
        _print_per_query(config_label, per_query_details, top_k)

    return {
        "config": config_label,
        "aggregate": aggregate,
        "elapsed": round(elapsed, 2),
        "details": per_query_details,
    }


def _print_per_query(label: str, details: list[dict], k: int) -> None:
    """Print per-query breakdown for one config."""
    table = Table(title=f"Per-Query: {label}", show_lines=True)
    table.add_column("Query", max_width=45)
    table.add_column("Expected", style="green", max_width=25)
    table.add_column("Retrieved", style="yellow", max_width=25)
    table.add_column(f"Hit@{k}", justify="right", width=5)
    table.add_column("MRR", justify="right", width=5)
    table.add_column("KW", justify="right", width=5)

    for d in details:
        s = d["scores"]
        hit_val = s[f"hit_rate@{k}"]
        hit_str = f"[green]1[/green]" if hit_val == 1.0 else f"[red]0[/red]"

        table.add_row(
            d["query"][:45],
            "\n".join(d["expected"][:2]),
            "\n".join(d["retrieved"][:3]),
            hit_str,
            f"{s['mrr']:.2f}",
            f"{s['keyword_hit_rate']:.2f}",
        )
    console.print(table)
    console.print()


def print_comparison(all_results: list[dict], k: int) -> None:
    """Print final side-by-side comparison table."""
    table = Table(title=f"Retrieval Eval Summary (k={k})", show_lines=True)
    table.add_column("Config", style="bold cyan")
    table.add_column(f"P@{k}", justify="right")
    table.add_column(f"R@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"NDCG@{k}", justify="right")
    table.add_column(f"Hit@{k}", justify="right")
    table.add_column("KW Hit", justify="right")
    table.add_column("Time", justify="right", style="dim")

    for r in all_results:
        a = r["aggregate"]
        table.add_row(
            r["config"],
            f"{a.get(f'precision@{k}', 0):.3f}",
            f"{a.get(f'recall@{k}', 0):.3f}",
            f"{a.get('mrr', 0):.3f}",
            f"{a.get(f'ndcg@{k}', 0):.3f}",
            f"{a.get(f'hit_rate@{k}', 0):.3f}",
            f"{a.get('keyword_hit_rate', 0):.3f}",
            f"{r['elapsed']:.1f}s",
        )
    console.print()
    console.print(table)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation harness")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--strategy", choices=STRATEGIES, default=None)
    parser.add_argument("--embedder", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    queries = load_test_set()
    strategies = [args.strategy] if args.strategy else STRATEGIES
    embedders = [args.embedder] if args.embedder else list(EMBEDDER_REGISTRY.keys())

    console.print(f"\n[bold]Retrieval Evaluation[/bold]")
    console.print(f"  Queries:  {len(queries)}")
    console.print(f"  Configs:  {len(strategies)} strategies x {len(embedders)} embedders")
    console.print(f"  Top-k:    {args.top_k}\n")

    all_results = []
    for strat in strategies:
        for emb in embedders:
            label = f"{strat} / {emb}"
            console.print(f"  Running [cyan]{label}[/cyan] ...", end=" ")
            result = evaluate_config(queries, strat, emb, args.top_k, args.verbose)
            console.print(f"[green]done[/green] ({result['elapsed']}s)")
            all_results.append(result)

    print_comparison(all_results, args.top_k)


if __name__ == "__main__":
    main()
