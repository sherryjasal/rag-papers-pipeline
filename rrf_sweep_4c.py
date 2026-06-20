"""
RRF k-sweep: hybrid (BM25+dense+RRF) only, no reranker.

Sweeps RRF k in {10, 30, 60} across recursive × {minilm, bge-small, openai}.
Baseline NDCG@5 is loaded from results_4c.json (not recomputed).
Saves full results to rrf_sweep_4c.json.

Usage:
    python rrf_sweep_4c.py --save-results rrf_sweep_4c.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from pipeline import EMBEDDER_REGISTRY
from eval_metrics import score_single_query, score_test_set
from retrieval_4c import retrieve_hybrid

console = Console()
TEST_SET_PATH = Path(__file__).parent / "test_set.json"
BASELINE_RESULTS = Path(__file__).parent / "results_4c.json"
TOP_K = 5
RRF_K_VALUES = [10, 30, 60]
EMBEDDERS = ["minilm", "bge-small", "openai"]


def load_test_set() -> list[dict]:
    with open(TEST_SET_PATH) as f:
        return json.load(f)


def load_baseline_ndcg() -> dict[str, float]:
    """Read committed baseline NDCG@5 from results_4c.json — no recompute."""
    with open(BASELINE_RESULTS) as f:
        data = json.load(f)
    return {
        r["config"]: r["aggregate"]["ndcg@5"]
        for r in data["results"]
        if r["config"].startswith("baseline")
    }


def run_hybrid_sweep(
    queries: list[dict],
    emb_name: str,
    rrf_k: int,
    embedder,
) -> dict:
    per_query = []
    t0 = time.perf_counter()
    for q in queries:
        try:
            chunks = retrieve_hybrid(q["query"], embedder, top_k=TOP_K, n_dense=50, rrf_k=rrf_k)
        except Exception as e:
            console.print(f"  [red]ERROR[/red] hybrid/{emb_name} k={rrf_k}: {e}")
            chunks = []
        titles = [c.get("title", "") for c in chunks]
        texts = [c.get("context_text", "") for c in chunks]
        scores = score_single_query(
            titles, q["expected_papers"],
            texts, q["expected_keywords"],
            TOP_K,
        )
        per_query.append({"query": q["query"], "scores": scores})

    elapsed = time.perf_counter() - t0
    aggregate = score_test_set([pq["scores"] for pq in per_query])
    return {
        "config": f"hybrid  recursive/{emb_name} rrf_k={rrf_k}",
        "embedder": emb_name,
        "rrf_k": rrf_k,
        "aggregate": aggregate,
        "per_query": per_query,
        "elapsed": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RRF k-sweep for hybrid configs")
    parser.add_argument("--save-results", metavar="FILE", required=True,
                        help="Write results to this JSON file.")
    args = parser.parse_args()

    queries = load_test_set()
    baseline_ndcg = load_baseline_ndcg()

    console.print()
    console.rule("[bold cyan]RRF k-sweep — hybrid (BM25+dense), no reranker[/bold cyan]")
    console.print(f"  k values:  {RRF_K_VALUES}")
    console.print(f"  Embedders: {EMBEDDERS}")
    console.print(f"  Baseline NDCG@5 loaded from: {BASELINE_RESULTS.name}")
    console.print()

    emb_cache: dict[str, object] = {}
    def get_emb(name):
        if name not in emb_cache:
            emb_cache[name] = EMBEDDER_REGISTRY[name]()
        return emb_cache[name]

    all_results = []
    total_t0 = time.perf_counter()

    for emb_name in EMBEDDERS:
        for k in RRF_K_VALUES:
            label = f"hybrid recursive/{emb_name} rrf_k={k}"
            console.print(f"  Running [cyan]{label}[/cyan] ...", end=" ")
            result = run_hybrid_sweep(queries, emb_name, k, get_emb(emb_name))
            console.print(f"[green]done[/green] ({result['elapsed']}s)")
            all_results.append(result)

    total_elapsed = time.perf_counter() - total_t0
    console.print(f"\n  Total runtime: {total_elapsed:.1f}s\n")

    # Print comparison table
    table = Table(
        title=f"Hybrid RRF k-sweep vs Dense Baseline (NDCG@{TOP_K}, 15 queries)",
        show_lines=True,
    )
    table.add_column("Embedder", style="bold cyan")
    table.add_column("RRF k", justify="right")
    table.add_column("Hybrid NDCG@5", justify="right")
    table.add_column("Baseline NDCG@5", justify="right", style="dim")
    table.add_column("Gap", justify="right")

    baseline_key_map = {
        "minilm":    "baseline  recursive/minilm",
        "bge-small": "baseline  recursive/bge-small",
        "openai":    "baseline  recursive/openai",
    }

    for r in all_results:
        emb = r["embedder"]
        k = r["rrf_k"]
        hybrid_ndcg = r["aggregate"].get("ndcg@5", 0.0)
        baseline_ndcg_val = baseline_ndcg.get(baseline_key_map[emb], 0.0)
        gap = hybrid_ndcg - baseline_ndcg_val
        gap_str = f"[green]+{gap:.3f}[/green]" if gap > 0 else f"[red]{gap:.3f}[/red]"
        table.add_row(
            f"recursive/{emb}",
            str(k),
            f"{hybrid_ndcg:.3f}",
            f"{baseline_ndcg_val:.3f}",
            gap_str,
        )

    console.print(table)
    console.print()

    # Save
    payload = {
        "metadata": {
            "top_k": TOP_K,
            "rrf_k_values": RRF_K_VALUES,
            "embedders": EMBEDDERS,
            "n_queries": len(queries),
            "baseline_source": str(BASELINE_RESULTS),
        },
        "baseline_ndcg": baseline_ndcg,
        "results": all_results,
    }
    out_path = Path(args.save_results)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    console.print(f"[green]✓[/green] Results saved → [cyan]{out_path}[/cyan]")


if __name__ == "__main__":
    main()
