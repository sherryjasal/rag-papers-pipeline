#!/usr/bin/env python3
"""RAG pipeline for research papers using ChromaDB, sentence-transformers, and Claude."""

import os
import re
import sys
import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

PAPERS_DIR = Path(__file__).parent / "papers"
CHROMA_DIR = Path(__file__).parent / ".chroma"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"  # user requested claude-sonnet-4-20250514; current alias is claude-sonnet-4-6
TOP_K = 5

CHARS_PER_TOKEN = 4
CHUNK_TOKENS = 512
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN  # 2048 chars ≈ 512 tokens
OVERLAP_CHARS = int(CHUNK_CHARS * 0.15)       # ~307 chars (15% overlap)
MIN_SENTENCE_LEN = 20


# ---------- Embedder abstraction ----------

class Embedder(ABC):
    """Common interface for all embedding backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in collection names and logs."""

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors (one per input text)."""


class SentenceTransformerEmbedder(Embedder):
    """Wraps any sentence-transformers model behind the Embedder interface."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "Cite which paper each claim comes from in [Paper Title] format. "
    "If the context doesn't contain the answer, say so."
)

PAPER_TITLES = {
    "attention-is-all-you-need": "Attention Is All You Need",
    "bert": "BERT: Pre-training of Deep Bidirectional Transformers",
    "gpt3": "GPT-3: Language Models are Few-Shot Learners",
    "rag": "Retrieval-Augmented Generation",
    "hyde": "HyDE: Hypothetical Document Embeddings",
    "lora": "LoRA: Low-Rank Adaptation of Large Language Models",
    "constitutional-ai": "Constitutional AI",
    "react": "ReAct: Reasoning and Acting",
    "chain-of-thought": "Chain-of-Thought Prompting",
    "deepseek-r1": "DeepSeek-R1",
}

console = Console()


# ---------- PDF extraction ----------

def extract_pages(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"text": text, "page_num": i + 1})
    doc.close()
    return pages


def paper_title(pdf_path: Path) -> str:
    return PAPER_TITLES.get(pdf_path.stem, pdf_path.stem)


# ---------- Recursive chunking ----------

def _split_at_separators(text: str, separators: list[str], max_size: int) -> list[str]:
    if not text.strip():
        return []
    if len(text) <= max_size or not separators:
        return [text]
    sep = separators[0]
    parts = text.split(sep) if sep else list(text)
    chunks, current = [], ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(part) > max_size:
                chunks.extend(_split_at_separators(part, separators[1:], max_size))
                current = ""
            else:
                current = part
    if current.strip():
        chunks.append(current)
    return chunks


def chunk_recursive(text: str) -> list[str]:
    pieces = _split_at_separators(text, ["\n\n", "\n", ". ", " "], CHUNK_CHARS)
    if not pieces:
        return []
    chunks, current_pieces, current_len = [], [], 0
    for piece in pieces:
        if current_len + len(piece) > CHUNK_CHARS and current_pieces:
            chunks.append(" ".join(current_pieces))
            overlap_pieces, overlap_len = [], 0
            for p in reversed(current_pieces):
                if overlap_len + len(p) > OVERLAP_CHARS:
                    break
                overlap_pieces.insert(0, p)
                overlap_len += len(p)
            current_pieces, current_len = overlap_pieces, overlap_len
        current_pieces.append(piece)
        current_len += len(piece)
    if current_pieces:
        chunks.append(" ".join(current_pieces))
    return [c for c in chunks if c.strip()]


# ---------- Hierarchical chunking ----------

def chunk_hierarchical(text: str) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    items = []
    for para in paragraphs:
        sentences = re.split(r"(?<=[.?!])\s+", para)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= MIN_SENTENCE_LEN:
                items.append({"sentence": sent, "parent": para})
    return items


# ---------- ChromaDB ----------

def get_chroma_collection(strategy: str, embedder_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=f"papers_{strategy}_{embedder_name}",
        metadata={"hnsw:space": "cosine"},
    )


# ---------- Ingestion ----------

def ingest(strategy: str, embedder: Embedder) -> None:
    collection = get_chroma_collection(strategy, embedder.name)
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        console.print("[red]No PDFs found in papers/[/red]")
        return
    total = 0
    for pdf_path in pdfs:
        title = paper_title(pdf_path)
        console.print(f"  [cyan]{title}[/cyan]...", end=" ")
        pages = extract_pages(pdf_path)
        docs, metas, ids = [], [], []
        chunk_idx = 0
        for page in pages:
            page_num, page_text = page["page_num"], page["text"]
            if strategy == "recursive":
                for chunk in chunk_recursive(page_text):
                    docs.append(chunk)
                    metas.append({
                        "title": title,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "strategy": strategy,
                    })
                    ids.append(f"{pdf_path.stem}_p{page_num}_c{chunk_idx}")
                    chunk_idx += 1
            else:
                for item in chunk_hierarchical(page_text):
                    docs.append(item["sentence"])
                    metas.append({
                        "title": title,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "strategy": strategy,
                        "parent_context": item["parent"][:1500],
                    })
                    ids.append(f"{pdf_path.stem}_p{page_num}_c{chunk_idx}")
                    chunk_idx += 1
        if docs:
            BATCH = 128
            for i in range(0, len(docs), BATCH):
                embeddings = embedder.encode(docs[i : i + BATCH])
                collection.upsert(
                    documents=docs[i : i + BATCH],
                    embeddings=embeddings,
                    metadatas=metas[i : i + BATCH],
                    ids=ids[i : i + BATCH],
                )
            total += len(docs)
            console.print(f"[yellow]{len(docs)}[/yellow] chunks")
        else:
            console.print("[dim]skipped (no text)[/dim]")
    console.print(
        f"\n[green]✓[/green] Ingested [yellow]{total}[/yellow] chunks "
        f"into [cyan]papers_{strategy}_{embedder.name}[/cyan]"
    )


# ---------- Retrieval ----------

def retrieve(query: str, strategy: str, embedder: Embedder, top_k: int = TOP_K) -> list[dict]:
    collection = get_chroma_collection(strategy, embedder.name)
    n = collection.count()
    if n == 0:
        console.print(
            f"[red]Collection papers_{strategy}_{embedder.name} is empty. Run with --ingest first.[/red]"
        )
        return []
    query_embedding = embedder.encode([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, n),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        context = meta.get("parent_context", doc) if strategy == "hierarchical" else doc
        chunks.append({
            "context_text": context,
            "matched_text": doc,
            "title": meta["title"],
            "page": meta["page"],
            "similarity": 1.0 - float(dist),
            "strategy": strategy,
        })
    return chunks


# ---------- Generation ----------

def generate(query: str, chunks: list[dict]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set. Add it to .env[/red]")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)
    context_parts = [
        f"[Source {i}: {c['title']}, page {c['page']}]\n{c['context_text']}"
        for i, c in enumerate(chunks, 1)
    ]
    context = "\n\n---\n\n".join(context_parts)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache stable system prompt across queries
            }
        ],
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return response.content[0].text


# ---------- Rich display ----------

def show_chunks(chunks: list[dict], strategy: str) -> None:
    console.print(f"\n[magenta]Retrieved chunks ({strategy}):[/magenta]")
    for i, c in enumerate(chunks, 1):
        score = c["similarity"]
        score_color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        console.print(
            f"  [dim]{i}.[/dim] [cyan]{c['title']}[/cyan] [dim]p.{c['page']}[/dim]  "
            f"score=[{score_color}]{score:.3f}[/{score_color}]"
        )
        if c["matched_text"] != c["context_text"]:
            preview = c["matched_text"][:120].replace("\n", " ")
            console.print(f"     [dim italic]matched:[/dim italic] {preview}…")
        preview = c["context_text"][:200].replace("\n", " ")
        console.print(f"     [dim]{preview}…[/dim]")
    console.print()


def show_answer(answer: str, strategy: str, compare: bool = False) -> None:
    styled = re.sub(r"\[([^\]]+)\]", r"[green][\1][/green]", answer)
    title = f"Answer  [dim]({strategy})[/dim]" if compare else "Answer"
    console.print(Panel(styled, title=title, border_style="blue", expand=False))


# ---------- Query runner ----------

def run_query(
    query: str,
    strategy: str,
    embedder: Embedder,
    verbose: bool,
    compare: bool = False,
    top_k: int = TOP_K,
) -> None:
    chunks = retrieve(query, strategy, embedder, top_k=top_k)
    if not chunks:
        return
    if verbose:
        show_chunks(chunks, strategy)
    with console.status(f"[dim]Asking {CLAUDE_MODEL}…[/dim]"):
        answer = generate(query, chunks)
    show_answer(answer, strategy, compare=compare)


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline for research papers")
    parser.add_argument(
        "--chunk-strategy",
        choices=["recursive", "hierarchical"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks and scores")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion before querying")
    parser.add_argument(
        "--compare", action="store_true", help="Run query against both strategies side by side"
    )
    parser.add_argument("--top-k", type=int, default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold cyan]RAG Papers Pipeline[/bold cyan]\n"
            f"[dim]Embed: {EMBED_MODEL_NAME}  |  LLM: {CLAUDE_MODEL}  |  "
            f"Strategy: {'both' if args.compare else args.chunk_strategy}[/dim]",
            border_style="cyan",
        )
    )

    with console.status("[dim]Loading embedding model…[/dim]"):
        embedder = SentenceTransformerEmbedder()
    console.print(f"[green]✓[/green] Loaded [cyan]{embedder.name}[/cyan]\n")

    if args.ingest:
        strategies = ["recursive", "hierarchical"] if args.compare else [args.chunk_strategy]
        for s in strategies:
            console.rule(f"[magenta]Ingesting ({s})[/magenta]")
            ingest(s, embedder)
        console.print()

    console.print("Type your question below. Enter [bold]quit[/bold] or [bold]exit[/bold] to stop.\n")

    while True:
        try:
            query = console.input("[bold green]>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break
        if args.compare:
            console.rule("[magenta]recursive[/magenta]")
            run_query(query, "recursive", embedder, args.verbose, compare=True, top_k=args.top_k)
            console.rule("[magenta]hierarchical[/magenta]")
            run_query(query, "hierarchical", embedder, args.verbose, compare=True, top_k=args.top_k)
        else:
            run_query(query, args.chunk_strategy, embedder, args.verbose, top_k=args.top_k)


if __name__ == "__main__":
    main()
