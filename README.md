# RAG Papers Pipeline

> Companion code for **AI Snippets Issue 4a — The Retrieval Layer: Chunking Strategies**.  
> [Read the full breakdown →](https://thesherrycode.substack.com/p/i-tested-two-chunking-strategies)

A local RAG (Retrieval-Augmented Generation) pipeline over 10 ML research papers. Uses ChromaDB for vector storage, `sentence-transformers` for local embeddings, and Claude for answer generation.

## Papers included

10 ML research papers sourced from arXiv. Use under their respective licenses.

| File | Title |
|---|---|
| attention-is-all-you-need.pdf | Attention Is All You Need |
| bert.pdf | BERT: Pre-training of Deep Bidirectional Transformers |
| gpt3.pdf | GPT-3: Language Models are Few-Shot Learners |
| rag.pdf | Retrieval-Augmented Generation |
| hyde.pdf | HyDE: Hypothetical Document Embeddings |
| lora.pdf | LoRA: Low-Rank Adaptation of Large Language Models |
| constitutional-ai.pdf | Constitutional AI |
| react.pdf | ReAct: Reasoning and Acting |
| chain-of-thought.pdf | Chain-of-Thought Prompting |
| deepseek-r1.pdf | DeepSeek-R1 |

## Prerequisites

- Python 3.10+
- An Anthropic API key

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# Edit .env and replace the placeholder with your actual key
```

## Use your own corpus

Drop your own PDFs into `papers/` and re-run `--ingest`. The pipeline will chunk, embed, and index whatever's in the folder.

## Usage

### First run — ingest the papers

Before querying, embed and index all PDFs into ChromaDB:

```bash
# Ingest with the default recursive strategy
python pipeline.py --ingest

# Ingest with hierarchical chunking
python pipeline.py --ingest --chunk-strategy hierarchical

# Ingest both strategies at once (needed for --compare)
python pipeline.py --ingest --compare
```

Ingestion is skipped on subsequent runs unless you pass `--ingest` again.

### Query

```bash
# Basic query (recursive chunking, default)
python pipeline.py

# Choose chunking strategy
python pipeline.py --chunk-strategy hierarchical

# Show retrieved chunks and similarity scores
python pipeline.py --verbose

# Compare both strategies side by side for every query
python pipeline.py --compare

# Combine flags
python pipeline.py --chunk-strategy hierarchical --verbose
python pipeline.py --compare --verbose
```

### Example session

```
> What is the attention mechanism in transformers?
> How does LoRA compare to full fine-tuning?
> Why does chain-of-thought prompting improve reasoning?
> quit
```

## Chunking strategies

| Strategy | Chunk unit | Size | Overlap | Context sent to LLM |
|---|---|---|---|---|
| `recursive` | Character-bounded | ~512 tokens | 15% | The chunk itself |
| `hierarchical` | Sentence | Sentence-level | None | Parent paragraph |

**Recursive** splits at `\n\n` → `\n` → sentences → chars, merging pieces up to ~2048 characters with 15% overlap. Good for dense technical text.

**Hierarchical** indexes individual sentences for precise retrieval but sends the full parent paragraph to Claude for richer context. Good when answers span multiple sentences in a paragraph.

## Project layout

```
rag-papers-pipeline/
├── papers/          # PDF source files
├── .chroma/         # ChromaDB persistent storage (auto-created)
├── pipeline.py      # Main script
├── requirements.txt
├── .env             # API key (not committed)
└── .env.example
```

## CLI flags

| Flag | Description |
|---|---|
| `--chunk-strategy` | `recursive` (default) or `hierarchical` |
| `--ingest` | Re-run ingestion before querying |
| `--verbose` | Print retrieved chunks with similarity scores |
| `--compare` | Run each query against both strategies and show both answers |

## License

MIT — see [LICENSE](LICENSE) for details.
