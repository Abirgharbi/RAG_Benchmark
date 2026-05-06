# STM32Cube RAG Chunking Benchmark

**A local, reproducible RAG pipeline** to evaluate chunking strategies for STM32Cube knowledge bases.

---

## Architecture

```
data_generator.py   →  synthetic STM32Cube JSON (or load your own)
        ↓
chunking.py         →  5 chunking strategies applied to every document
        ↓
rag_pipeline.py     →  Embed (TF-IDF+LSA) → Index (ChromaDB) → Retrieve
        ↓
run_benchmark.py    →  Eval loop → Metrics → Summary table
```

## Quick start

```bash
pip install chromadb scikit-learn tqdm rich
python run_benchmark.py --docs 120 --top-k 5
```

## Using your own STM32Cube JSON

Replace the synthetic data with your real JSON:

```bash
python run_benchmark.py --data /path/to/your/stm32cube_kb.json
```

Your JSON must be a list of objects with at least:
```json
[
  {
    "id": "doc_0001",
    "title": "STM32H7 UART — Initialization",
    "content": "Full text content here...",
    "paragraphs": ["Para 1...", "Para 2..."],   // optional but recommended
    "peripheral": "UART",
    "mcu_family": "STM32H7",
    "topic": "init"
  }
]
```

## Chunking strategies

| Strategy | Description | Avg words | Use when |
|----------|-------------|-----------|----------|
| `full_document` | 1 chunk = entire doc | ~250 | Short, focused docs |
| `paragraph` | Split on double newlines | ~47 | Well-structured prose |
| `sliding_window` | 300-word windows, 50-word overlap | ~190 | Uniform text blocks |
| `section` | Header-delimited sections | ~47 | Docs with ## headers |
| `parent_child` | Child indexed, parent returned | ~47 | LLM needs wider context |

## Metrics

- **P@k** — Precision at rank k: fraction of top-k results that are relevant
- **MRR** — Mean Reciprocal Rank: how early does the first relevant result appear?
- **Avg Sim** — Average cosine similarity of retrieved chunks to query
- **Avg Len** — Average word count of retrieved chunks (richness proxy)
- **Latency** — End-to-end retrieval time including embedding

## Embedding

Uses **TF-IDF + Latent Semantic Analysis (256 dimensions)** locally — no API key, 
no internet required. The model is fitted on the full corpus + eval questions jointly.

To use sentence-transformers instead (needs HuggingFace access):
```python
# In rag_pipeline.py, replace Embedder with:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
```

## Output files

After running:
- `results/metrics.json` — full per-question and aggregate metrics
- `results/summary_table.json` — flat table for report/presentation
- `results/chunk_stats.json` — chunk count and size statistics
- `results/raw_results.json` — full retrieval results per question
- `chroma_db/` — persisted ChromaDB vector stores (one per strategy)

## Key findings (synthetic data run)

| Strategy | P@1 | MRR | Recommendation |
|----------|-----|-----|---------------|
| paragraph | **1.000** | **1.000** | Default delivery strategy |
| section | **1.000** | **1.000** | Requires structured headers |
| parent_child | **1.000** | **1.000** | Best for generation context |
| sliding_window | **1.000** | 1.000 | Lowest memory, P@3+ degrades |
| full_document | 0.950 | 0.975 | Acceptable for short docs only |

## Disclaimer

This is a **complementary local experiment**, not a reproduction of ST internal platforms.
The synthetic data mimics STM32Cube JSON structure for experimental purposes.
Always validate findings against your actual production delivery pipeline.
