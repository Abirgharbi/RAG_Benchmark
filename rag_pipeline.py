"""
RAG Pipeline: Embedding → ChromaDB Vector Store → Retrieval
Uses sentence-transformers for local embeddings (no API key needed).
"""

import os
import time
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

from chunking import Chunk

# ── Config ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "tfidf-lsa-256"   # local TF-IDF + LSA, no downloads needed
CHROMA_BASE_DIR = "chroma_db"


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    strategy: str
    text: str
    distance: float
    metadata: dict
    rank: int
    # For parent-child: the full parent text returned for context
    context_text: Optional[str] = None


# ── Embedder (TF-IDF + LSA — fully local, no downloads) ────────────────────

class Embedder:
    """
    Local embedding using TF-IDF + Latent Semantic Analysis (TruncatedSVD).
    No internet required. Fitted lazily on first corpus seen.
    Produces L2-normalized dense vectors suitable for cosine similarity.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, n_components: int = 256):
        self.model_name  = model_name
        self.n_components = n_components
        self.tfidf   = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=20_000,
            sublinear_tf=True,
        )
        self.svd     = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False
        print(f"📦 Embedder ready: TF-IDF + LSA ({n_components}-dim) — fully local")

    def fit(self, texts: List[str]):
        print(f"   🔧 Fitting TF-IDF + SVD on {len(texts)} texts…", end="", flush=True)
        tfidf_mat = self.tfidf.fit_transform(texts)
        self.svd.fit(tfidf_mat)
        self._fitted = True
        var = self.svd.explained_variance_ratio_.sum()
        print(f" done  (explained variance={var:.3f})")

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        tfidf_mat = self.tfidf.transform(texts)
        vecs      = self.svd.transform(tfidf_mat)
        vecs      = normalize(vecs, norm="l2")
        return vecs.tolist()

    def fit_and_embed(self, texts: List[str]) -> List[List[float]]:
        self.fit(texts)
        return self.embed(texts)

    @property
    def dim(self) -> int:
        return self.n_components


# ── Vector Store (ChromaDB) ────────────────────────────────────────────────

class VectorStore:
    def __init__(self, strategy_name: str, persist_dir: str = CHROMA_BASE_DIR,
                 reset: bool = False):
        self.strategy_name = strategy_name
        self.collection_name = f"stm32_{strategy_name}"
        self.persist_dir = persist_dir
        self._parent_store: dict = {}  # parent_id → parent_text (for parent-child)

        db_path = os.path.join(persist_dir, strategy_name)
        if reset and os.path.exists(db_path):
            shutil.rmtree(db_path)

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, chunks: List[Chunk], embedder: Embedder,
              batch_size: int = 128) -> float:
        """Embed and store chunks. Returns indexing time in seconds."""
        # Separate children from parents for parent-child strategy
        indexable = [c for c in chunks if c.metadata.get("role") != "parent"]
        parents   = [c for c in chunks if c.metadata.get("role") == "parent"]

        # Build parent lookup (not indexed, used at retrieval time)
        for p in parents:
            self._parent_store[p.chunk_id] = p.text

        if not indexable:
            return 0.0

        t0 = time.time()
        for start in range(0, len(indexable), batch_size):
            batch = indexable[start: start + batch_size]
            texts = [c.text for c in batch]
            vecs  = embedder.embed(texts)
            ids   = [c.chunk_id for c in batch]
            metas = [{
                "doc_id":     c.doc_id,
                "strategy":   c.strategy,
                "parent_id":  c.parent_id or "",
                **{k: str(v) for k, v in c.metadata.items()},
            } for c in batch]
            self.collection.add(
                ids=ids,
                embeddings=vecs,
                documents=texts,
                metadatas=metas,
            )
        return time.time() - t0

    def query(self, query_embedding: List[float], top_k: int = 5,
              where: Optional[dict] = None) -> List[RetrievalResult]:
        """Retrieve top-k chunks by cosine similarity."""
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        hits = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            parent_id = meta.get("parent_id", "")
            ctx_text  = self._parent_store.get(parent_id) if parent_id else None
            hits.append(RetrievalResult(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                strategy=meta.get("strategy", ""),
                text=doc,
                distance=float(dist),
                metadata=meta,
                rank=rank + 1,
                context_text=ctx_text,
            ))
        return hits

    @property
    def count(self) -> int:
        return self.collection.count()


# ── Full RAG Pipeline ──────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, strategy_name: str, chunks: List[Chunk],
                 embedder: Embedder, reset: bool = True):
        self.strategy_name = strategy_name
        self.embedder = embedder
        self.store = VectorStore(strategy_name, reset=reset)
        self.index_time = 0.0
        self._indexed = False
        self._all_chunks = chunks

    def build_index(self, batch_size: int = 128) -> float:
        print(f"   🔨 Indexing [{self.strategy_name}] "
              f"({len(self._all_chunks)} total chunks)…", end="", flush=True)
        self.index_time = self.store.index(
            self._all_chunks, self.embedder, batch_size=batch_size
        )
        self._indexed = True
        print(f" {self.index_time:.2f}s  ({self.store.count} vectors)")
        return self.index_time

    def retrieve(self, question: str, top_k: int = 5) -> Tuple[List[RetrievalResult], float]:
        """Embed question and retrieve chunks. Returns (results, latency_ms)."""
        t0 = time.perf_counter()
        q_vec = self.embedder.embed([question])[0]
        results = self.store.query(q_vec, top_k=top_k)
        latency = (time.perf_counter() - t0) * 1000
        return results, latency

    @property
    def n_indexed(self) -> int:
        return self.store.count


# ── Multi-Strategy Orchestrator ────────────────────────────────────────────

class RAGBenchmark:
    def __init__(self, docs: List[dict], questions: List[dict],
                 embedding_model: str = EMBEDDING_MODEL):
        self.docs = docs
        self.questions = questions
        self.embedder = Embedder(embedding_model)
        self.pipelines: dict = {}
        self.results: dict = {}
        self._corpus_fitted = False

    def _ensure_corpus_fitted(self, all_chunks_map: dict):
        """Fit TF-IDF on all texts across all strategies (union corpus)."""
        if self._corpus_fitted:
            return
        all_texts = []
        for chunks in all_chunks_map.values():
            for c in chunks:
                if c.metadata.get("role") != "parent":
                    all_texts.append(c.text)
        # Add questions to corpus so query terms are in vocabulary
        for q in self.questions:
            all_texts.append(q["question"])
        self.embedder.fit(list(set(all_texts)))
        self._corpus_fitted = True

    def setup_all_strategies(self, all_chunks_map: dict, reset: bool = True):
        """Fit embedder once, then index all strategies."""
        self._ensure_corpus_fitted(all_chunks_map)
        for strategy_name, chunks in all_chunks_map.items():
            pipe = RAGPipeline(strategy_name, chunks, self.embedder, reset=reset)
            pipe.build_index()
            self.pipelines[strategy_name] = pipe

    def setup_strategy(self, strategy_name: str, chunks: List[Chunk],
                       reset: bool = True) -> "RAGPipeline":
        pipe = RAGPipeline(strategy_name, chunks, self.embedder, reset=reset)
        pipe.build_index()
        self.pipelines[strategy_name] = pipe
        return pipe

    def run_eval(self, top_k: int = 5) -> dict:
        """Run all questions against all indexed strategies."""
        print(f"\n🔍 Running evaluation ({len(self.questions)} questions × "
              f"{len(self.pipelines)} strategies)…")
        raw = {}
        for strategy_name, pipe in self.pipelines.items():
            raw[strategy_name] = {}
            latencies = []
            for q in self.questions:
                results, lat = pipe.retrieve(q["question"], top_k=top_k)
                raw[strategy_name][q["id"]] = {
                    "question":   q["question"],
                    "results":    results,
                    "latency_ms": lat,
                }
                latencies.append(lat)
            avg_lat = sum(latencies) / len(latencies)
            print(f"   ✅ {strategy_name:20s}  avg_latency={avg_lat:.1f}ms")
        self.results = raw
        return raw

    def compute_metrics(self) -> dict:
        """
        Compute retrieval metrics for each strategy × question.

        Relevance signal: a retrieved chunk is considered relevant if its
        doc metadata (peripheral or topic) matches the ground-truth annotation
        in the eval question. This is a proxy for recall in absence of human labels.
        """
        metrics = {}
        for strategy, q_results in self.results.items():
            strategy_metrics = []
            for qid, data in q_results.items():
                q_meta = next((q for q in self.questions if q["id"] == qid), {})
                results: List[RetrievalResult] = data["results"]

                # Relevance: chunk peripheral or topic matches question annotation
                def is_relevant(r: RetrievalResult) -> bool:
                    p = q_meta.get("relevant_periph", "")
                    t = q_meta.get("relevant_topic", "")
                    m = r.metadata
                    periph_match = p.lower() in m.get("peripheral", "").lower() if p else False
                    topic_match  = t.lower() in m.get("topic", "").lower() if t else False
                    text_match   = p.lower() in r.text.lower() if p else False
                    return periph_match or topic_match or text_match

                relevances = [1 if is_relevant(r) else 0 for r in results]
                n = len(results)

                # Precision@k
                p_at_1 = relevances[0] if n > 0 else 0
                p_at_3 = sum(relevances[:3]) / 3 if n >= 3 else sum(relevances) / max(n, 1)
                p_at_5 = sum(relevances[:5]) / 5 if n >= 5 else sum(relevances) / max(n, 1)

                # MRR
                mrr = 0.0
                for rank, rel in enumerate(relevances, 1):
                    if rel:
                        mrr = 1.0 / rank
                        break

                # Average similarity score (lower distance = more similar for cosine)
                avg_sim = 1 - (sum(r.distance for r in results) / n) if n else 0

                # Context length (proxy for richness vs noise)
                avg_len = sum(len(r.text.split()) for r in results) / n if n else 0

                strategy_metrics.append({
                    "q_id":       qid,
                    "p@1":        p_at_1,
                    "p@3":        p_at_3,
                    "p@5":        p_at_5,
                    "mrr":        mrr,
                    "avg_sim":    round(avg_sim, 4),
                    "avg_len":    round(avg_len, 1),
                    "latency_ms": round(data["latency_ms"], 2),
                })

            # Aggregate
            def mean(key):
                return round(sum(m[key] for m in strategy_metrics) /
                             len(strategy_metrics), 4)

            metrics[strategy] = {
                "per_question": strategy_metrics,
                "aggregate": {
                    "p@1":        mean("p@1"),
                    "p@3":        mean("p@3"),
                    "p@5":        mean("p@5"),
                    "mrr":        mean("mrr"),
                    "avg_sim":    mean("avg_sim"),
                    "avg_len":    mean("avg_len"),
                    "latency_ms": mean("latency_ms"),
                    "n_vectors":  self.pipelines[strategy].n_indexed,
                },
            }

        return metrics

    def save_results(self, path: str = "results/raw_results.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for strategy, q_data in self.results.items():
            serializable[strategy] = {}
            for qid, data in q_data.items():
                serializable[strategy][qid] = {
                    "question":   data["question"],
                    "latency_ms": data["latency_ms"],
                    "results": [
                        {
                            "rank":     r.rank,
                            "doc_id":   r.doc_id,
                            "text":     r.text[:300],
                            "distance": r.distance,
                            "metadata": r.metadata,
                        }
                        for r in data["results"]
                    ],
                }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"💾 Raw results saved → {path}")


if __name__ == "__main__":
    # Quick smoke test
    from data_generator import generate_dataset, EVAL_QUESTIONS
    from chunking import apply_all_strategies

    docs = generate_dataset(20, "data/stm32cube_kb.json")
    all_chunks = apply_all_strategies(docs)
    bench = RAGBenchmark(docs, EVAL_QUESTIONS[:3])
    for name, chunks in list(all_chunks.items())[:2]:
        bench.setup_strategy(name, chunks)
    bench.run_eval(top_k=3)
    metrics = bench.compute_metrics()
    for s, m in metrics.items():
        print(f"{s}: {m['aggregate']}")
