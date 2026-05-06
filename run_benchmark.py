"""
STM32Cube RAG Benchmark — Main Runner
Usage: python run_benchmark.py [--docs N] [--top-k K] [--reset]
"""

import json
import time
import argparse
from pathlib import Path

from data_generator import generate_dataset, save_eval_questions, EVAL_QUESTIONS
from chunking import apply_all_strategies, chunk_stats
from rag_pipeline import RAGBenchmark

BANNER = """
╔══════════════════════════════════════════════════════╗
║   STM32Cube RAG Chunking Benchmark                   ║
║   Embedding: sentence-transformers/all-MiniLM-L6-v2  ║
║   VectorDB : ChromaDB (local, persistent)            ║
╚══════════════════════════════════════════════════════╝
"""


def run(n_docs: int = 120, top_k: int = 5, reset: bool = True,
        data_path: str = "data/stm32cube_kb.json"):

    print(BANNER)
    t_total = time.time()

    # ── 1. Data ─────────────────────────────────────────────────────────────
    print("━" * 56)
    print("STEP 1 — Loading / generating data")
    print("━" * 56)

    if Path(data_path).exists() and not reset:
        with open(data_path) as f:
            docs = json.load(f)
        print(f"📂 Loaded {len(docs)} existing documents from {data_path}")
    else:
        docs = generate_dataset(n_docs, data_path)

    questions = EVAL_QUESTIONS
    save_eval_questions("data/eval_questions.json")
    print(f"❓ Evaluation set: {len(questions)} questions")

    # ── 2. Chunking ─────────────────────────────────────────────────────────
    print("\n━" * 56)
    print("STEP 2 — Applying chunking strategies")
    print("━" * 56)

    all_chunks = apply_all_strategies(docs)
    stats = chunk_stats(all_chunks)

    print(f"\n{'Strategy':<22} {'Chunks':>8} {'Avg Words':>10} {'Min':>6} {'Max':>6}")
    print("-" * 56)
    for name, s in stats.items():
        print(f"{name:<22} {s['n_chunks']:>8} {s['avg_words']:>10.1f} "
              f"{s['min_words']:>6} {s['max_words']:>6}")

    # Save chunk stats
    Path("results").mkdir(exist_ok=True)
    with open("results/chunk_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── 3. Indexing ─────────────────────────────────────────────────────────
    print("\n━" * 56)
    print("STEP 3 — Embedding + Indexing into ChromaDB")
    print("━" * 56)

    bench = RAGBenchmark(docs, questions)
    bench.setup_all_strategies(all_chunks, reset=reset)

    index_times = {name: bench.pipelines[name].index_time for name in all_chunks}

    # ── 4. Retrieval Evaluation ──────────────────────────────────────────────
    print("\n━" * 56)
    print(f"STEP 4 — Retrieval evaluation (top_k={top_k})")
    print("━" * 56)

    bench.run_eval(top_k=top_k)
    bench.save_results("results/raw_results.json")

    # ── 5. Metrics ───────────────────────────────────────────────────────────
    print("\n━" * 56)
    print("STEP 5 — Computing metrics")
    print("━" * 56)

    metrics = bench.compute_metrics()

    # Merge chunk stats into metrics for reporting
    for strategy in metrics:
        metrics[strategy]["chunk_stats"] = stats.get(strategy, {})
        metrics[strategy]["aggregate"]["index_time_s"] = round(
            index_times.get(strategy, 0), 2
        )

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("💾 Metrics saved → results/metrics.json")

    # ── 6. Print Summary Table ───────────────────────────────────────────────
    print("\n━" * 56)
    print("RESULTS — Aggregate Metrics per Strategy")
    print("━" * 56)

    header = (f"{'Strategy':<22} {'Chunks':>7} {'P@1':>6} {'P@3':>6} "
              f"{'P@5':>6} {'MRR':>6} {'AvgSim':>8} {'AvgLen':>8} "
              f"{'Lat(ms)':>9} {'IdxT(s)':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for strategy, m in metrics.items():
        ag = m["aggregate"]
        cs = m["chunk_stats"]
        row = {
            "strategy":    strategy,
            "n_chunks":    cs.get("n_chunks", 0),
            "p@1":         ag["p@1"],
            "p@3":         ag["p@3"],
            "p@5":         ag["p@5"],
            "mrr":         ag["mrr"],
            "avg_sim":     ag["avg_sim"],
            "avg_len":     ag["avg_len"],
            "latency_ms":  ag["latency_ms"],
            "index_time_s": ag["index_time_s"],
        }
        rows.append(row)
        print(f"{strategy:<22} {row['n_chunks']:>7} {row['p@1']:>6.3f} "
              f"{row['p@3']:>6.3f} {row['p@5']:>6.3f} {row['mrr']:>6.3f} "
              f"{row['avg_sim']:>8.4f} {row['avg_len']:>8.1f} "
              f"{row['latency_ms']:>9.1f} {row['index_time_s']:>8.2f}")

    with open("results/summary_table.json", "w") as f:
        json.dump(rows, f, indent=2)

    elapsed = time.time() - t_total
    print(f"\n✅ Benchmark complete in {elapsed:.1f}s")
    print("📁 Outputs: results/metrics.json | results/raw_results.json | "
          "results/chunk_stats.json | results/summary_table.json")

    return metrics, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STM32Cube RAG Benchmark")
    parser.add_argument("--docs",  type=int, default=120,
                        help="Number of synthetic docs to generate (default: 120)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K results to retrieve per question (default: 5)")
    parser.add_argument("--reset", action="store_true", default=True,
                        help="Reset and re-index all vector stores")
    parser.add_argument("--data",  type=str, default="data/stm32cube_kb.json",
                        help="Path to existing data file (or generate new)")
    args = parser.parse_args()

    run(n_docs=args.docs, top_k=args.top_k, reset=args.reset, data_path=args.data)
