"""
run_real.py — STM32Cube RAG Benchmark + Preprocessing R&D
=========================================================

Pipeline complet :
- Chargement & normalisation
- Analyse preprocessing R&D
- Chunking multi-stratégies
- Embedding TF-IDF + LSA
- Indexation ChromaDB
- Benchmark retrieval
- Calcul des métriques
- Génération automatique de recommandations

Usage :
    python run_real.py --input mon_kb.json
"""

import json
import time
import argparse
from pathlib import Path

from loader_real import load_and_normalize, EVAL_QUESTIONS_REAL
from chunking import apply_all_strategies, chunk_stats
from rag_pipeline import RAGBenchmark


# ═══════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════════╗
║      STM32Cube RAG Benchmark — Données réelles          ║
║                                                          ║
║      Embedding : TF-IDF + LSA 256-dim                   ║
║      VectorDB  : ChromaDB persisté                      ║
║      Evaluation : P@K + MRR + Latency                   ║
║      R&D : Preprocessing Analysis                       ║
╚══════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING R&D ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_preprocessing_quality(docs):

    report = []

    report.append("# STM32Cube Preprocessing Analysis\n")

    report.append(
        "This report analyzes STM32Cube GitHub issues and documents "
        "before ingestion into the Parent-Child RAG pipeline.\n"
    )

    total_docs = len(docs)

    # ─────────────────────────────────────────────
    # Counters
    # ─────────────────────────────────────────────

    missing_series = 0
    missing_component = 0
    missing_titles = 0

    duplicated_docs = 0

    short_docs = 0
    noisy_docs = 0
    code_docs = 0

    very_large_docs = 0

    seen = set()

    # ─────────────────────────────────────────────
    # Document analysis
    # ─────────────────────────────────────────────

    for d in docs:

        content = d.get("content", "")
        title = d.get("title", "")

        # Missing metadata

        if not d.get("series"):
            missing_series += 1

        if not d.get("component"):
            missing_component += 1

        if not title:
            missing_titles += 1

        # Duplicate detection

        signature = content[:300]

        if signature in seen:
            duplicated_docs += 1

        seen.add(signature)

        # Short docs

        n_words = len(content.split())

        if n_words < 30:
            short_docs += 1

        # Very large docs

        if n_words > 3000:
            very_large_docs += 1

        # Noise detection

        special_chars = sum(
            1 for c in content
            if c in "@@@###$$$%%%^^^^====>>>>"
        )

        if len(content) > 0:

            ratio = special_chars / len(content)

            if ratio > 0.05:
                noisy_docs += 1

        # Code detection

        if (
            "```" in content
            or "HAL_" in content
            or "#include" in content
        ):
            code_docs += 1

    # ─────────────────────────────────────────────
    # Dataset overview
    # ─────────────────────────────────────────────

    report.append("## Dataset Overview\n")

    report.append(f"- Total documents analyzed: **{total_docs}**")
    report.append(f"- Missing series metadata: **{missing_series}**")
    report.append(f"- Missing component metadata: **{missing_component}**")
    report.append(f"- Missing titles: **{missing_titles}**")
    report.append(f"- Potential duplicated documents: **{duplicated_docs}**")
    report.append(f"- Very short documents/issues: **{short_docs}**")
    report.append(f"- Very large documents: **{very_large_docs}**")
    report.append(f"- Noisy documents detected: **{noisy_docs}**")
    report.append(f"- Documents containing code: **{code_docs}**")

    # ─────────────────────────────────────────────
    # Recommendations
    # ─────────────────────────────────────────────

    report.append("\n## Recommended Preprocessing Improvements\n")

    if missing_series > 0:

        report.append(
            "- Add MCU series metadata (H7, F4, L4...) "
            "to improve filtering and retrieval precision."
        )

    if missing_component > 0:

        report.append(
            "- Add component metadata (DMA, UART, USB...) "
            "for component-aware retrieval."
        )

    if missing_titles > 0:

        report.append(
            "- Preserve GitHub issue titles during preprocessing."
        )

    if duplicated_docs > 0:

        report.append(
            "- Remove duplicated GitHub issues before indexing."
        )

    if short_docs > 0:

        report.append(
            "- Merge very short issues with parent context "
            "to avoid semantic fragmentation."
        )

    if very_large_docs > 0:

        report.append(
            "- Split very large documents semantically "
            "before embedding generation."
        )

    if noisy_docs > 0:

        report.append(
            "- Clean markdown artifacts, logs, and noisy symbols."
        )

    if code_docs > 0:

        report.append(
            "- Preserve code blocks separately from natural language text."
        )

    # ─────────────────────────────────────────────
    # Parent-child best practices
    # ─────────────────────────────────────────────

    report.append("\n## Parent-Child Chunking Best Practices\n")

    best_practices = [

        "Preserve hierarchical parent-child relationships.",

        "Store parent references inside child metadata.",

        "Keep section titles inside child chunks.",

        "Avoid splitting code examples across multiple chunks.",

        "Use semantic chunking instead of fixed-size chunking.",

        "Normalize repository paths and technical references.",

        "Use overlap between 10% and 20% to preserve context continuity.",

        "Separate issue discussions from source-code snippets.",

        "Preserve issue comments and linked technical references.",

        "Group STM32 issues by component and MCU series."
    ]

    for bp in best_practices:
        report.append(f"- {bp}")

    # ─────────────────────────────────────────────
    # Conclusion
    # ─────────────────────────────────────────────

    report.append("\n## Conclusion\n")

    report.append(
        "The analysis confirms that preprocessing quality "
        "directly impacts embedding coherence, retrieval quality, "
        "and contextual reconstruction in Parent-Child RAG systems."
    )

    return "\n".join(report)


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run(args):

    print(BANNER)

    t_total = time.time()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # STEP 1 — LOAD DATA
    # ═══════════════════════════════════════════════════════════

    print("━" * 60)
    print("ÉTAPE 1 — Chargement des données STM32Cube")
    print("━" * 60)

    docs, load_stats = load_and_normalize(
        args.input,
        max_docs=args.max,
        filter_series=args.series,
        filter_component=args.component,
        filter_kind=args.kind,
    )

    if not docs:

        print("❌ Aucun document valide trouvé.")
        return

    print(f"\n✅ {len(docs)} documents chargés")

    # Save normalized KB

    norm_path = out_dir / "kb_normalized.json"

    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"💾 KB normalisée → {norm_path}")

    # ═══════════════════════════════════════════════════════════
    # STEP 2 — PREPROCESSING R&D ANALYSIS
    # ═══════════════════════════════════════════════════════════

    print("\n━" * 60)
    print("ÉTAPE 2 — Analyse preprocessing R&D")
    print("━" * 60)

    preprocessing_report = analyze_preprocessing_quality(docs)

    preprocessing_path = (
        out_dir / "preprocessing_recommendations.md"
    )

    with open(preprocessing_path, "w", encoding="utf-8") as f:
        f.write(preprocessing_report)

    print(f"📄 Rapport preprocessing → {preprocessing_path}")

    # ═══════════════════════════════════════════════════════════
    # STEP 3 — EVAL QUESTIONS
    # ═══════════════════════════════════════════════════════════

    print("\n━" * 60)
    print("ÉTAPE 3 — Questions d'évaluation")
    print("━" * 60)

    if args.questions and Path(args.questions).exists():

        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)

        print(f"✅ {len(questions)} questions custom chargées")

    else:

        questions = EVAL_QUESTIONS_REAL

        print(f"✅ {len(questions)} questions par défaut")

    with open(
        out_dir / "eval_questions_used.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            questions,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ═══════════════════════════════════════════════════════════
    # STEP 4 — CHUNKING
    # ═══════════════════════════════════════════════════════════

    print("\n━" * 60)
    print("ÉTAPE 4 — Chunking")
    print("━" * 60)

    all_chunks = apply_all_strategies(docs)

    stats = chunk_stats(all_chunks)

    print(f"\n{'Stratégie':<22} {'Chunks':>8} {'AvgWords':>12}")

    print("-" * 50)

    for name, s in stats.items():

        print(
            f"{name:<22} "
            f"{s['n_chunks']:>8} "
            f"{s['avg_words']:>12.1f}"
        )

    with open(
        out_dir / "chunk_stats.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(stats, f, indent=2)

    # ═══════════════════════════════════════════════════════════
    # STEP 5 — EMBEDDING + INDEXATION
    # ═══════════════════════════════════════════════════════════

    print("\n━" * 60)
    print("ÉTAPE 5 — Embedding + Indexation")
    print("━" * 60)

    bench = RAGBenchmark(docs, questions)

    bench.setup_all_strategies(
        all_chunks,
        reset=args.reset
    )

    index_times = {

        name: bench.pipelines[name].index_time

        for name in all_chunks
    }

    # ═══════════════════════════════════════════════════════════
    # STEP 6 — RETRIEVAL EVALUATION
    # ═══════════════════════════════════════════════════════════

    print("\n━" * 60)
    print("ÉTAPE 6 — Évaluation Retrieval")
    print("━" * 60)

    bench.run_eval(top_k=args.top_k)

    bench.save_results(
        str(out_dir / "raw_results.json")
    )

    # ═══════════════════════════════════════════════════════════
    # STEP 7 — METRICS
    # ═══════════════════════════════════════════════════════════

    print("\n━" * 60)
    print("ÉTAPE 7 — Calcul des métriques")
    print("━" * 60)

    metrics = bench.compute_metrics()

    for strategy in metrics:

        metrics[strategy]["chunk_stats"] = (
            stats.get(strategy, {})
        )

        metrics[strategy]["aggregate"]["index_time_s"] = round(
            index_times.get(strategy, 0),
            2
        )

    metrics_path = out_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"📊 Metrics → {metrics_path}")

    # ═══════════════════════════════════════════════════════════
    # STEP 8 — SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════

    print("\n━" * 60)
    print("RÉSULTATS — Tableau agrégé")
    print("━" * 60)

    header = (
        f"{'Stratégie':<22} "
        f"{'P@1':>6} "
        f"{'P@3':>6} "
        f"{'P@5':>6} "
        f"{'MRR':>6} "
        f"{'Latency':>10}"
    )

    print(header)

    print("-" * len(header))

    rows = []

    for strategy, m in metrics.items():

        ag = m["aggregate"]

        row = {

            "strategy": strategy,

            "p@1": ag["p@1"],

            "p@3": ag["p@3"],

            "p@5": ag["p@5"],

            "mrr": ag["mrr"],

            "latency_ms": ag["latency_ms"],

            "index_time_s": ag["index_time_s"],
        }

        rows.append(row)

        print(
            f"{strategy:<22} "
            f"{row['p@1']:>6.3f} "
            f"{row['p@3']:>6.3f} "
            f"{row['p@5']:>6.3f} "
            f"{row['mrr']:>6.3f} "
            f"{row['latency_ms']:>10.1f}"
        )

    summary_path = out_dir / "summary_table.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"📋 Summary → {summary_path}")

    # ═══════════════════════════════════════════════════════════
    # FINAL
    # ═══════════════════════════════════════════════════════════

    elapsed = time.time() - t_total

    print("\n" + "=" * 60)

    print(f"✅ Benchmark terminé en {elapsed:.1f}s")

    print(f"\n📁 Résultats disponibles dans : {out_dir}/")

    print("\nFichiers générés :")

    print("   • kb_normalized.json")
    print("   • preprocessing_recommendations.md")
    print("   • eval_questions_used.json")
    print("   • chunk_stats.json")
    print("   • raw_results.json")
    print("   • metrics.json")
    print("   • summary_table.json")

    print("=" * 60)

    return bench, metrics, rows


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="STM32Cube RAG Benchmark + Preprocessing R&D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Chemin vers le fichier JSON STM32Cube"
    )

    parser.add_argument(
        "--output",
        default="results_real",
        help="Dossier de sortie"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--series",
        default=""
    )

    parser.add_argument(
        "--component",
        default=""
    )

    parser.add_argument(
        "--kind",
        default=""
    )

    parser.add_argument(
        "--max",
        type=int,
        default=0
    )

    parser.add_argument(
        "--questions",
        default=""
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--no-reset",
        dest="reset",
        action="store_false"
    )

    args = parser.parse_args()

    run(args)