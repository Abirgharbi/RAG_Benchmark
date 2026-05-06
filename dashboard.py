import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="STM32Cube RAG Benchmark",
    page_icon="📊",
    layout="wide"
)

RESULTS_DIR = "results_real"


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_json(path):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


metrics = load_json(f"{RESULTS_DIR}/metrics.json")
summary = load_json(f"{RESULTS_DIR}/summary_table.json")
raw_results = load_json(f"{RESULTS_DIR}/raw_results.json")
chunk_stats = load_json(f"{RESULTS_DIR}/chunk_stats.json")


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.title("📊 STM32Cube RAG Benchmark Dashboard")

st.markdown("""
Ce dashboard permet de visualiser :

- les performances retrieval des stratégies de chunking
- les métriques RAG
- les statistiques de chunks
- les résultats détaillés
- les recommandations preprocessing
""")


# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

st.header("📋 Tableau des résultats")

if summary:

    df = pd.DataFrame(summary)

    st.dataframe(df, width='stretch')

else:
    st.warning("summary_table.json introuvable")


# ═══════════════════════════════════════════════════════════════
# KPI METRICS
# ═══════════════════════════════════════════════════════════════

if summary:

    best_p1 = max(df["p@1"])
    best_mrr = max(df["mrr"])
    best_latency = min(df["latency_ms"])

    col1, col2, col3 = st.columns(3)

    col1.metric("🏆 Best P@1", f"{best_p1:.3f}")
    col2.metric("🎯 Best MRR", f"{best_mrr:.3f}")
    col3.metric("⚡ Lowest Latency", f"{best_latency:.1f} ms")


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE GRAPHS
# ═══════════════════════════════════════════════════════════════

st.header("📈 Visualisations interactives")

if summary:

    metric_options = [
        "p@1",
        "p@3",
        "p@5",
        "mrr",
        "latency_ms",
        "avg_len",
        "index_time_s",
        "n_chunks"
    ]

    selected_metric = st.selectbox(
        "Choisir une métrique",
        metric_options,
        key="metric_selectbox"
    )

    chart_type = st.selectbox(
        "Choisir le type de graphe",
        ["Bar", "Line", "Scatter"],
        key="chart_selectbox"
    )

    if chart_type == "Bar":

        fig = px.bar(
            df,
            x="strategy",
            y=selected_metric,
            text=selected_metric,
            title=f"{selected_metric} par stratégie"
        )

    elif chart_type == "Line":

        fig = px.line(
            df,
            x="strategy",
            y=selected_metric,
            markers=True,
            title=f"{selected_metric} par stratégie"
        )

    else:

        fig = px.scatter(
            df,
            x="strategy",
            y=selected_metric,
            size=selected_metric,
            title=f"{selected_metric} par stratégie"
        )

    st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════
# MULTI METRICS COMPARISON
# ═══════════════════════════════════════════════════════════════

st.header("📊 Comparaison multi-métriques")

if summary:

    selected_metrics = st.multiselect(
        "Choisir plusieurs métriques",
        ["p@1", "p@3", "p@5", "mrr"],
        default=["p@1", "mrr"]
    )

    if selected_metrics:

        fig_multi = px.bar(
            df,
            x="strategy",
            y=selected_metrics,
            barmode="group",
            title="Comparaison des métriques"
        )

        st.plotly_chart(fig_multi, width='stretch')


# ═══════════════════════════════════════════════════════════════
# CHUNK STATS
# ═══════════════════════════════════════════════════════════════

st.header("🧩 Chunk Statistics")

if chunk_stats:

    chunk_rows = []

    for strategy, stats in chunk_stats.items():

        chunk_rows.append({
            "strategy": strategy,
            "n_chunks": stats.get("n_chunks", 0),
            "avg_words": stats.get("avg_words", 0),
            "min_words": stats.get("min_words", 0),
            "max_words": stats.get("max_words", 0),
        })

    chunk_df = pd.DataFrame(chunk_rows)

    st.dataframe(chunk_df, width='stretch')

    chunk_metric = st.selectbox(
        "Choisir une métrique de chunk",
        ["n_chunks", "avg_words", "min_words", "max_words"],
        key="chunk_metric_select"
    )

    fig_chunk = px.bar(
        chunk_df,
        x="strategy",
        y=chunk_metric,
        text=chunk_metric,
        title=f"{chunk_metric} par stratégie"
    )

    st.plotly_chart(fig_chunk, width='stretch')


# ═══════════════════════════════════════════════════════════════
# DETAILED RESULTS
# ═══════════════════════════════════════════════════════════════

st.header("🔍 Résultats détaillés")

if raw_results:

    strategy_names = list(raw_results.keys())

    selected_strategy = st.selectbox(
        "Choisir une stratégie",
        strategy_names,
        key="strategy_selectbox"
    )

    strategy_data = raw_results[selected_strategy]

    if isinstance(strategy_data, dict):

        strategy_items = list(strategy_data.items())

        for idx, (question, result) in enumerate(strategy_items[:10]):

            with st.expander(f"Question {idx + 1}"):

                st.markdown("### ❓ Question")
                st.write(question)

                st.markdown("### 📄 Résultat")
                st.json(result)

    elif isinstance(strategy_data, list):

        for idx, item in enumerate(strategy_data[:10]):

            with st.expander(f"Question {idx + 1}"):

                st.json(item)


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

st.header("🧠 Preprocessing R&D Recommendations")

preprocessing_path = Path(
    f"{RESULTS_DIR}/preprocessing_recommendations.md"
)

if preprocessing_path.exists():

    content = preprocessing_path.read_text(encoding="utf-8")

    st.markdown(content)

else:

    st.info(
        "Le fichier preprocessing_recommendations.md n'existe pas encore."
    )


# ═══════════════════════════════════════════════════════════════
# BEST PRACTICES SECTION
# ═══════════════════════════════════════════════════════════════

st.header("✅ Best Practices — Parent Child Chunking")

best_practices = [

    "Préserver les relations parent-child dans les métadonnées.",

    "Conserver les titres GitHub issues dans les chunks enfants.",

    "Séparer les blocs de code des discussions naturelles.",

    "Éviter les chunks trop petits (<30 mots).",

    "Nettoyer les logs et artefacts markdown.",

    "Ajouter les métadonnées MCU series et component.",

    "Éviter les duplications de documents.",

    "Utiliser un overlap entre 10% et 20%.",

    "Préserver les références techniques STM32Cube.",

    "Grouper les issues par composant et série MCU."
]

for bp in best_practices:
    st.success(bp)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("---")

st.caption(
    "STM32Cube RAG Benchmark Dashboard — PFE Research & Development"
)