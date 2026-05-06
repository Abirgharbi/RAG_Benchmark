"""
loader_real.py — Adaptateur pour les vraies données STM32Cube
============================================================
Lit le JSON produit par ton pipeline v2.0 et le normalise
en documents compatibles avec chunking.py + rag_pipeline.py.

Structure JSON attendue (d'après tes exemples réels) :
  - source_kind : "file" | "issue" | "example"
  - id, repo, mcu_series, mcu_package
  - clean_text  : le texte principal (toujours présent)
  - path, file_type, component, board, category
  - Pour les issues : issue_title, issue_number, labels,
                      state, is_confirmed_bug, layer, severity, issue_kind
  - Pour les files  : path, file_type, github_url

Usage direct :
  python loader_real.py --input mon_fichier.json --output data/kb.json --stats
"""

import json
import re
import argparse
from pathlib import Path
from typing import Optional


# ── Types de fichiers à EXCLURE du RAG (peu de valeur sémantique) ──────────
SKIP_FILE_TYPES = {
    "repo_metadata",   # .gitmodules, CODE_OF_CONDUCT, LICENCE…
    "binary",
    "image",
}

SKIP_PATHS_PATTERNS = [
    r"\.gitmodules$",
    r"CODE_OF_CONDUCT",
    r"LICENSE",
    r"LICENCE",
    r"\.github/",
    r"Makefile$",
    r"\.ioc$",
]
_SKIP_RE = re.compile("|".join(SKIP_PATHS_PATTERNS), re.IGNORECASE)


# ── Extraction de paragraphes depuis clean_text ────────────────────────────

def extract_paragraphs(text: str, min_len: int = 40) -> list[str]:
    """
    Découpe le texte brut en paragraphes exploitables.
    Gère les blocs de code, les listes, et la prose normale.
    """
    if not text:
        return []

    # Normalise les retours à la ligne multiples
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Sépare sur double newline
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Fusionne les très petits fragments avec le suivant
    merged = []
    buf = ""
    for p in raw_paras:
        if len(buf) + len(p) < min_len * 2:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)

    return [p for p in merged if len(p) >= min_len]


# ── Normalisation d'un document "file" ────────────────────────────────────

def normalize_file(doc: dict) -> Optional[dict]:
    path      = doc.get("path", "") or ""
    file_type = doc.get("file_type", "") or ""
    text      = doc.get("clean_text", "") or ""

    # Filtrer les fichiers sans valeur sémantique
    if file_type in SKIP_FILE_TYPES:
        return None
    if _SKIP_RE.search(path):
        return None
    if len(text.strip()) < 80:
        return None

    paragraphs = extract_paragraphs(text)

    return {
        "id":           doc["id"],
        "source_kind":  "file",
        "title":        f"[{doc.get('mcu_series','?')}] {path}",
        "content":      text,
        "paragraphs":   paragraphs,
        # Champs de filtrage / métriques
        "mcu_series":   doc.get("mcu_series", ""),
        "mcu_package":  doc.get("mcu_package", ""),
        "repo":         doc.get("repo", ""),
        "component":    doc.get("component", "") or "",
        "board":        doc.get("board", "") or "",
        "category":     doc.get("category", "") or "",
        "file_type":    file_type,
        "path":         path,
        "github_url":   doc.get("github_url", ""),
        # Champs legacy attendus par chunking.py
        "peripheral":   doc.get("component", "") or "",
        "mcu_family":   doc.get("mcu_series", ""),
        "topic":        file_type or "file",
        "metadata": {
            "source":      doc.get("repo", ""),
            "doc_type":    "file",
            "file_type":   file_type,
            "word_count":  len(text.split()),
            "ingest_version": doc.get("ingest_version", ""),
        },
    }


# ── Normalisation d'un document "issue" ────────────────────────────────────

def normalize_issue(doc: dict) -> Optional[dict]:
    text  = doc.get("clean_text", "") or ""
    title = doc.get("issue_title", "") or ""
    if len(text.strip()) < 80:
        return None

    # Enrichit le texte avec le titre pour le rendre plus retrouvable
    full_text = f"Issue #{doc.get('issue_number','?')}: {title}\n\n{text}"
    paragraphs = extract_paragraphs(full_text)

    labels    = doc.get("labels", []) or []
    component = doc.get("component", "") or ""
    layer     = doc.get("layer", "") or ""
    severity  = doc.get("severity", "") or ""

    return {
        "id":              doc["id"],
        "source_kind":     "issue",
        "title":           f"[{doc.get('mcu_series','?')}] #{doc.get('issue_number','?')}: {title}",
        "content":         full_text,
        "paragraphs":      paragraphs,
        # Issue-specific
        "issue_number":    doc.get("issue_number"),
        "issue_title":     title,
        "issue_kind":      doc.get("issue_kind", ""),
        "state":           doc.get("state", ""),
        "labels":          labels,
        "is_confirmed_bug": doc.get("is_confirmed_bug", False),
        "layer":           layer,
        "severity":        severity,
        "created_at":      doc.get("created_at", ""),
        # Champs communs
        "mcu_series":      doc.get("mcu_series", ""),
        "mcu_package":     doc.get("mcu_package", ""),
        "repo":            doc.get("repo", ""),
        "component":       component,
        "board":           doc.get("board", "") or "",
        "github_url":      doc.get("github_url", ""),
        # Champs legacy pour chunking.py
        "peripheral":      component,
        "mcu_family":      doc.get("mcu_series", ""),
        "topic":           doc.get("issue_kind", "issue"),
        "metadata": {
            "source":       doc.get("repo", ""),
            "doc_type":     "issue",
            "layer":        layer,
            "severity":     severity,
            "labels":       ",".join(labels),
            "word_count":   len(full_text.split()),
            "ingest_version": doc.get("ingest_version", ""),
        },
    }


# ── Normalisation d'un document "example" ─────────────────────────────────

def normalize_example(doc: dict) -> Optional[dict]:
    text = doc.get("clean_text", "") or ""
    if len(text.strip()) < 80:
        return None

    name = doc.get("example_name", "") or doc.get("path", "") or "example"
    paragraphs = extract_paragraphs(text)

    return {
        "id":           doc["id"],
        "source_kind":  "example",
        "title":        f"[{doc.get('mcu_series','?')}] Example: {name}",
        "content":      text,
        "paragraphs":   paragraphs,
        "example_name": name,
        "mcu_series":   doc.get("mcu_series", ""),
        "mcu_package":  doc.get("mcu_package", ""),
        "repo":         doc.get("repo", ""),
        "component":    doc.get("component", "") or "",
        "board":        doc.get("board", "") or "",
        "category":     doc.get("category", "") or "",
        "github_url":   doc.get("github_url", ""),
        # Champs legacy
        "peripheral":   doc.get("component", "") or "",
        "mcu_family":   doc.get("mcu_series", ""),
        "topic":        "example",
        "metadata": {
            "source":    doc.get("repo", ""),
            "doc_type":  "example",
            "word_count": len(text.split()),
            "ingest_version": doc.get("ingest_version", ""),
        },
    }


# ── Dispatcher principal ────────────────────────────────────────────────────

NORMALIZERS = {
    "file":    normalize_file,
    "issue":   normalize_issue,
    "example": normalize_example,
}


def load_and_normalize(
    input_path: str,
    max_docs: int = 0,
    filter_series: str = "",
    filter_component: str = "",
    filter_kind: str = "",
) -> tuple[list[dict], dict]:
    """
    Charge le JSON STM32Cube réel et retourne (docs_normalisés, stats).

    Args:
        input_path      : chemin vers ton fichier JSON
        max_docs        : limite de docs (0 = tout charger)
        filter_series   : ex. "H7" pour ne garder que cette série
        filter_component: ex. "DMA" pour filtrer par composant
        filter_kind     : "file" | "issue" | "example" | "" (tous)
    """
    print(f"📂 Chargement : {input_path}")
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Le JSON doit être une liste de documents.")

    print(f"   {len(raw)} documents bruts trouvés")

    stats = {
        "total_raw": len(raw),
        "by_kind":   {"file": 0, "issue": 0, "example": 0, "other": 0},
        "skipped":   0,
        "normalized": 0,
    }

    docs = []
    for raw_doc in raw:
        kind = raw_doc.get("source_kind", raw_doc.get("type", ""))

        # Filtres optionnels
        if filter_kind and kind != filter_kind:
            continue
        if filter_series and raw_doc.get("mcu_series", "") != filter_series:
            continue
        if filter_component:
            comp = (raw_doc.get("component", "") or "").upper()
            if filter_component.upper() not in comp:
                continue

        # Normalisation
        fn = NORMALIZERS.get(kind)
        if fn is None:
            stats["by_kind"]["other"] += 1
            stats["skipped"] += 1
            continue

        normalized = fn(raw_doc)
        if normalized is None:
            stats["skipped"] += 1
            continue

        docs.append(normalized)
        stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
        stats["normalized"] += 1

        if max_docs and len(docs) >= max_docs:
            break

    print(f"   ✅ {stats['normalized']} docs normalisés "
          f"(files={stats['by_kind'].get('file',0)}, "
          f"issues={stats['by_kind'].get('issue',0)}, "
          f"examples={stats['by_kind'].get('example',0)}, "
          f"skipped={stats['skipped']})")

    return docs, stats


def save_normalized(docs: list[dict], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"💾 Données normalisées sauvegardées → {output_path}")


# ── Questions d'éval spécifiques aux vraies données ────────────────────────
# À compléter selon tes propres connaissances du corpus

EVAL_QUESTIONS_REAL = [
    # Issues / bugs
    {"id": "rq01",
     "question": "HAL_CRYP_Decrypt_DMA fails for large buffers over 65535 bytes",
     "relevant_topic": "issue", "relevant_periph": "CRYP"},
    {"id": "rq02",
     "question": "DMA transfer size overflow uint16_t HAL bug",
     "relevant_topic": "issue", "relevant_periph": "DMA"},
    {"id": "rq03",
     "question": "CRYP encryption decryption word size DataWidthUnit issue",
     "relevant_topic": "issue", "relevant_periph": "CRYP"},

    # Configuration / init
    {"id": "rq04",
     "question": "How to configure USB device on STM32H7?",
     "relevant_topic": "example", "relevant_periph": "USB"},
    {"id": "rq05",
     "question": "FreeRTOS task configuration STM32H7",
     "relevant_topic": "example", "relevant_periph": "FreeRTOS"},
    {"id": "rq06",
     "question": "ETH Ethernet LwIP STM32H7 configuration",
     "relevant_topic": "example", "relevant_periph": "ETH"},
    {"id": "rq07",
     "question": "SDMMC SD card initialization STM32H7 HAL",
     "relevant_topic": "example", "relevant_periph": "SDMMC"},
    {"id": "rq08",
     "question": "UART DMA transmit receive STM32H7",
     "relevant_topic": "example", "relevant_periph": "UART"},

    # BSP / drivers
    {"id": "rq09",
     "question": "STM32H743I-EVAL BSP initialization LCD LTDC",
     "relevant_topic": "file", "relevant_periph": "LTDC"},
    {"id": "rq10",
     "question": "NUCLEO-H7 board support package GPIO configuration",
     "relevant_topic": "file", "relevant_periph": "GPIO"},

    # Sécurité / crypto
    {"id": "rq11",
     "question": "mbedTLS TLS configuration STM32H7",
     "relevant_topic": "example", "relevant_periph": "mbedTLS"},
    {"id": "rq12",
     "question": "AES encryption hardware accelerator STM32H7",
     "relevant_topic": "example", "relevant_periph": "CRYP"},

    # Middleware
    {"id": "rq13",
     "question": "FatFs filesystem SD card STM32H7",
     "relevant_topic": "example", "relevant_periph": "FatFs"},
    {"id": "rq14",
     "question": "LibJPEG JPEG decoding STM32H7 DMA2D",
     "relevant_topic": "example", "relevant_periph": "JPEG"},

    # HAL specifics
    {"id": "rq15",
     "question": "HAL_Init SystemClock_Config startup sequence STM32H7",
     "relevant_topic": "example", "relevant_periph": "RCC"},
    {"id": "rq16",
     "question": "ADC multimode DMA STM32H7xx HAL driver",
     "relevant_topic": "file", "relevant_periph": "ADC"},
    {"id": "rq17",
     "question": "SPI full duplex master DMA STM32H7",
     "relevant_topic": "example", "relevant_periph": "SPI"},
    {"id": "rq18",
     "question": "TIM PWM output configuration STM32H7",
     "relevant_topic": "example", "relevant_periph": "TIM"},
    {"id": "rq19",
     "question": "IWDG WWDG watchdog configuration STM32H7",
     "relevant_topic": "example", "relevant_periph": "WWDG"},
    {"id": "rq20",
     "question": "FLASH programming erase sectors STM32H7",
     "relevant_topic": "file", "relevant_periph": "FLASH"},
]


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chargeur de données STM32Cube réelles")
    parser.add_argument("--input",     required=True,  help="Fichier JSON source")
    parser.add_argument("--output",    default="data/kb_real.json")
    parser.add_argument("--max",       type=int, default=0, help="Limite de docs (0=tous)")
    parser.add_argument("--series",    default="", help="Filtrer par série ex: H7")
    parser.add_argument("--component", default="", help="Filtrer par composant ex: DMA")
    parser.add_argument("--kind",      default="", help="file | issue | example | (vide=tous)")
    parser.add_argument("--stats",     action="store_true", help="Afficher les stats détaillées")
    args = parser.parse_args()

    docs, stats = load_and_normalize(
        args.input,
        max_docs=args.max,
        filter_series=args.series,
        filter_component=args.component,
        filter_kind=args.kind,
    )

    if args.stats:
        print("\n── Statistiques ───────────────────────────────────")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if docs:
            lens = [len(d["content"].split()) for d in docs]
            print(f"  avg_words: {sum(lens)/len(lens):.0f}")
            print(f"  min_words: {min(lens)}")
            print(f"  max_words: {max(lens)}")

    save_normalized(docs, args.output)

    # Sauvegarder aussi les questions d'éval spécifiques aux vraies données
    import json
    Path("data").mkdir(exist_ok=True)
    with open("data/eval_questions_real.json", "w") as f:
        json.dump(EVAL_QUESTIONS_REAL, f, indent=2)
    print(f"💾 Questions d'éval sauvegardées → data/eval_questions_real.json")
