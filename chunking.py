"""
Chunking Strategies for STM32Cube RAG Benchmark
Implements 5 distinct chunking approaches with metadata preservation.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: Optional[str] = None   # for parent-child strategy


# ── Utility ────────────────────────────────────────────────────────────────

def _make_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def _base_meta(doc: dict) -> dict:
    return {
        "title":      doc.get("title", ""),
        "peripheral": doc.get("peripheral", ""),
        "mcu_family": doc.get("mcu_family", ""),
        "topic":      doc.get("topic", ""),
        "source":     doc.get("metadata", {}).get("source", ""),
    }


# ── Strategy 1 : Full Document ─────────────────────────────────────────────

def chunk_full_document(docs: List[dict]) -> List[Chunk]:
    """One chunk = entire document. Maximum context, maximum noise."""
    chunks = []
    for doc in docs:
        text = f"{doc['title']}\n\n{doc['content']}"
        if doc.get("faq"):
            text += f"\n\nQ: {doc['faq']['question']}\nA: {doc['faq']['answer']}"
        chunks.append(Chunk(
            chunk_id=_make_id("full_"),
            doc_id=doc["id"],
            strategy="full_document",
            text=text,
            metadata={**_base_meta(doc), "char_count": len(text)},
        ))
    return chunks


# ── Strategy 2 : Paragraph Chunks ──────────────────────────────────────────

def chunk_by_paragraph(docs: List[dict]) -> List[Chunk]:
    """Split on double-newlines. Natural prose boundaries."""
    chunks = []
    for doc in docs:
        # Use pre-split paragraphs if available, else split content
        paragraphs = doc.get("paragraphs") or [
            p.strip() for p in re.split(r"\n{2,}", doc["content"]) if p.strip()
        ]
        for i, para in enumerate(paragraphs):
            if len(para.strip()) < 30:
                continue
            text = f"[{doc['title']}]\n{para.strip()}"
            chunks.append(Chunk(
                chunk_id=_make_id("para_"),
                doc_id=doc["id"],
                strategy="paragraph",
                text=text,
                metadata={
                    **_base_meta(doc),
                    "para_index": i,
                    "char_count": len(text),
                },
            ))
    return chunks


# ── Strategy 3 : Fixed-Size Sliding Window ─────────────────────────────────

def chunk_sliding_window(
    docs: List[dict],
    chunk_size: int = 300,    # tokens approximated as words
    overlap: int = 50,
) -> List[Chunk]:
    """Fixed word-count windows with configurable overlap."""
    chunks = []
    for doc in docs:
        words = doc["content"].split()
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            window = words[start: start + chunk_size]
            if len(window) < 20:
                continue
            text = f"[{doc['title']}]\n" + " ".join(window)
            chunks.append(Chunk(
                chunk_id=_make_id("win_"),
                doc_id=doc["id"],
                strategy="sliding_window",
                text=text,
                metadata={
                    **_base_meta(doc),
                    "word_start": start,
                    "word_end": start + len(window),
                    "char_count": len(text),
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                },
            ))
    return chunks


# ── Strategy 4 : Section-Based (Semantic Headers) ──────────────────────────

_SECTION_PATTERNS = [
    r"(?:#{1,3}\s+.+)",                  # markdown headers
    r"(?:^[A-Z][A-Za-z ]{5,50}:$)",      # TITLE: style
    r"(?:^\d+\.\s+[A-Z].{5,60}$)",       # 1. Title style
]
_SECTION_RE = re.compile(
    "|".join(_SECTION_PATTERNS), re.MULTILINE
)


def chunk_by_section(docs: List[dict]) -> List[Chunk]:
    """
    Try to split on detected section headers.
    Falls back to paragraph chunking if no headers found.
    """
    chunks = []
    for doc in docs:
        content = doc["content"]
        # Find all header positions
        header_spans = [(m.start(), m.group()) for m in _SECTION_RE.finditer(content)]

        if len(header_spans) < 2:
            # No structure detected → fall back to paragraph
            para_chunks = chunk_by_paragraph([doc])
            for c in para_chunks:
                c.strategy = "section"
                c.metadata["fallback"] = "paragraph"
            chunks.extend(para_chunks)
            continue

        # Build sections between headers
        boundaries = [pos for pos, _ in header_spans] + [len(content)]
        for j, (start, header) in enumerate(header_spans):
            end = boundaries[j + 1]
            section_text = content[start:end].strip()
            if len(section_text) < 30:
                continue
            text = f"[{doc['title']} › {header.strip()}]\n{section_text}"
            chunks.append(Chunk(
                chunk_id=_make_id("sec_"),
                doc_id=doc["id"],
                strategy="section",
                text=text,
                metadata={
                    **_base_meta(doc),
                    "section_header": header.strip(),
                    "char_count": len(text),
                },
            ))
    return chunks


# ── Strategy 5 : Parent-Child (Hierarchical) ───────────────────────────────

def chunk_parent_child(
    docs: List[dict],
    child_size: int = 150,   # words per child chunk
    child_overlap: int = 20,
) -> List[Chunk]:
    """
    Two-level hierarchy:
    - Parent = full paragraph  (stored, retrieved for context)
    - Child  = sliding window over the paragraph (indexed for retrieval)

    At retrieval time, the *parent* text is returned, giving broader context
    while the child's embedding captures the precise semantic unit.
    """
    chunks = []
    for doc in docs:
        paragraphs = doc.get("paragraphs") or [
            p.strip() for p in re.split(r"\n{2,}", doc["content"]) if p.strip()
        ]
        for para in paragraphs:
            if len(para.split()) < 20:
                continue

            # Create parent
            parent_id = _make_id("parent_")
            parent_text = f"[{doc['title']}]\n{para}"
            chunks.append(Chunk(
                chunk_id=parent_id,
                doc_id=doc["id"],
                strategy="parent_child",
                text=parent_text,
                metadata={
                    **_base_meta(doc),
                    "role": "parent",
                    "char_count": len(parent_text),
                },
            ))

            # Create children over this paragraph
            words = para.split()
            step = max(1, child_size - child_overlap)
            for start in range(0, len(words), step):
                child_words = words[start: start + child_size]
                if len(child_words) < 10:
                    continue
                child_text = f"[{doc['title']}]\n" + " ".join(child_words)
                chunks.append(Chunk(
                    chunk_id=_make_id("child_"),
                    doc_id=doc["id"],
                    strategy="parent_child",
                    text=child_text,
                    parent_id=parent_id,
                    metadata={
                        **_base_meta(doc),
                        "role": "child",
                        "parent_id": parent_id,
                        "word_start": start,
                        "char_count": len(child_text),
                    },
                ))
    return chunks


# ── Registry ───────────────────────────────────────────────────────────────

STRATEGIES = {
    "full_document":   chunk_full_document,
    "paragraph":       chunk_by_paragraph,
    "sliding_window":  chunk_sliding_window,
    "section":         chunk_by_section,
    "parent_child":    chunk_parent_child,
}


def apply_all_strategies(docs: List[dict]) -> dict:
    """Return {strategy_name: [Chunk, ...]} for all strategies."""
    return {name: fn(docs) for name, fn in STRATEGIES.items()}


def chunk_stats(all_chunks: dict) -> dict:
    stats = {}
    for name, chunks in all_chunks.items():
        texts = [c.text for c in chunks]
        lengths = [len(t.split()) for t in texts]
        stats[name] = {
            "n_chunks":   len(chunks),
            "avg_words":  round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "min_words":  min(lengths) if lengths else 0,
            "max_words":  max(lengths) if lengths else 0,
        }
    return stats


if __name__ == "__main__":
    from data_generator import generate_dataset
    docs = generate_dataset(20, "data/stm32cube_kb.json")
    all_chunks = apply_all_strategies(docs)
    stats = chunk_stats(all_chunks)
    for name, s in stats.items():
        print(f"{name:20s}  chunks={s['n_chunks']:4d}  avg_words={s['avg_words']:6.1f}  "
              f"min={s['min_words']}  max={s['max_words']}")
