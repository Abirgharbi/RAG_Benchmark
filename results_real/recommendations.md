# Recommendations & Best Practices

These recommendations were automatically generated from the benchmark results obtained on STM32Cube datasets.

## Best Performing Strategy

- Best retrieval strategy: **full_document**
- Best MRR score: **0.750**

## Chunking Recommendations

- **full_document** → Average words/chunk = 457.5 → Chunk size appears balanced for retrieval tasks.
- **paragraph** → Average words/chunk = 34.4 → Chunks are very small and may lose semantic context.
- **sliding_window** → Average words/chunk = 239.8 → Chunk size appears balanced for retrieval tasks.
- **section** → Average words/chunk = 52.5 → Chunks are very small and may lose semantic context.
- **parent_child** → Average words/chunk = 55.7 → Chunks are very small and may lose semantic context.

## Parent-Child Best Practices

- Preserve hierarchical parent-child relationships during preprocessing.
- Keep document titles and section headers inside child chunks.
- Use metadata enrichment (series, component, source_kind).
- Avoid arbitrary fixed-size chunk splitting.
- Use semantic-aware chunking whenever possible.
- Normalize repository paths and technical references.
- Preserve code blocks separately from natural language text.
- Remove duplicated documents before indexing.
- Use overlap between 10% and 20% to preserve context continuity.
- Store parent document references inside child chunks.

## Retrieval Observations

- **full_document** → P@1=0.700, P@5=0.550, MRR=0.750, Latency=12.8ms → Moderate retrieval quality. Retrieval latency acceptable.
- **paragraph** → P@1=0.700, P@5=0.600, MRR=0.700, Latency=13.0ms → Moderate retrieval quality. Retrieval latency acceptable.
- **sliding_window** → P@1=0.700, P@5=0.580, MRR=0.746, Latency=13.2ms → Moderate retrieval quality. Retrieval latency acceptable.
- **section** → P@1=0.700, P@5=0.590, MRR=0.700, Latency=13.3ms → Moderate retrieval quality. Retrieval latency acceptable.
- **parent_child** → P@1=0.700, P@5=0.590, MRR=0.713, Latency=13.8ms → Moderate retrieval quality. Retrieval latency acceptable.

## Conclusion

The benchmark confirms that preprocessing quality directly impacts retrieval precision, MRR, contextual coherence, and overall Parent-Child RAG performance.