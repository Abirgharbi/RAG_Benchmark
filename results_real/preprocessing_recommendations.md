# STM32Cube Preprocessing Analysis

This report analyzes STM32Cube GitHub issues and documents before ingestion into the Parent-Child RAG pipeline.

## Dataset Overview

- Total documents analyzed: **295**
- Missing series metadata: **295**
- Missing component metadata: **138**
- Missing titles: **0**
- Potential duplicated documents: **0**
- Very short documents/issues: **1**
- Very large documents: **2**
- Noisy documents detected: **0**
- Documents containing code: **129**

## Recommended Preprocessing Improvements

- Add MCU series metadata (H7, F4, L4...) to improve filtering and retrieval precision.
- Add component metadata (DMA, UART, USB...) for component-aware retrieval.
- Merge very short issues with parent context to avoid semantic fragmentation.
- Split very large documents semantically before embedding generation.
- Preserve code blocks separately from natural language text.

## Parent-Child Chunking Best Practices

- Preserve hierarchical parent-child relationships.
- Store parent references inside child metadata.
- Keep section titles inside child chunks.
- Avoid splitting code examples across multiple chunks.
- Use semantic chunking instead of fixed-size chunking.
- Normalize repository paths and technical references.
- Use overlap between 10% and 20% to preserve context continuity.
- Separate issue discussions from source-code snippets.
- Preserve issue comments and linked technical references.
- Group STM32 issues by component and MCU series.

## Conclusion

The analysis confirms that preprocessing quality directly impacts embedding coherence, retrieval quality, and contextual reconstruction in Parent-Child RAG systems.