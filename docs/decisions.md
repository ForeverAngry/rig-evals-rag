# Architectural Decisions

This document captures the rationale, formulas, and references for the metric definitions and evaluation strategy implemented in `rig-evals-rag`.

## Supported Metrics (v0.1)

Traditional Information Retrieval (IR) metrics based on graded relevance (or binary relevance). We follow the standard formulas from TREC `trec_eval` and BEIR.

### Recall @ K
Fractions of relevant documents retrieved in the top K.
- **Formula**: `|retrieved[1..K] ∩ relevant| / |relevant|`
- **Notes**: Vacuously `1.0` if `|relevant| == 0`. Best for evaluating if the knowledge base contains the answer at all.

### Precision @ K
Fraction of the top K retrieved documents that are relevant.
- **Formula**: `|retrieved[1..K] ∩ relevant| / K`
- **Notes**: Returns `0.0` if `K == 0`. Useful for understanding how much "noise" is in the context window.

### Hit Rate @ K
Binary check if *any* relevant document appears in the top K.
- **Formula**: `1.0` if `|retrieved[1..K] ∩ relevant| > 0` else `0.0`.
- **Notes**: Useful minimum-viability metric for single-hop QA (if you only need one correct chunk to answer).

### MRR (Mean Reciprocal Rank)
Reciprocal of the rank of the *first* relevant document.
- **Formula**: `1.0 / rank` of the first relevant hit, or `0.0` if none.
- **Notes**: Emphasizes finding the answer as early as possible.

### MAP @ K (Mean Average Precision)
Approximation of the area under the Precision-Recall curve up to rank K.
- **Formula**: `(1 / |relevant|) * Σ_{i=1..k} (Precision@i * rel(i))` (where `rel(i)` is 1 if relevant, else 0).
- **Notes**: Highly sensitive to the rank of *all* relevant documents.

### nDCG @ K (Normalized Discounted Cumulative Gain)
Measures ranking quality, penalizing relevant documents that appear lower in the list, while gracefully handling different *grades* of relevance.
- **Formula**: `DCG@K / IDCG@K`
- **DCG@K**: `Σ_{i=1..k} ( (2^grade_i - 1) / log2(i + 1) )`
- **IDCG@K**: The DCG of the ideal (perfectly sorted) ranking of the available graded labels.
- **Notes**: BEIR/TREC standard for graded relevance.

## Planned Metrics (v0.2+)

### RAGAS-style LLM Judges
Based on the [RAGAS framework](https://arxiv.org/abs/2309.15217), we plan to introduce:
- **Faithfulness**: Are the claims generated in the answer attributable to the retrieved context?
- **Context Precision**: Is the retrieved context actually useful?
- **Answer Relevance**: Does the generated answer directly address the user's query?

### Knowledge Gain & Novelty (v0.3)
To measure whether ingesting a *new* document actually improves the knowledge base:
- **Gain(d, K)**: `α * ΔRecall@k + β * ΔContextRecall + γ * novelty(d, K)`
- **Novelty(d, K)**: `1 - max(cosine(embed(chunks(d)), embed(chunks(K))))`

## References

1. **BEIR**: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models [Thakur et al., 2021]
2. **RAGAS**: Automated Evaluation of Retrieval Augmented Generation [Es et al., 2023]
3. **MTEB**: Massive Text Embedding Benchmark [Muennighoff et al., 2022]
4. **TREC eval**: Standard IR metric computation (https://github.com/usnistgov/trec_eval)
