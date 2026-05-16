# rig-evals-rag Roadmap

This roadmap is the crate-local operating plan for `rig-evals-rag`. The cross-crate coordination summary lives in [`rig-ecosystem/docs/roadmap.md`](../rig-ecosystem/docs/roadmap.md). The fuller planning record for phased evaluation work remains in [`../rig-ecosystem/docs/evals-rag-plan.md`](../rig-ecosystem/docs/evals-rag-plan.md).

## Role

`rig-evals-rag` measures retrieval and knowledge-base quality for any Rig `VectorStoreIndex`. It evaluates stores and RAG context, not general agent behavior or product dashboards.

## Landed

- BEIR-compatible JSONL `Qrels` loader.
- `RetrievalMetric` trait with `RecallAtK`, `PrecisionAtK`, `HitRateAtK`, `Mrr`, `MapAtK`, and `NdcgAtK`.
- Async `RetrievalHarness` over `VectorStoreIndexDyn` with bounded concurrency.
- `MetricReport` and `MultiReport` with JSON, Markdown, aggregation, and baseline diffing.
- Off-by-default `ragas` feature with RAGAS-style judges, `RagasHarness`, bounded-concurrency judge calls, XML-fenced prompts, abstention scores, and judge fingerprint diff protection.
- Off-by-default `ingestion` feature with zero-waste ingestion deltas, deterministic IoC extraction, proposition extraction, redundancy checks, and model-independent contract tests for LLM-backed proposition extractors.
- Off-by-default `ingestion-graph` sub-feature with knowledge-graph triples, in-memory and `petgraph` baselines, and model-independent contract tests for LLM-backed triple extractors.
- Repeated-trial reliability reporting with thresholded pass@k and pass^k over
  shared `MetricReport`s.

## Prototype Grade

- Retrieval metrics are usable; RAGAS is merged but release pending.
- Zero-waste ingestion tracks are merged and covered by deterministic tests; chunk-stat linting, language/encoding linting, and MinHash-style near-duplicate detection remain planned.
- There is no committed `eval_memvid.rs` integration example yet.
- Knowledge-gain scoring and shadow-store contracts remain planned.
- Bootstrap confidence intervals and non-zero CI regression exits are not implemented.

## Next Work

1. Cut the RAGAS and ingestion release after final validation and README/changelog sync.
2. Add chunk-stat ingestion linting for token-length distributions, metadata coverage, language/encoding sanity, and MinHash near-duplicates.
3. Add `examples/eval_memvid.rs` wiring `rig-memvid::MemvidStore` through `RetrievalHarness` with committed fixture data.
4. Define `EvalShadowStore` for pre/post-ingest scoring and implement the memvid shadow path.
5. Add knowledge-gain scoring: pre/post qrels delta plus embedding novelty.
6. Add bootstrap confidence intervals and non-zero regression exits for baseline comparisons.

## Maturity Bar

- A KB regression can be reproduced in CI against committed qrels fixtures.
- Retrieval and RAGAS reports include stable metadata and refuse unsafe comparisons across judge fingerprints.
- Ingestion quality failures identify chunking, duplication, metadata, retrieval, or judging as the likely layer.
- Knowledge gain produces a single ranked score per candidate document.

## Non-Goals

- Do not become a trace-level agent evaluation harness until the platform runner exists.
- Do not own product dashboards, online shadow scoring, tenancy slicing, or safety/adversarial suites.
- Do not route models; measure quality and report signals for other layers to consume.
