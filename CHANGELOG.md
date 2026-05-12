# Changelog

All notable changes to `rig-evals-rag` are documented here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this crate
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ragas` feature (off by default): LLM-based RAGAS-style judges
  (`FaithfulnessMetric`, `ContextPrecisionMetric`, `ContextRecallMetric`,
  `AnswerRelevanceMetric`) wired through a single `RagasMetric` trait and an
  object-safe `DynRagasMetric` shim.
- `RagasInputs` / `RagasScore` envelopes; `RagasScore::not_measurable`
  lets a judge abstain instead of fabricating a score.
- `RagasHarness` async driver with `futures::buffered` concurrency. The
  harness composes every judge's `fingerprint_component` into the
  `MultiReport::judge_fingerprint` so diffs across judge revisions refuse
  to compare.
- Typed `Error::Extraction` (`#[from] rig::extractor::ExtractionError`) and
  `Error::Embedding` (`#[from] rig::embeddings::EmbeddingError`) variants
  under the `ragas` feature, replacing the prior `Error::Llm(String)`.

### Changed

- Per-claim and per-chunk LLM calls inside every RAGAS judge now run with
  bounded concurrency via `futures::stream::buffered`.
- All RAGAS prompts now fence their inputs in XML-style tags and instruct
  the judge to treat fenced contents as data — narrower prompt-injection
  surface.

## [0.1.0] - TBD

### Added

- BEIR-compatible JSONL qrels loader (`Qrels::load_jsonl`).
- `RetrievalMetric` trait + concrete implementations: `RecallAtK`,
  `PrecisionAtK`, `HitRateAtK`, `Mrr`, `MapAtK`, `NdcgAtK`.
- Async `RetrievalHarness` over any `rig::vector_store::VectorStoreIndexDyn`,
  with bounded concurrency and per-query tracing spans.
- `MetricReport` / `MultiReport` with mean, stddev, P50/P95, min/max.
- JSON + Markdown report serialization and `MultiReport::diff` with
  `judge_fingerprint` mismatch detection.

[Unreleased]: https://github.com/ForeverAngry/rig-evals-rag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ForeverAngry/rig-evals-rag/releases/tag/v0.1.0
