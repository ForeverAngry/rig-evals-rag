# Changelog

All notable changes to `rig-evals-rag` are documented here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this crate
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
