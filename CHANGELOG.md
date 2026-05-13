# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to `rig-evals-rag` are documented here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this crate
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ReportDiff` now carries per-query winners/losers/unchanged counts and
  a sorted `query_changes: Vec<QueryDelta>` listing the largest movers per
  metric, computed by intersecting the two reports' `per_query` vectors on
  `query_id`. Queries missing from either side are skipped. The new
  `RegressionGate` lets callers configure per-metric tolerated drops in
  mean score; `ReportDiff::regressions(&gate)` returns the metrics whose
  delta breaches the gate. `ReportDiff::to_json` complements the existing
  `to_markdown`, and the Markdown table grew `win`/`lose`/`same` columns.
  Backward compatible: new `MetricDelta` fields use `#[serde(default)]`.
- `tests/ingestion_ab.rs` — fixture-based ingestion A/B test that runs
  the harness against two `MockStore` configurations (improvement vs
  regression) and asserts the diff/gate verdicts. No LLM, no network.
- `LlmTripleExtractor` and `LlmPropositionExtractor` wrappers enabling live extraction via `rig-core`.
- Deterministic `ingestion_llm.rs` contract tests validating LLM adapters with a fake `rig::CompletionModel`, independent of model/vendor behavior.
- `live_ollama_ingestion.rs` smoke test for manual validation against a local tool-capable Ollama model.
- Track 2 (knowledge-graph distillation) on the `ingestion` feature:
  `Triple` (with normalised predicate — lowercased, whitespace collapsed
  to underscores), `TripleExtractor` + `GraphBaseline` traits, the
  deterministic `StubTripleExtractor`, and `InMemoryGraphBaseline`
  (`HashSet`-backed; always available). `DistillationPipeline::with_graph`
  layers the track via a second type-state axis (`NoGraphTrack` /
  `ActiveGraphTrack`) so unconfigured pipelines pay no runtime cost and
  Tracks 2 and 3 compose independently. `IngestionDelta` grew a
  `triples: Vec<Triple>` field, `DroppedItem::Triple(Triple)` and
  `DroppedReason::DuplicateEdge` were added (all `non_exhaustive`,
  additive).
- `ingestion-graph` sub-feature (depends on `ingestion`): pulls
  [`petgraph`](https://docs.rs/petgraph) and ships `PetgraphBaseline`,
  a `Graph<String, String>`-backed `GraphBaseline` for hosts that want
  richer downstream queries (path / predicate adjacency) against the same
  store the pipeline uses for dedup.
- Track 3 (propositional distillation) on the `ingestion` feature:
  `Proposition`, `PropositionExtractor` + `RedundancyCheck` traits, the
  deterministic `StubPropositionExtractor` (sentence splitter), and
  `VectorStoreRedundancyCheck` which drives any `VectorStoreIndexDyn`.
  `DistillationPipeline::with_propositions` layers the track via a
  type-state (`NoPropositionTrack` / `ActivePropositionTrack`) so
  unconfigured pipelines pay no runtime cost. `IngestionDelta` grew a
  `propositions: Vec<Proposition>` field and `DroppedReason::Redundant
  { similarity }` was added (both `non_exhaustive`, additive).
- `ingestion` feature (off by default): zero-waste ingestion pipeline that
  emits structured `IngestionDelta`s (net-new items + dropped items with
  reasons) instead of committing chunks. Ships Track 1 (IoC filter):
  `Document` / `Section`, `IocExtractor` + `IocBaseline` traits, the
  deterministic `RegexIocExtractor` (CVE / IPv4 / IPv6 / MD5 / SHA-1 /
  SHA-256 / domain / URL / Windows registry key) and an
  `InMemoryIocBaseline`. `DistillationPipeline` orchestrates the track and
  is generic over extractor/baseline so callers can swap implementations
  without trait objects. Tracks 2 (knowledge graph) and 3 (propositions)
  layer onto the same orchestrator in follow-up PRs.
- Add crate-local `ROADMAP.md` documenting maturity status, next work, and
  non-goals for retrieval and RAG evaluation.
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
