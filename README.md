# rig-evals-rag

[![Crates.io](https://img.shields.io/crates/v/rig-evals-rag.svg)](https://crates.io/crates/rig-evals-rag)
[![Docs.rs](https://docs.rs/rig-evals-rag/badge.svg)](https://docs.rs/rig-evals-rag)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

> Retrieval and knowledge-base evaluation harness for
> [Rig](https://crates.io/crates/rig-core) agents.

`rig-evals-rag` measures **knowledge-base quality**, not just answer quality.
Point it at any `VectorStoreIndex` (`rig`'s in-memory store, `rig-memvid`,
`rig-lancedb`, …), give it a labeled qrels file, and get a report you can
diff between runs to catch regressions before they ship.

## Status

The default build ships **retrieval-quality** evaluation only. The current
unreleased branch also contains optional RAGAS judges and zero-waste ingestion
tracks behind feature flags; knowledge-gain scoring remains planned.

| Capability | Default | Feature | Validation |
| --- | :---: | --- | --- |
| BEIR-style qrels loader | ✅ | `retrieval` | Unit + harness integration tests |
| Recall / Precision / MRR / MAP / nDCG / HitRate | ✅ | `retrieval` | Metric unit tests |
| Async `RetrievalHarness` over any `VectorStoreIndexDyn` | ✅ | `retrieval` | `tests/harness.rs` |
| JSON / Markdown reports + baseline diff | ✅ | `retrieval` | Report unit tests + harness test |
| RAGAS-style LLM judges (faithfulness, context recall, …) | — | `ragas` | Unit tests with deterministic judge fixtures |
| Zero-waste IoC ingestion | — | `ingestion` | `tests/ingestion_ioc.rs` |
| Proposition distillation + redundancy checks | — | `ingestion` | `tests/ingestion_propositions.rs` |
| Knowledge-graph triples + graph baseline | — | `ingestion-graph` | `tests/ingestion_graph.rs` |
| LLM-backed ingestion extractors | — | `ingestion` | Model-independent fake-provider contract tests + optional live Ollama smoke |
| Knowledge-gain scoring (per-doc Δ-recall + novelty) | — | planned | Not implemented |

The crate-local maturity plan lives in [ROADMAP.md](ROADMAP.md). The fuller
phased planning record, including out-of-scope items and reopen triggers, lives
in
[`rig-contributions/docs/evals-rag-plan.md`](https://github.com/ForeverAngry/rig-contributions/blob/main/docs/evals-rag-plan.md).
Cross-crate coordination lives in
[`rig-contributions/docs/roadmap.md`](https://github.com/ForeverAngry/rig-contributions/blob/main/docs/roadmap.md).

## Feature flags

| Feature | Default | Enables |
| --- | --- | --- |
| `retrieval` | yes | Pure-Rust retrieval metrics, qrels loading, harness, reports, and diffs. |
| `ragas` | no | LLM-backed RAGAS-style judges and `RagasHarness`. |
| `ingestion` | no | Zero-waste ingestion Track 1 (IoCs), Track 3 (propositions), and LLM extractor adapters. |
| `ingestion-graph` | no | Track 2 knowledge-graph triples plus `petgraph`-backed baseline. Implies `ingestion`. |

## Quickstart

```rust,no_run
use anyhow::Result;
use rig::vector_store::VectorStoreIndexDyn;
use rig_evals_rag::{
    NdcgAtK, Qrels, RecallAtK, RetrievalHarness, RetrievalMetric,
};

# async fn run(store: impl VectorStoreIndexDyn + 'static) -> Result<()> {
let qrels = Qrels::load_jsonl("tests/data/tiny_qrels.jsonl")?;

let metrics: Vec<Box<dyn RetrievalMetric>> = vec![
    Box::new(RecallAtK::new(10)),
    Box::new(NdcgAtK::new(10)),
];

let report = RetrievalHarness::new(&store, 10)
    .with_concurrency(4)
    .run(&qrels, &metrics)
    .await?;

println!("{}", report.to_markdown());
std::fs::write("report.json", report.to_json()?)?;
# Ok(()) }
```

### Diffing against a baseline

```rust,no_run
# use rig_evals_rag::MultiReport;
# fn read(p: &str) -> anyhow::Result<MultiReport> { unimplemented!() }
# fn demo() -> anyhow::Result<()> {
let current  = read("report.json")?;
let baseline = read("baseline.json")?;
let diff = current.diff(&baseline)?;
println!("{}", diff.to_markdown());
# Ok(()) }
```

The diff refuses to compare reports whose `judge_fingerprint` differs, so
swapping an LLM judge never silently moves your score.

## Optional ingestion checks

The ingestion feature family moves quality control upstream of vector-store
commit. Instead of storing every chunk and hoping retrieval compensates later,
the pipeline emits an `IngestionDelta` containing net-new IoCs, propositions,
and graph triples plus structured drop reasons for duplicates or redundant
facts.

Deterministic extractors and baselines are the CI path. `LlmTripleExtractor`
and `LlmPropositionExtractor` adapt Rig's structured `Extractor` for hosts that
want model-backed extraction; their contract tests use a fake `CompletionModel`
so validation does not depend on a specific provider or local model. The ignored
`live_ollama_ingestion` test remains a manual smoke test for tool-capable local
models.

## Dataset format

`qrels.jsonl`, one query per line, BEIR-compatible semantics:

```jsonl
{"query_id":"q1","query":"who wrote 1984?","relevant_docs":{"doc-orwell":2,"doc-1984":1}}
{"query_id":"q2","query":"…","relevant_docs":{"doc-7":1},"reference_answer":"…"}
```

Grades are integers in `1..=N`; documents not listed are non-relevant
(grade 0). The optional `reference_answer` field is used by answer-level judges
when the `ragas` feature is enabled.

## License

Dual-licensed under either:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
