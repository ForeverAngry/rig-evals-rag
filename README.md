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

`v0.1.x` ships **retrieval-quality** evaluation only:

| Capability | v0.1 | v0.2 | v0.3 |
| --- | :---: | :---: | :---: |
| BEIR-style qrels loader | ✅ | ✅ | ✅ |
| Recall / Precision / MRR / MAP / nDCG / HitRate | ✅ | ✅ | ✅ |
| Async `RetrievalHarness` over any `VectorStoreIndex` | ✅ | ✅ | ✅ |
| JSON / Markdown reports + baseline diff | ✅ | ✅ | ✅ |
| RAGAS-style LLM judges (faithfulness, context_recall, …) | — | ✅ | ✅ |
| Knowledge-gain scoring (per-doc Δ-recall + novelty) | — | — | ✅ |

The crate-local maturity plan lives in [ROADMAP.md](ROADMAP.md). The fuller
phased planning record, including out-of-scope items and reopen triggers, lives
in
[`rig-contributions/docs/evals-rag-plan.md`](https://github.com/ForeverAngry/rig-contributions/blob/main/docs/evals-rag-plan.md).
Cross-crate coordination lives in
[`rig-contributions/docs/roadmap.md`](https://github.com/ForeverAngry/rig-contributions/blob/main/docs/roadmap.md).

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

The diff refuses to compare reports whose `judge_fingerprint` differs — so
swapping an LLM judge never silently moves your score.

## Dataset format

`qrels.jsonl`, one query per line, BEIR-compatible semantics:

```jsonl
{"query_id":"q1","query":"who wrote 1984?","relevant_docs":{"doc-orwell":2,"doc-1984":1}}
{"query_id":"q2","query":"…","relevant_docs":{"doc-7":1},"reference_answer":"…"}
```

Grades are integers in `1..=N`; documents not listed are non-relevant
(grade 0). The optional `reference_answer` field is reserved for the
upcoming answer-level judges in v0.2.

## License

Dual-licensed under either:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
