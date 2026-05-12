//! # rig-evals-rag
//!
//! Retrieval and knowledge-base evaluation harness for
//! [Rig](https://crates.io/crates/rig-core) agents.
//!
//! The crate gives you:
//!
//! - A BEIR-compatible [`dataset::Qrels`] loader (JSONL).
//! - A pure-Rust catalogue of standard IR metrics (Recall, Precision, MRR,
//!   MAP, nDCG, HitRate) in [`retrieval`].
//! - An async [`harness::RetrievalHarness`] that drives any store
//!   implementing [`rig::vector_store::VectorStoreIndexDyn`].
//! - JSON / Markdown [`report::MultiReport`]s with baseline diffing.
//!
//! See the crate README for an end-to-end quickstart.
//!
//! ## Stability
//!
//! v0.1.x ships retrieval-quality evaluation only. The `ragas` feature adds
//! LLM-based RAGAS-style judges (`FaithfulnessMetric`, `ContextPrecisionMetric`,
//! `ContextRecallMetric`, `AnswerRelevanceMetric`) driven by
//! [`ragas::RagasHarness`]. Knowledge-gain metrics remain planned for v0.3.

#![deny(missing_docs)]
#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod dataset;
pub mod error;
pub mod harness;
#[cfg(feature = "ragas")]
pub mod ragas;
pub mod report;
pub mod retrieval;

pub use dataset::{GoldQuery, Qrels, RetrievedDoc, RetrievedSet};
pub use error::{Error, Result};
pub use harness::RetrievalHarness;
pub use report::{MetricDelta, MetricReport, MultiReport, ReportDiff};
pub use retrieval::{HitRateAtK, MapAtK, Mrr, NdcgAtK, PrecisionAtK, RecallAtK, RetrievalMetric};
