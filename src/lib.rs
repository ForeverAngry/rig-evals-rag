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
//! The default build ships retrieval-quality evaluation only. Optional features
//! add RAGAS-style judges, zero-waste ingestion checks, knowledge-gain scoring,
//! and optional embedding novelty adapters.

#![deny(missing_docs)]
#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod dataset;
pub mod error;
pub mod harness;
#[cfg(feature = "knowledge-gain")]
pub mod knowledge_gain;
#[cfg(feature = "ragas")]
pub mod ragas;
pub mod report;
pub mod retrieval;
#[cfg(feature = "shadow")]
pub mod shadow;

pub use dataset::{GoldQuery, Qrels, RetrievedDoc, RetrievedSet};
pub use error::{Error, Result};
pub use harness::RetrievalHarness;
#[cfg(feature = "knowledge-gain")]
pub use knowledge_gain::{
    CandidateDocumentGain, CandidateDocumentGainInput, CandidateQueryGain, KnowledgeGainConfig,
    KnowledgeGainReport, MetricGain, QueryGain,
};
#[cfg(feature = "embedding-novelty")]
pub use knowledge_gain::{CandidateNoveltyInput, EmbeddingNoveltyAdapter};
pub use report::{
    MetricDelta, MetricReport, MultiReport, QueryDelta, QueryReliability, RegressionGate,
    ReliabilityReport, ReportDiff,
};
pub use retrieval::{HitRateAtK, MapAtK, Mrr, NdcgAtK, PrecisionAtK, RecallAtK, RetrievalMetric};
#[cfg(feature = "shadow")]
pub use shadow::{EvalShadowStore, ShadowEvalReport};

#[cfg(feature = "ingestion")]
pub mod ingestion;

#[cfg(feature = "skills")]
pub mod skills;

#[cfg(feature = "ingestion-graph")]
pub use ingestion::PetgraphBaseline;

#[cfg(feature = "ingestion")]
pub use ingestion::{
    ActiveGraphTrack, ActivePropositionTrack, Chunk, ChunkLintConfig, ChunkLintReport,
    ChunkLintWarning, ChunkStats, DistillationPipeline, Document, Dropped, DroppedItem,
    DroppedReason, GraphBaseline, GraphTrack, InMemoryGraphBaseline, InMemoryIocBaseline,
    IngestionDelta, IngestionReport, Ioc, IocBaseline, IocExtractor, IocKind,
    LlmPropositionExtractor, LlmTripleExtractor, NoGraphTrack, NoPropositionTrack, Proposition,
    PropositionExtractor, PropositionTrack, RedundancyCheck, RedundancyVerdict, RegexIocExtractor,
    Section, SectionKind, StubPropositionExtractor, StubTripleExtractor, Triple, TripleExtractor,
    VectorStoreRedundancyCheck, lint_chunks, lint_chunks_strict,
};
