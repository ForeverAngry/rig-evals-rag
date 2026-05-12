//! Error types for `rig-evals-rag`.

use rig::vector_store::VectorStoreError;

/// Errors produced by `rig-evals-rag`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Failed to parse a qrels / corpus / answers JSONL file.
    #[error("dataset parse error at line {line}: {source}")]
    DatasetParse {
        /// 1-indexed source line that failed to parse.
        line: usize,
        /// The underlying JSON error.
        #[source]
        source: serde_json::Error,
    },

    /// I/O error while reading or writing a dataset or report.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialization error outside dataset parsing (e.g. report
    /// writing or qrels round-trip).
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// The underlying [`rig::vector_store::VectorStoreIndex`] returned an
    /// error during a retrieval run.
    #[error("vector store error: {0}")]
    Store(#[from] VectorStoreError),

    /// The configured top-k or sample count was invalid (e.g. zero).
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A metric requested a comparison against a baseline whose schema does
    /// not match the current report (different judge fingerprint, different
    /// metric set, etc.).
    #[error("baseline mismatch: {0}")]
    BaselineMismatch(String),
}

/// Convenience alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
