//! RAGAS-style LLM-based evaluation metrics.
//!
//! Requires the `ragas` feature. The four shipped judges are wired through a
//! single [`RagasMetric`] trait so that a heterogeneous set of judges can be
//! driven by [`RagasHarness::run`] and aggregated into a
//! [`crate::report::MultiReport`] alongside the pure-Rust retrieval metrics.
//!
//! Per-claim work inside each judge is fanned out through
//! `futures::stream::iter(...).buffered(concurrency)` so a single `score()`
//! call respects the configured rate budget.
//!
//! ## Judge fingerprint
//!
//! Every judge contributes to a [`MultiReport::judge_fingerprint`][fp] via
//! [`RagasMetric::fingerprint_component`]. Reports produced with different
//! judge fingerprints refuse to diff (silent comparison drift defense).
//!
//! [fp]: crate::report::MultiReport::judge_fingerprint

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Answer Relevance metric.
pub mod answer_relevance;
/// Context Precision metric.
pub mod context_precision;
/// Context Recall metric.
pub mod context_recall;
/// Faithfulness metric.
pub mod faithfulness;
/// Async driver for [`RagasMetric`] over a set of [`RagasInputs`].
pub mod harness;

pub use answer_relevance::{AnswerRelevanceMetric, HypotheticalQuestions};
pub use context_precision::{ContextPrecisionMetric, ContextRelevance};
pub use context_recall::ContextRecallMetric;
pub use faithfulness::{Claim, ClaimAttribution, Claims, FaithfulnessMetric};
pub use harness::RagasHarness;

/// Inputs evaluated by a [`RagasMetric`] for a single query.
///
/// Every judge ignores the fields it doesn't need (e.g. `AnswerRelevance`
/// ignores `context`). A judge that requires a field which is absent emits
/// [`RagasScore::not_measurable`] rather than fabricating a score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagasInputs {
    /// Stable identifier matching the gold query id.
    pub query_id: String,
    /// The original user query.
    pub query: String,
    /// The generated answer under test. Optional so query-only judges remain
    /// runnable on partially-populated datasets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    /// Retrieved context chunks in rank order (best first).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<String>,
    /// Gold / reference answer used by recall-style judges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_answer: Option<String>,
}

impl RagasInputs {
    /// Convenience constructor for a fully-specified sample.
    #[must_use]
    pub fn new(
        query_id: impl Into<String>,
        query: impl Into<String>,
        answer: impl Into<String>,
        context: Vec<String>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            query: query.into(),
            answer: Some(answer.into()),
            context,
            reference_answer: None,
        }
    }

    /// Attach a reference / gold answer (required by context-recall).
    #[must_use]
    pub fn with_reference(mut self, reference: impl Into<String>) -> Self {
        self.reference_answer = Some(reference.into());
        self
    }
}

/// Outcome of scoring a single [`RagasInputs`] sample.
///
/// `value == None` denotes "not measurable on this sample" — e.g. faithfulness
/// on an empty answer. The harness skips those rather than averaging them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagasScore {
    /// Aggregate scalar score in `[0.0, 1.0]`, or `None` when the metric
    /// abstained.
    pub value: Option<f64>,
    /// Free-form rationale lines preserved in the JSON report for audit.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rationales: Vec<String>,
}

impl RagasScore {
    /// Measured score.
    #[must_use]
    pub fn measured(value: f64) -> Self {
        Self {
            value: Some(value),
            rationales: Vec::new(),
        }
    }

    /// Measured score with attached rationale lines.
    #[must_use]
    pub fn with_rationales(value: f64, rationales: Vec<String>) -> Self {
        Self {
            value: Some(value),
            rationales,
        }
    }

    /// Sample-level abstention. Aggregation skips this entry.
    #[must_use]
    pub fn not_measurable(reason: impl Into<String>) -> Self {
        Self {
            value: None,
            rationales: vec![reason.into()],
        }
    }
}

/// An async LLM-based judge that scores a single [`RagasInputs`] sample.
///
/// Implementations must be `Send + Sync` so the harness can drive them with
/// bounded concurrency. The trait is intentionally **not** object-safe;
/// [`DynRagasMetric`] is the object-safe shim used by the harness.
pub trait RagasMetric: Send + Sync {
    /// Human-readable identifier (e.g. `"faithfulness"`).
    fn name(&self) -> &'static str;

    /// Per-judge component contributed to the report fingerprint. Should be
    /// stable for the lifetime of a given model+prompt combination.
    fn fingerprint_component(&self) -> String;

    /// Score a single sample.
    fn score(&self, inputs: &RagasInputs) -> impl Future<Output = Result<RagasScore>> + Send;
}

/// Object-safe sibling of [`RagasMetric`], used by [`RagasHarness`].
pub trait DynRagasMetric: Send + Sync {
    /// See [`RagasMetric::name`].
    fn name(&self) -> &'static str;
    /// See [`RagasMetric::fingerprint_component`].
    fn fingerprint_component(&self) -> String;
    /// See [`RagasMetric::score`].
    fn score<'a>(
        &'a self,
        inputs: &'a RagasInputs,
    ) -> Pin<Box<dyn Future<Output = Result<RagasScore>> + Send + 'a>>;
}

impl<M: RagasMetric> DynRagasMetric for M {
    fn name(&self) -> &'static str {
        RagasMetric::name(self)
    }
    fn fingerprint_component(&self) -> String {
        RagasMetric::fingerprint_component(self)
    }
    fn score<'a>(
        &'a self,
        inputs: &'a RagasInputs,
    ) -> Pin<Box<dyn Future<Output = Result<RagasScore>> + Send + 'a>> {
        Box::pin(RagasMetric::score(self, inputs))
    }
}

/// Cosine similarity between two same-length `f64` vectors.
///
/// Returns `0.0` for empty / mismatched / zero-norm inputs so callers can
/// treat the result as a strict similarity in `[-1.0, 1.0]` without
/// branching on edge cases.
#[must_use]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_identical_is_one() {
        assert!((cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_mismatched_length_is_zero() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn cosine_zero_vector_is_zero() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn not_measurable_carries_reason() {
        let s = RagasScore::not_measurable("empty answer");
        assert!(s.value.is_none());
        assert_eq!(s.rationales, vec!["empty answer".to_string()]);
    }
}
