//! Track 3 — propositional distillation with cosine-novelty redundancy check.
//!
//! A *proposition* is an atomic factual claim extracted from a document
//! ("APT-28 is attributed to GRU Unit 26165"). Track 3 commits only those
//! propositions whose nearest neighbour in a vector store falls *below* a
//! similarity threshold — everything else is redundant and would only
//! crowd retrieval results.
//!
//! The library ships a deterministic `StubPropositionExtractor` (sentence
//! splitter) so hosts have a model-free CI gate. Production hosts plug in
//! an LLM-backed extractor through the [`PropositionExtractor`] trait.

use std::future::Future;

use rig::vector_store::{VectorSearchRequest, VectorStoreIndexDyn, request::Filter};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::types::Document;

/// An atomic factual claim extracted from a [`Document`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Proposition {
    /// Optional caller- or extractor-assigned identifier. Echoed back in
    /// the [`IngestionDelta`](super::IngestionDelta) for downstream
    /// commits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The proposition's text. Used verbatim as the query to the
    /// redundancy [`VectorStoreIndexDyn`].
    pub text: String,
}

impl Proposition {
    /// Build a proposition from its text with no explicit id.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: None,
            text: text.into(),
        }
    }

    /// Build a proposition with an explicit id.
    pub fn with_id(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
            text: text.into(),
        }
    }
}

/// Verdict from a [`RedundancyCheck`] lookup.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RedundancyVerdict {
    /// `true` if the proposition should be dropped as redundant against
    /// the existing corpus.
    pub is_redundant: bool,
    /// Similarity of the nearest neighbour (typically cosine in `[0, 1]`).
    /// `0.0` when the store returned no hits.
    pub similarity: f64,
}

/// Extracts atomic propositions from a [`Document`].
///
/// Implementations are async so the production impl can call an LLM via
/// `rig-core`'s `Extractor`. The library-shipped
/// [`StubPropositionExtractor`] resolves synchronously inside an `async`
/// block so it remains deterministic for CI.
pub trait PropositionExtractor: Send + Sync {
    /// Return every proposition the implementation produced for `doc`.
    fn extract(&self, doc: &Document) -> impl Future<Output = Result<Vec<Proposition>>> + Send;
}

/// Caller-owned oracle that answers "has something semantically equivalent
/// to this proposition already been committed".
///
/// The default impl ([`VectorStoreRedundancyCheck`]) wraps any
/// [`VectorStoreIndexDyn`] — the same store the host will eventually
/// commit propositions to. Hosts with their own dedup logic (e.g. a
/// LSH index, a propositional KB, a stricter cross-encoder) implement
/// this trait directly.
pub trait RedundancyCheck: Send + Sync {
    /// Look up the proposition's most similar neighbour and return a
    /// verdict.
    fn check(
        &self,
        proposition: &Proposition,
    ) -> impl Future<Output = Result<RedundancyVerdict>> + Send;
}

/// Deterministic sentence-splitter [`PropositionExtractor`].
///
/// Splits `doc.text` and every section's text on `.`, `!`, and `?`
/// (followed by whitespace or end-of-string), trims, drops empties, and
/// returns one proposition per non-empty sentence. Useful as a CI gate
/// for hosts who don't yet wire up an LLM extractor.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubPropositionExtractor;

impl StubPropositionExtractor {
    /// Construct the splitter. Equivalent to `StubPropositionExtractor`.
    pub fn new() -> Self {
        Self
    }

    fn split(&self, text: &str, out: &mut Vec<Proposition>) {
        let mut buf = String::new();
        for ch in text.chars() {
            buf.push(ch);
            if matches!(ch, '.' | '!' | '?') {
                let trimmed = buf.trim();
                if trimmed.len() > 1 {
                    out.push(Proposition::new(trimmed));
                }
                buf.clear();
            }
        }
        let trimmed = buf.trim();
        if !trimmed.is_empty() {
            out.push(Proposition::new(trimmed));
        }
    }
}

impl PropositionExtractor for StubPropositionExtractor {
    fn extract(&self, doc: &Document) -> impl Future<Output = Result<Vec<Proposition>>> + Send {
        let mut out = Vec::new();
        self.split(&doc.text, &mut out);
        for section in &doc.sections {
            self.split(&section.text, &mut out);
        }
        async move { Ok(out) }
    }
}

/// [`RedundancyCheck`] backed by any [`VectorStoreIndexDyn`].
///
/// Runs `top_n_ids(proposition.text, samples)` against the store and
/// compares the top hit's score against `threshold`. Stores that return
/// cosine similarity in `[0, 1]` (the Rig convention) work out of the
/// box. Hosts whose stores return a different similarity scale should
/// implement [`RedundancyCheck`] directly.
///
/// The check borrows the store immutably; pair it with a `Arc<S>` or
/// keep the store alive at least as long as the pipeline.
pub struct VectorStoreRedundancyCheck<'s> {
    store: &'s dyn VectorStoreIndexDyn,
    threshold: f64,
    samples: u64,
}

impl<'s> VectorStoreRedundancyCheck<'s> {
    /// Default similarity threshold (matches the architecture decision in
    /// `docs/decisions.md`).
    pub const DEFAULT_THRESHOLD: f64 = 0.90;

    /// Build a redundancy check against `store` with the configured
    /// `threshold`.
    ///
    /// Returns [`Error::Config`] if `threshold` is outside `[0.0, 1.0]`.
    pub fn new(store: &'s dyn VectorStoreIndexDyn, threshold: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::Config(format!(
                "RedundancyCheck threshold must be in [0.0, 1.0], got {threshold}"
            )));
        }
        Ok(Self {
            store,
            threshold,
            samples: 1,
        })
    }

    /// Build a redundancy check with [`Self::DEFAULT_THRESHOLD`].
    pub fn with_default_threshold(store: &'s dyn VectorStoreIndexDyn) -> Self {
        Self {
            store,
            threshold: Self::DEFAULT_THRESHOLD,
            samples: 1,
        }
    }

    /// Override the number of neighbours fetched per check. Defaults to 1.
    /// Larger values do not change the verdict (only the top hit matters)
    /// but can be useful when a host wants to log the runner-up.
    pub fn with_samples(mut self, samples: u64) -> Self {
        self.samples = samples.max(1);
        self
    }

    /// The configured threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl<'s> RedundancyCheck for VectorStoreRedundancyCheck<'s> {
    fn check(
        &self,
        proposition: &Proposition,
    ) -> impl Future<Output = Result<RedundancyVerdict>> + Send {
        let req: VectorSearchRequest<Filter<serde_json::Value>> = VectorSearchRequest::builder()
            .query(proposition.text.clone())
            .samples(self.samples)
            .build();
        let threshold = self.threshold;
        async move {
            let hits = self.store.top_n_ids(req).await?;
            let similarity = hits.first().map(|(score, _)| *score).unwrap_or(0.0);
            Ok(RedundancyVerdict {
                is_redundant: similarity >= threshold,
                similarity,
            })
        }
    }
}
