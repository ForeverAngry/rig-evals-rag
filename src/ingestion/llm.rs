//! LLM-backed extractors for Tracks 2 and 3.

use std::future::Future;

use rig::completion::CompletionModel;
use rig::extractor::Extractor;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::error::Result;

use super::graph::{Triple, TripleExtractor};
use super::proposition::{Proposition, PropositionExtractor};
use super::types::Document;

/// Extracted knowledge graph triples.
#[derive(Debug, Clone, Deserialize, serde::Serialize, JsonSchema)]
pub struct ExtractedTriples {
    triples: Vec<ExtractedTriple>,
}

/// A single extracted triple.
#[derive(Debug, Clone, Deserialize, serde::Serialize, JsonSchema)]
pub struct ExtractedTriple {
    subject: String,
    predicate: String,
    object: String,
}

/// Extracted propositions.
#[derive(Debug, Clone, Deserialize, serde::Serialize, JsonSchema)]
pub struct ExtractedPropositions {
    propositions: Vec<String>,
}

/// An LLM-backed extractor for knowledge graph triples.
///
/// Uses `rig::extractor::Extractor` to prompt an LLM to emit structured
/// JSON conforming to [`ExtractedTriples`].
pub struct LlmTripleExtractor<M: CompletionModel> {
    extractor: Extractor<M, ExtractedTriples>,
}

impl<M: CompletionModel> LlmTripleExtractor<M> {
    /// Create a new `LlmTripleExtractor` from a `rig::extractor::Extractor`.
    pub fn new(extractor: Extractor<M, ExtractedTriples>) -> Self {
        Self { extractor }
    }
}

/// An LLM-backed extractor for isolated propositions.
///
/// Uses `rig::extractor::Extractor` to prompt an LLM to emit structured
/// JSON conforming to [`ExtractedPropositions`].
pub struct LlmPropositionExtractor<M: CompletionModel> {
    extractor: Extractor<M, ExtractedPropositions>,
}

impl<M: CompletionModel> LlmPropositionExtractor<M> {
    /// Create a new `LlmPropositionExtractor` from a `rig::extractor::Extractor`.
    pub fn new(extractor: Extractor<M, ExtractedPropositions>) -> Self {
        Self { extractor }
    }
}

impl<M: CompletionModel + Send + Sync> TripleExtractor for LlmTripleExtractor<M> {
    fn extract(&self, doc: &Document) -> impl Future<Output = Result<Vec<Triple>>> + Send {
        let text = doc.text.clone();
        async move {
            let extracted = self
                .extractor
                .extract(&text)
                .await
                .map_err(|e| crate::error::Error::Ingestion(e.to_string()))?;
            Ok(extracted
                .triples
                .into_iter()
                .map(|t| Triple::new(t.subject, t.predicate, t.object))
                .collect())
        }
    }
}

impl<M: CompletionModel + Send + Sync> PropositionExtractor for LlmPropositionExtractor<M> {
    fn extract(&self, doc: &Document) -> impl Future<Output = Result<Vec<Proposition>>> + Send {
        let text = doc.text.clone();
        async move {
            let extracted = self
                .extractor
                .extract(&text)
                .await
                .map_err(|e| crate::error::Error::Ingestion(e.to_string()))?;
            Ok(extracted
                .propositions
                .into_iter()
                .map(Proposition::new)
                .collect())
        }
    }
}
