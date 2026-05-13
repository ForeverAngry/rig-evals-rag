//! Track 2 — knowledge-graph distillation.
//!
//! This module ships the [`Triple`] payload, the [`TripleExtractor`] and
//! [`GraphBaseline`] traits, and two built-in baselines:
//!
//! * [`InMemoryGraphBaseline`] — a `HashSet`-backed oracle that ships in
//!   the default `ingestion` feature; sufficient for tests and for hosts
//!   that only need exact-edge deduplication.
//! * [`PetgraphBaseline`] — a [`petgraph`](https://docs.rs/petgraph)-backed
//!   oracle gated on the `ingestion-graph` sub-feature. Hosts who already
//!   use petgraph for downstream queries (path/predicate adjacency) can
//!   wire the same store through both layers.
//!
//! All triples are normalised on construction: `predicate` is lowercased
//! with whitespace collapsed to underscores so `"Affects"` and
//! `"affects"` collide. `subject` and `object` are trimmed and stored
//! verbatim — case sensitivity there is load-bearing for proper-noun
//! entities.

use std::collections::HashSet;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::types::Document;

/// A `(subject, predicate, object)` statement extracted from a document.
///
/// Use [`Triple::new`] so the predicate is normalised. Manual struct
/// construction is intentionally avoided via `#[non_exhaustive]` so
/// invariants stay enforceable as the schema evolves.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Triple {
    /// Entity the statement is about.
    pub subject: String,
    /// Relation, normalised: lower-case ASCII with runs of whitespace
    /// folded to a single underscore.
    pub predicate: String,
    /// Entity or literal the subject is related to.
    pub object: String,
}

impl Triple {
    /// Construct a triple, normalising `predicate` and trimming surrounding
    /// whitespace from each field.
    pub fn new<S, P, O>(subject: S, predicate: P, object: O) -> Self
    where
        S: Into<String>,
        P: Into<String>,
        O: Into<String>,
    {
        let predicate = normalise_predicate(&predicate.into());
        Self {
            subject: subject.into().trim().to_owned(),
            predicate,
            object: object.into().trim().to_owned(),
        }
    }
}

fn normalise_predicate(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_underscore = false;
    for ch in raw.trim().chars() {
        if ch.is_whitespace() {
            if !prev_underscore && !out.is_empty() {
                out.push('_');
                prev_underscore = true;
            }
        } else {
            out.extend(ch.to_lowercase());
            prev_underscore = false;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

/// Caller-supplied extractor that proposes [`Triple`]s for a document.
///
/// Library implementations are deterministic. Hosts that wire an LLM
/// extractor must produce normalised triples (use [`Triple::new`]) so
/// dedup against the baseline is consistent.
pub trait TripleExtractor: Send + Sync {
    /// Return every candidate triple the implementation produced for `doc`.
    fn extract(
        &self,
        doc: &Document,
    ) -> impl std::future::Future<Output = Result<Vec<Triple>>> + Send;
}

/// Caller-provided oracle that answers "is this edge already in the
/// knowledge graph?".
///
/// Implementations must be deterministic and side-effect-free; the
/// pipeline calls `contains` once per candidate triple.
pub trait GraphBaseline: Send + Sync {
    /// `true` if `triple` is already present in the baseline graph.
    fn contains(&self, triple: &Triple) -> impl std::future::Future<Output = Result<bool>> + Send;
}

/// Deterministic [`TripleExtractor`] used in tests and examples.
///
/// Returns the same fixed `Vec<Triple>` on every call. Hosts wiring real
/// extractors should still emit triples through [`Triple::new`] for
/// predicate normalisation; the stub does so on construction.
#[derive(Debug, Clone)]
pub struct StubTripleExtractor {
    triples: Vec<Triple>,
}

impl StubTripleExtractor {
    /// Build a stub that emits `triples` (re-normalised) on every call.
    pub fn new<I, S, P, O>(triples: I) -> Self
    where
        I: IntoIterator<Item = (S, P, O)>,
        S: Into<String>,
        P: Into<String>,
        O: Into<String>,
    {
        let triples = triples
            .into_iter()
            .map(|(s, p, o)| Triple::new(s, p, o))
            .collect();
        Self { triples }
    }

    /// Borrow the configured triples. Primarily useful for tests.
    pub fn triples(&self) -> &[Triple] {
        &self.triples
    }
}

impl TripleExtractor for StubTripleExtractor {
    async fn extract(&self, _doc: &Document) -> Result<Vec<Triple>> {
        Ok(self.triples.clone())
    }
}

/// `HashSet`-backed [`GraphBaseline`].
///
/// Always available under the `ingestion` feature. Concurrent reads and
/// writes are serialised behind a `std::sync::Mutex`; the guard is never
/// held across an `.await` so `await_holding_lock` stays clean.
#[derive(Debug, Default)]
pub struct InMemoryGraphBaseline {
    edges: Mutex<HashSet<Triple>>,
}

impl InMemoryGraphBaseline {
    /// Empty baseline.
    pub fn new() -> Self {
        Self {
            edges: Mutex::new(HashSet::new()),
        }
    }

    /// Build a baseline seeded with `edges` (re-normalised).
    pub fn with_edges<I, S, P, O>(edges: I) -> Self
    where
        I: IntoIterator<Item = (S, P, O)>,
        S: Into<String>,
        P: Into<String>,
        O: Into<String>,
    {
        let baseline = Self::new();
        if let Ok(mut guard) = baseline.edges.lock() {
            for (s, p, o) in edges {
                guard.insert(Triple::new(s, p, o));
            }
        }
        baseline
    }

    /// Insert `triple` into the baseline. Returns `true` if the edge was
    /// newly inserted.
    pub fn insert(&self, triple: Triple) -> Result<bool> {
        let mut guard = self
            .edges
            .lock()
            .map_err(|_| crate::error::Error::Ingestion("graph baseline mutex poisoned".into()))?;
        Ok(guard.insert(triple))
    }

    /// Number of edges in the baseline.
    pub fn len(&self) -> Result<usize> {
        let guard = self
            .edges
            .lock()
            .map_err(|_| crate::error::Error::Ingestion("graph baseline mutex poisoned".into()))?;
        Ok(guard.len())
    }

    /// `true` if the baseline has no edges.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

impl GraphBaseline for InMemoryGraphBaseline {
    async fn contains(&self, triple: &Triple) -> Result<bool> {
        let guard = self
            .edges
            .lock()
            .map_err(|_| crate::error::Error::Ingestion("graph baseline mutex poisoned".into()))?;
        Ok(guard.contains(triple))
    }
}

/// `petgraph`-backed [`GraphBaseline`], available behind the
/// `ingestion-graph` sub-feature.
///
/// Edges are stored in a [`petgraph::graphmap::DiGraphMap`] keyed by
/// `(subject, object)` strings, with the predicate as the edge weight.
/// This lets downstream code answer richer queries
/// (`baseline.graph().neighbors(...)`) while still serving as the
/// pipeline's edge-existence oracle.
#[cfg(feature = "ingestion-graph")]
pub use self::pet::PetgraphBaseline;

#[cfg(feature = "ingestion-graph")]
mod pet {
    use std::collections::{HashMap, HashSet};
    use std::sync::Mutex;

    use petgraph::Graph;
    use petgraph::graph::NodeIndex;

    use super::{GraphBaseline, Triple};
    use crate::error::{Error, Result};

    /// `petgraph`-backed graph baseline. See module-level docs.
    #[derive(Debug)]
    pub struct PetgraphBaseline {
        inner: Mutex<Inner>,
    }

    #[derive(Debug, Default)]
    struct Inner {
        edges: HashSet<Triple>,
        graph: Graph<String, String>,
        node_index: HashMap<String, NodeIndex>,
    }

    impl Inner {
        fn node_for(&mut self, value: &str) -> NodeIndex {
            if let Some(idx) = self.node_index.get(value) {
                return *idx;
            }
            let idx = self.graph.add_node(value.to_owned());
            self.node_index.insert(value.to_owned(), idx);
            idx
        }
    }

    impl PetgraphBaseline {
        /// Empty baseline.
        pub fn new() -> Self {
            Self {
                inner: Mutex::new(Inner::default()),
            }
        }

        /// Build a baseline seeded with `edges` (re-normalised).
        pub fn with_edges<I, S, P, O>(edges: I) -> Self
        where
            I: IntoIterator<Item = (S, P, O)>,
            S: Into<String>,
            P: Into<String>,
            O: Into<String>,
        {
            let baseline = Self::new();
            for (s, p, o) in edges {
                let _ = baseline.insert(Triple::new(s, p, o));
            }
            baseline
        }

        /// Insert `triple` into the baseline. Returns `true` if the edge
        /// was newly inserted.
        pub fn insert(&self, triple: Triple) -> Result<bool> {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| Error::Ingestion("petgraph baseline mutex poisoned".into()))?;
            let newly_inserted = guard.edges.insert(triple.clone());
            if newly_inserted {
                let subject_idx = guard.node_for(&triple.subject);
                let object_idx = guard.node_for(&triple.object);
                guard
                    .graph
                    .add_edge(subject_idx, object_idx, triple.predicate);
            }
            Ok(newly_inserted)
        }

        /// Number of distinct edges currently in the baseline.
        pub fn len(&self) -> Result<usize> {
            let guard = self
                .inner
                .lock()
                .map_err(|_| Error::Ingestion("petgraph baseline mutex poisoned".into()))?;
            Ok(guard.edges.len())
        }

        /// `true` if the baseline has no edges.
        pub fn is_empty(&self) -> Result<bool> {
            Ok(self.len()? == 0)
        }
    }

    impl Default for PetgraphBaseline {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GraphBaseline for PetgraphBaseline {
        async fn contains(&self, triple: &Triple) -> Result<bool> {
            let guard = self
                .inner
                .lock()
                .map_err(|_| Error::Ingestion("petgraph baseline mutex poisoned".into()))?;
            Ok(guard.edges.contains(triple))
        }
    }
}
