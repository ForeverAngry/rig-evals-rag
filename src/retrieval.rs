//! Retrieval-quality metrics.
//!
//! All metrics in this module are **pure functions** of a single
//! [`GoldQuery`] and its corresponding [`RetrievedSet`]. They have no LLM
//! dependency, no async, and no side effects: feed them ground truth and a
//! ranked candidate list and they return a score in `[0.0, 1.0]`.
//!
//! Implemented metrics:
//!
//! - [`RecallAtK`]   — fraction of relevant docs retrieved in the top-k.
//! - [`PrecisionAtK`] — fraction of top-k that are relevant.
//! - [`HitRateAtK`]  — 1 if any relevant doc is in top-k, else 0.
//! - [`Mrr`]         — reciprocal rank of the first relevant hit.
//! - [`MapAtK`]      — mean average precision @ k.
//! - [`NdcgAtK`]     — graded normalized DCG @ k (uses the integer grade
//!   stored in [`GoldQuery::relevant_docs`]).
//!
//! With the exception of [`PrecisionAtK`], all metrics return `1.0` for an
//! empty `relevant_docs` set on the gold query by convention (no relevant
//! docs ⇒ vacuously perfect). [`PrecisionAtK`] always divides by `k` and
//! therefore returns `0.0` when no top-k hit is relevant — including the
//! case where the gold query has no relevant docs at all. Callers who
//! prefer to drop unjudged queries should filter them out of the
//! [`Qrels`](crate::dataset::Qrels) before scoring.

use crate::dataset::{GoldQuery, RetrievedSet};

/// A retrieval metric: a pure scoring function over a single
/// `(gold, retrieved)` pair.
///
/// Implement [`RetrievalMetric`] for new metrics; the harness will pick them
/// up via [`crate::harness`] without further wiring.
pub trait RetrievalMetric: Send + Sync {
    /// Human-readable identifier (e.g. `"recall@10"`).
    fn name(&self) -> String;

    /// Score a single retrieved set against its gold labels. The returned
    /// value should be in `[0.0, 1.0]` for the metrics shipped in this
    /// module; user metrics are free to use other ranges as long as the
    /// invariant is documented.
    fn score(&self, gold: &GoldQuery, retrieved: &RetrievedSet) -> f64;
}

/// Recall @ k.
///
/// Recall@k = |retrieved\[..k\] ∩ relevant| / |relevant|.
///
/// Vacuously `1.0` when `relevant` is empty.
#[derive(Debug, Clone, Copy)]
pub struct RecallAtK {
    /// Cut-off rank.
    pub k: usize,
}

impl RecallAtK {
    /// Construct a `Recall@k` metric.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl RetrievalMetric for RecallAtK {
    fn name(&self) -> String {
        format!("recall@{}", self.k)
    }
    fn score(&self, gold: &GoldQuery, retrieved: &RetrievedSet) -> f64 {
        let total = gold.relevant_count();
        if total == 0 {
            return 1.0;
        }
        let hits = retrieved
            .ranked
            .iter()
            .take(self.k)
            .filter(|d| gold.is_relevant(&d.doc_id))
            .count();
        hits as f64 / total as f64
    }
}

/// Precision @ k.
///
/// Precision@k = |retrieved\[..k\] ∩ relevant| / k.
///
/// Returns `0.0` if `k` is zero.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionAtK {
    /// Cut-off rank.
    pub k: usize,
}

impl PrecisionAtK {
    /// Construct a `Precision@k` metric.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl RetrievalMetric for PrecisionAtK {
    fn name(&self) -> String {
        format!("precision@{}", self.k)
    }
    fn score(&self, gold: &GoldQuery, retrieved: &RetrievedSet) -> f64 {
        if self.k == 0 {
            return 0.0;
        }
        let hits = retrieved
            .ranked
            .iter()
            .take(self.k)
            .filter(|d| gold.is_relevant(&d.doc_id))
            .count();
        hits as f64 / self.k as f64
    }
}

/// Hit Rate @ k. Returns `1.0` if any relevant doc appears in the top-k,
/// else `0.0`. Vacuously `1.0` when `relevant` is empty.
#[derive(Debug, Clone, Copy)]
pub struct HitRateAtK {
    /// Cut-off rank.
    pub k: usize,
}

impl HitRateAtK {
    /// Construct a `HitRate@k` metric.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl RetrievalMetric for HitRateAtK {
    fn name(&self) -> String {
        format!("hit_rate@{}", self.k)
    }
    fn score(&self, gold: &GoldQuery, retrieved: &RetrievedSet) -> f64 {
        if gold.relevant_count() == 0 {
            return 1.0;
        }
        let any = retrieved
            .ranked
            .iter()
            .take(self.k)
            .any(|d| gold.is_relevant(&d.doc_id));
        if any { 1.0 } else { 0.0 }
    }
}

/// Mean Reciprocal Rank.
///
/// For a single query, MRR is `1 / rank` where `rank` is the 1-indexed
/// position of the first relevant hit. Returns `0.0` if no relevant hit
/// appears in the ranked list.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mrr;

impl RetrievalMetric for Mrr {
    fn name(&self) -> String {
        "mrr".to_string()
    }
    fn score(&self, gold: &GoldQuery, retrieved: &RetrievedSet) -> f64 {
        if gold.relevant_count() == 0 {
            return 1.0;
        }
        for (idx, doc) in retrieved.ranked.iter().enumerate() {
            if gold.is_relevant(&doc.doc_id) {
                return 1.0 / ((idx + 1) as f64);
            }
        }
        0.0
    }
}

/// Mean Average Precision @ k.
///
/// AP@k = (1 / |relevant|) · Σ_{i=1..k} Precision@i · rel(i)
///
/// where `rel(i)` is 1 if the i-th retrieved doc is relevant, else 0.
#[derive(Debug, Clone, Copy)]
pub struct MapAtK {
    /// Cut-off rank.
    pub k: usize,
}

impl MapAtK {
    /// Construct a `MAP@k` metric.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl RetrievalMetric for MapAtK {
    fn name(&self) -> String {
        format!("map@{}", self.k)
    }
    fn score(&self, gold: &GoldQuery, retrieved: &RetrievedSet) -> f64 {
        let total = gold.relevant_count();
        if total == 0 {
            return 1.0;
        }
        let mut hits = 0usize;
        let mut sum = 0.0_f64;
        for (idx, doc) in retrieved.ranked.iter().take(self.k).enumerate() {
            if gold.is_relevant(&doc.doc_id) {
                hits += 1;
                sum += hits as f64 / ((idx + 1) as f64);
            }
        }
        sum / total as f64
    }
}

/// Normalized Discounted Cumulative Gain @ k (graded).
///
/// Uses the integer grade stored in [`GoldQuery::relevant_docs`] (0 for
/// unlabeled). Formula:
///
/// DCG@k = Σ_{i=1..k} (2^{grade_i} − 1) / log2(i + 1)
///
/// nDCG@k = DCG@k / IDCG@k where IDCG@k is the DCG of the best possible
/// ordering. Returns `1.0` when there are no relevant docs.
#[derive(Debug, Clone, Copy)]
pub struct NdcgAtK {
    /// Cut-off rank.
    pub k: usize,
}

impl NdcgAtK {
    /// Construct an `nDCG@k` metric.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl RetrievalMetric for NdcgAtK {
    fn name(&self) -> String {
        format!("ndcg@{}", self.k)
    }
    fn score(&self, gold: &GoldQuery, retrieved: &RetrievedSet) -> f64 {
        if gold.relevant_count() == 0 {
            return 1.0;
        }
        let dcg = retrieved
            .ranked
            .iter()
            .take(self.k)
            .enumerate()
            .map(|(idx, doc)| {
                let grade = gold.grade(&doc.doc_id) as f64;
                if grade <= 0.0 {
                    0.0
                } else {
                    ((2.0_f64).powf(grade) - 1.0) / ((idx as f64 + 2.0).log2())
                }
            })
            .sum::<f64>();

        let mut grades: Vec<u8> = gold.relevant_docs.values().copied().collect();
        grades.sort_unstable_by(|a, b| b.cmp(a));
        let idcg = grades
            .into_iter()
            .take(self.k)
            .enumerate()
            .map(|(idx, grade)| {
                if grade == 0 {
                    0.0
                } else {
                    ((2.0_f64).powf(grade as f64) - 1.0) / ((idx as f64 + 2.0).log2())
                }
            })
            .sum::<f64>();

        if idcg == 0.0 { 1.0 } else { dcg / idcg }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::dataset::RetrievedDoc;
    use std::collections::HashMap;

    fn gold(relevant: &[(&str, u8)]) -> GoldQuery {
        GoldQuery {
            query_id: "q".into(),
            query: "test".into(),
            relevant_docs: relevant
                .iter()
                .map(|(d, g)| ((*d).to_string(), *g))
                .collect::<HashMap<_, _>>(),
            reference_answer: None,
        }
    }

    fn retrieved(ids: &[&str]) -> RetrievedSet {
        RetrievedSet {
            query_id: "q".into(),
            ranked: ids
                .iter()
                .enumerate()
                .map(|(i, id)| RetrievedDoc {
                    doc_id: (*id).to_string(),
                    score: 1.0 - (i as f64) * 0.01,
                })
                .collect(),
        }
    }

    #[test]
    fn recall_monotonic_in_k() {
        let g = gold(&[("d1", 1), ("d2", 1), ("d3", 1)]);
        let r = retrieved(&["x", "d1", "y", "d2", "d3"]);
        let r1 = RecallAtK::new(1).score(&g, &r);
        let r3 = RecallAtK::new(3).score(&g, &r);
        let r10 = RecallAtK::new(10).score(&g, &r);
        assert!(r1 <= r3);
        assert!(r3 <= r10);
        assert!((r10 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn precision_correct() {
        let g = gold(&[("d1", 1), ("d2", 1)]);
        let r = retrieved(&["d1", "x", "d2", "y"]);
        let p = PrecisionAtK::new(4).score(&g, &r);
        assert!((p - 0.5).abs() < 1e-9);
    }

    #[test]
    fn mrr_perfect_rank_is_one() {
        let g = gold(&[("d1", 1)]);
        let r = retrieved(&["d1", "x", "y"]);
        assert!((Mrr.score(&g, &r) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mrr_zero_when_no_hit() {
        let g = gold(&[("d1", 1)]);
        let r = retrieved(&["x", "y", "z"]);
        assert_eq!(Mrr.score(&g, &r), 0.0);
    }

    #[test]
    fn ndcg_perfect_when_ordered_by_grade() {
        let g = gold(&[("d1", 3), ("d2", 2), ("d3", 1)]);
        let r = retrieved(&["d1", "d2", "d3"]);
        let s = NdcgAtK::new(3).score(&g, &r);
        assert!((s - 1.0).abs() < 1e-9, "got {s}");
    }

    #[test]
    fn ndcg_drops_with_bad_ordering() {
        let g = gold(&[("d1", 3), ("d2", 1)]);
        let perfect = retrieved(&["d1", "d2"]);
        let bad = retrieved(&["d2", "d1"]);
        let m = NdcgAtK::new(2);
        assert!(m.score(&g, &perfect) > m.score(&g, &bad));
    }

    #[test]
    fn hit_rate_binary() {
        let g = gold(&[("d1", 1)]);
        assert_eq!(
            HitRateAtK::new(3).score(&g, &retrieved(&["a", "d1", "c"])),
            1.0
        );
        assert_eq!(
            HitRateAtK::new(3).score(&g, &retrieved(&["a", "b", "c"])),
            0.0
        );
    }

    #[test]
    fn map_matches_hand_computation() {
        // gold: d1, d2 relevant. retrieved: [d1, x, d2, y]
        // AP = (1/2) * (1/1 + 2/3) = (1 + 0.6666...)/2 = 0.8333...
        let g = gold(&[("d1", 1), ("d2", 1)]);
        let r = retrieved(&["d1", "x", "d2", "y"]);
        let s = MapAtK::new(4).score(&g, &r);
        assert!((s - (1.0 + 2.0 / 3.0) / 2.0).abs() < 1e-9, "got {s}");
    }

    #[test]
    fn empty_relevance_is_vacuously_perfect() {
        let g = gold(&[]);
        let r = retrieved(&["a", "b", "c"]);
        assert_eq!(RecallAtK::new(3).score(&g, &r), 1.0);
        assert_eq!(NdcgAtK::new(3).score(&g, &r), 1.0);
        assert_eq!(Mrr.score(&g, &r), 1.0);
    }

    #[test]
    fn empty_relevance_is_vacuously_perfect_for_every_metric() {
        // Lock in the documented vacuous-perfect contract for every metric
        // that honours it. PrecisionAtK is the documented exception
        // (always divides by k) and is pinned separately below.
        let g = gold(&[]);
        let r = retrieved(&["a", "b", "c"]);
        let vacuous: Vec<(String, f64)> = vec![
            (RecallAtK::new(3).name(), RecallAtK::new(3).score(&g, &r)),
            (HitRateAtK::new(3).name(), HitRateAtK::new(3).score(&g, &r)),
            (Mrr.name(), Mrr.score(&g, &r)),
            (MapAtK::new(3).name(), MapAtK::new(3).score(&g, &r)),
            (NdcgAtK::new(3).name(), NdcgAtK::new(3).score(&g, &r)),
        ];
        for (name, score) in vacuous {
            assert_eq!(
                score, 1.0,
                "{name} broke the vacuous-perfect contract for empty relevance"
            );
        }

        // Contract must also hold when both gold and retrieved are empty
        // for every vacuous-perfect metric.
        let empty = retrieved(&[]);
        assert_eq!(RecallAtK::new(3).score(&g, &empty), 1.0);
        assert_eq!(HitRateAtK::new(3).score(&g, &empty), 1.0);
        assert_eq!(Mrr.score(&g, &empty), 1.0);
        assert_eq!(MapAtK::new(3).score(&g, &empty), 1.0);
        assert_eq!(NdcgAtK::new(3).score(&g, &empty), 1.0);
    }

    #[test]
    fn precision_at_k_is_not_vacuously_perfect() {
        // PrecisionAtK is the documented exception to the vacuous-perfect
        // contract: numerator can only be 0 when there are no relevant docs,
        // and the denominator stays `k`. Pin the actual behaviour so any
        // future refactor of the divergence is intentional and visible.
        let g = gold(&[]);
        let r = retrieved(&["a", "b", "c"]);
        assert_eq!(PrecisionAtK::new(3).score(&g, &r), 0.0);
        assert_eq!(PrecisionAtK::new(3).score(&g, &retrieved(&[])), 0.0);
    }
}
