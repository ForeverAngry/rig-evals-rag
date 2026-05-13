//! Aggregation, serialization, and baseline diffing of per-query metric
//! scores produced by [`crate::harness::RetrievalHarness`].
//!
//! Two layers:
//!
//! - [`MetricReport`] — aggregates a single metric across all queries (mean,
//!   stddev, P50/P95, min/max, per-query scores).
//! - [`MultiReport`]  — bundles several [`MetricReport`]s with optional
//!   metadata (dataset id, store kind, judge fingerprint) so reports can be
//!   diffed across runs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Aggregated statistics for a single metric across a query set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricReport {
    /// Metric identifier (e.g. `"recall@10"`).
    pub metric: String,
    /// Number of queries scored.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation (N-1). `0.0` for `n < 2`.
    pub stddev: f64,
    /// Minimum observed score.
    pub min: f64,
    /// Maximum observed score.
    pub max: f64,
    /// 50th percentile (median) via linear interpolation.
    pub p50: f64,
    /// 95th percentile via linear interpolation.
    pub p95: f64,
    /// Per-query `(query_id, score)` pairs, in input order.
    pub per_query: Vec<(String, f64)>,
}

impl MetricReport {
    /// Build a [`MetricReport`] from per-query `(query_id, score)` pairs.
    ///
    /// Scores are aggregated in-place; the original ordering is preserved
    /// in [`MetricReport::per_query`] for diff and audit use cases.
    pub fn from_per_query(metric: String, per_query: Vec<(String, f64)>) -> Self {
        let n = per_query.len();
        if n == 0 {
            return Self {
                metric,
                n: 0,
                mean: 0.0,
                stddev: 0.0,
                min: 0.0,
                max: 0.0,
                p50: 0.0,
                p95: 0.0,
                per_query,
            };
        }
        let scores: Vec<f64> = per_query.iter().map(|(_, s)| *s).collect();
        let sum: f64 = scores.iter().sum();
        let mean = sum / n as f64;
        let var = if n > 1 {
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
        } else {
            0.0
        };
        let stddev = var.sqrt();

        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let p50 = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);

        Self {
            metric,
            n,
            mean,
            stddev,
            min,
            max,
            p50,
            p95,
            per_query,
        }
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let rank = q * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let lo_v = sorted.get(lo).copied().unwrap_or(0.0);
    let hi_v = sorted.get(hi).copied().unwrap_or(lo_v);
    let frac = rank - lo as f64;
    lo_v + (hi_v - lo_v) * frac
}

/// A bundle of [`MetricReport`]s with optional run metadata, suitable for
/// JSON persistence and baseline comparison.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiReport {
    /// Free-form dataset identifier (e.g. `"beir/nq"` or `"internal/v3"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
    /// Free-form store identifier (e.g. `"memvid:livetest.mv2"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store_kind: Option<String>,
    /// Opaque fingerprint of any LLM judges used. Reports with mismatched
    /// fingerprints refuse to diff to prevent silent comparison drift.
    /// Reserved for the upcoming `ragas` feature; pure retrieval runs leave
    /// this empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_fingerprint: Option<String>,
    /// One report per metric, in the order metrics were declared.
    pub metrics: Vec<MetricReport>,
}

impl MultiReport {
    /// Construct a [`MultiReport`] from a metric report vector. Other
    /// metadata is filled in via the `with_*` builders.
    #[must_use]
    pub fn new(metrics: Vec<MetricReport>) -> Self {
        Self {
            metrics,
            ..Default::default()
        }
    }

    /// Attach a dataset identifier.
    #[must_use]
    pub fn with_dataset(mut self, id: impl Into<String>) -> Self {
        self.dataset_id = Some(id.into());
        self
    }

    /// Attach a store kind identifier.
    #[must_use]
    pub fn with_store(mut self, kind: impl Into<String>) -> Self {
        self.store_kind = Some(kind.into());
        self
    }

    /// Attach a judge fingerprint (reserved for `ragas`).
    #[must_use]
    pub fn with_judge_fingerprint(mut self, fp: impl Into<String>) -> Self {
        self.judge_fingerprint = Some(fp.into());
        self
    }

    /// Serialize as pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Render a compact Markdown summary table.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("| metric | n | mean | stddev | p50 | p95 | min | max |\n");
        out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for m in &self.metrics {
            out.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                m.metric, m.n, m.mean, m.stddev, m.p50, m.p95, m.min, m.max
            ));
        }
        out
    }

    /// Diff this report against a baseline. Returns a [`ReportDiff`] with
    /// per-metric Δ-mean and per-query winners/losers. Fails if the two
    /// reports were produced with different judge fingerprints (silent
    /// comparison drift).
    ///
    /// Per-query deltas are computed by intersecting the two reports'
    /// `per_query` vectors on `query_id`. Queries missing from either side
    /// are skipped (they cannot be compared). `winners`, `losers`, and
    /// `unchanged` use an absolute threshold of `1e-9` to filter floating
    /// point noise; callers needing different sensitivity should inspect
    /// [`MetricDelta::query_changes`] directly.
    pub fn diff(&self, baseline: &MultiReport) -> Result<ReportDiff> {
        if self.judge_fingerprint != baseline.judge_fingerprint {
            return Err(Error::BaselineMismatch(format!(
                "judge fingerprint mismatch: current={:?} baseline={:?}",
                self.judge_fingerprint, baseline.judge_fingerprint
            )));
        }
        let base_by_name: BTreeMap<&str, &MetricReport> = baseline
            .metrics
            .iter()
            .map(|m| (m.metric.as_str(), m))
            .collect();
        let mut rows = Vec::with_capacity(self.metrics.len());
        for m in &self.metrics {
            let base = base_by_name.get(m.metric.as_str()).copied();
            let baseline_mean = base.map(|b| b.mean);
            let (query_changes, winners, losers, unchanged) = match base {
                Some(b) => compute_query_changes(&m.per_query, &b.per_query),
                None => (Vec::new(), 0, 0, 0),
            };
            rows.push(MetricDelta {
                metric: m.metric.clone(),
                current_mean: m.mean,
                baseline_mean,
                delta: baseline_mean.map(|b| m.mean - b),
                winners,
                losers,
                unchanged,
                query_changes,
            });
        }
        Ok(ReportDiff { rows })
    }
}

/// Floating-point noise floor used when bucketing per-query deltas into
/// winners / losers / unchanged. Deltas with `|delta| <= EPSILON` count as
/// unchanged.
const EPSILON: f64 = 1e-9;

/// Intersect per-query scores and return `(changes, winners, losers, unchanged)`.
/// `changes` is sorted by `|delta|` descending so the largest movers are
/// surfaced first.
fn compute_query_changes(
    current: &[(String, f64)],
    baseline: &[(String, f64)],
) -> (Vec<QueryDelta>, usize, usize, usize) {
    let base_by_query: BTreeMap<&str, f64> =
        baseline.iter().map(|(q, s)| (q.as_str(), *s)).collect();
    let mut changes = Vec::new();
    let mut winners = 0usize;
    let mut losers = 0usize;
    let mut unchanged = 0usize;
    for (query_id, cur_score) in current {
        let Some(base_score) = base_by_query.get(query_id.as_str()).copied() else {
            continue;
        };
        let delta = cur_score - base_score;
        if delta > EPSILON {
            winners += 1;
        } else if delta < -EPSILON {
            losers += 1;
        } else {
            unchanged += 1;
        }
        changes.push(QueryDelta {
            query_id: query_id.clone(),
            current: *cur_score,
            baseline: base_score,
            delta,
        });
    }
    changes.sort_by(|a, b| {
        b.delta
            .abs()
            .partial_cmp(&a.delta.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (changes, winners, losers, unchanged)
}

/// Per-metric delta produced by [`MultiReport::diff`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    /// Metric identifier.
    pub metric: String,
    /// Mean from the current report.
    pub current_mean: f64,
    /// Mean from the baseline report, if the metric was present.
    pub baseline_mean: Option<f64>,
    /// `current_mean - baseline_mean`, if comparable.
    pub delta: Option<f64>,
    /// Number of queries whose score improved relative to the baseline.
    #[serde(default)]
    pub winners: usize,
    /// Number of queries whose score regressed relative to the baseline.
    #[serde(default)]
    pub losers: usize,
    /// Number of queries whose score was unchanged (within floating-point
    /// noise) relative to the baseline.
    #[serde(default)]
    pub unchanged: usize,
    /// Per-query deltas for queries present in both reports, sorted by
    /// `|delta|` descending. Empty if the metric was missing from the
    /// baseline.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub query_changes: Vec<QueryDelta>,
}

/// Per-query score change for a single metric, produced by
/// [`MultiReport::diff`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDelta {
    /// Gold-query identifier.
    pub query_id: String,
    /// Score on the current report.
    pub current: f64,
    /// Score on the baseline report.
    pub baseline: f64,
    /// `current - baseline`.
    pub delta: f64,
}

/// Result of [`MultiReport::diff`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDiff {
    /// One row per metric in the current report.
    pub rows: Vec<MetricDelta>,
}

impl ReportDiff {
    /// Render the diff as a Markdown table including per-metric mean delta
    /// and per-query winner/loser/unchanged counts. Per-query movers are
    /// not inlined; inspect [`MetricDelta::query_changes`] for that detail.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("| metric | current | baseline | Δ | win | lose | same |\n");
        out.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
        for r in &self.rows {
            let baseline = r
                .baseline_mean
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "—".to_string());
            let delta = r
                .delta
                .map(|v| format!("{v:+.4}"))
                .unwrap_or_else(|| "—".to_string());
            out.push_str(&format!(
                "| {} | {:.4} | {} | {} | {} | {} | {} |\n",
                r.metric, r.current_mean, baseline, delta, r.winners, r.losers, r.unchanged
            ));
        }
        out
    }

    /// Serialize as pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Evaluate the diff against a [`RegressionGate`]. Returns the subset
    /// of [`MetricDelta`] rows whose mean delta is more negative than the
    /// configured threshold for that metric. Metrics not listed in the
    /// gate are ignored.
    #[must_use]
    pub fn regressions(&self, gate: &RegressionGate) -> Vec<MetricDelta> {
        self.rows
            .iter()
            .filter(|r| match (gate.threshold(&r.metric), r.delta) {
                (Some(threshold), Some(delta)) => delta < -threshold,
                _ => false,
            })
            .cloned()
            .collect()
    }
}

/// Threshold-based regression gate over a [`ReportDiff`].
///
/// Each entry maps a metric name to the **minimum tolerated drop** in mean
/// score: a metric regresses when its `delta` is more negative than
/// `-threshold`. Thresholds are non-negative; negative values are clamped
/// to zero on insert.
///
/// ```
/// use rig_evals_rag::RegressionGate;
///
/// let gate = RegressionGate::new()
///     .with_threshold("recall@10", 0.02)
///     .with_threshold("ndcg@10", 0.01);
/// assert_eq!(gate.threshold("recall@10"), Some(0.02));
/// assert_eq!(gate.threshold("mrr"), None);
/// ```
#[derive(Debug, Clone, Default)]
pub struct RegressionGate {
    thresholds: BTreeMap<String, f64>,
}

impl RegressionGate {
    /// Build an empty gate. Metrics added via
    /// [`RegressionGate::with_threshold`] participate in regression checks;
    /// any others are ignored.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `threshold` as the maximum tolerated drop in mean score
    /// for `metric`. Negative values are clamped to `0.0`.
    #[must_use]
    pub fn with_threshold(mut self, metric: impl Into<String>, threshold: f64) -> Self {
        self.thresholds.insert(metric.into(), threshold.max(0.0));
        self
    }

    /// Threshold registered for `metric`, if any.
    #[must_use]
    pub fn threshold(&self, metric: &str) -> Option<f64> {
        self.thresholds.get(metric).copied()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn metric_report_aggregates() {
        let r = MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.0), ("q2".into(), 0.5), ("q3".into(), 1.0)],
        );
        assert_eq!(r.n, 3);
        assert!((r.mean - 0.5).abs() < 1e-9);
        assert!((r.min - 0.0).abs() < 1e-9);
        assert!((r.max - 1.0).abs() < 1e-9);
        assert!((r.p50 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn empty_report_is_zero() {
        let r = MetricReport::from_per_query("m".into(), vec![]);
        assert_eq!(r.n, 0);
        assert_eq!(r.mean, 0.0);
    }

    #[test]
    fn diff_flags_fingerprint_mismatch() {
        let a = MultiReport::new(vec![]).with_judge_fingerprint("a");
        let b = MultiReport::new(vec![]).with_judge_fingerprint("b");
        assert!(a.diff(&b).is_err());
    }

    #[test]
    fn diff_computes_per_metric_delta() {
        let cur = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.8)],
        )]);
        let base = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.6)],
        )]);
        let diff = cur.diff(&base).unwrap();
        assert_eq!(diff.rows.len(), 1);
        let row = &diff.rows[0];
        assert!((row.delta.unwrap_or(0.0) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn diff_buckets_per_query_winners_losers_and_unchanged() {
        let cur = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![
                ("q1".into(), 1.0), // winner: 0.5 -> 1.0
                ("q2".into(), 0.0), // loser:  0.5 -> 0.0
                ("q3".into(), 0.5), // unchanged
                ("q4".into(), 0.9), // current-only, skipped
            ],
        )]);
        let base = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![
                ("q1".into(), 0.5),
                ("q2".into(), 0.5),
                ("q3".into(), 0.5),
                ("q5".into(), 1.0), // baseline-only, skipped
            ],
        )]);
        let diff = cur.diff(&base).unwrap();
        let row = &diff.rows[0];
        assert_eq!(row.winners, 1);
        assert_eq!(row.losers, 1);
        assert_eq!(row.unchanged, 1);
        // q4 / q5 are skipped because they are not in both reports.
        assert_eq!(row.query_changes.len(), 3);
        // Sorted by |delta| desc: q1 and q2 tie at 0.5, q3 at 0.0.
        assert_eq!(row.query_changes[2].query_id, "q3");
        assert!((row.query_changes[2].delta).abs() < 1e-9);
    }

    #[test]
    fn diff_query_changes_empty_when_baseline_missing_metric() {
        let cur = MultiReport::new(vec![MetricReport::from_per_query(
            "ndcg@10".into(),
            vec![("q1".into(), 0.9)],
        )]);
        let base = MultiReport::new(vec![]);
        let diff = cur.diff(&base).unwrap();
        let row = &diff.rows[0];
        assert!(row.delta.is_none());
        assert_eq!(row.winners, 0);
        assert_eq!(row.losers, 0);
        assert_eq!(row.unchanged, 0);
        assert!(row.query_changes.is_empty());
    }

    #[test]
    fn regression_gate_flags_only_metrics_below_threshold() {
        // recall@10 drops 0.10 (regression), ndcg@10 drops 0.005 (within
        // tolerance), mrr is not in the gate (ignored).
        let cur = MultiReport::new(vec![
            MetricReport::from_per_query("recall@10".into(), vec![("q1".into(), 0.50)]),
            MetricReport::from_per_query("ndcg@10".into(), vec![("q1".into(), 0.595)]),
            MetricReport::from_per_query("mrr".into(), vec![("q1".into(), 0.10)]),
        ]);
        let base = MultiReport::new(vec![
            MetricReport::from_per_query("recall@10".into(), vec![("q1".into(), 0.60)]),
            MetricReport::from_per_query("ndcg@10".into(), vec![("q1".into(), 0.60)]),
            MetricReport::from_per_query("mrr".into(), vec![("q1".into(), 0.90)]),
        ]);
        let diff = cur.diff(&base).unwrap();
        let gate = RegressionGate::new()
            .with_threshold("recall@10", 0.02)
            .with_threshold("ndcg@10", 0.02);
        let regressed = diff.regressions(&gate);
        assert_eq!(regressed.len(), 1);
        assert_eq!(regressed[0].metric, "recall@10");
    }

    #[test]
    fn regression_gate_clamps_negative_thresholds() {
        let gate = RegressionGate::new().with_threshold("recall@10", -0.5);
        assert_eq!(gate.threshold("recall@10"), Some(0.0));
    }

    #[test]
    fn report_diff_to_json_round_trips() {
        let cur = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 1.0), ("q2".into(), 0.0)],
        )]);
        let base = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.5), ("q2".into(), 0.5)],
        )]);
        let diff = cur.diff(&base).unwrap();
        let json = diff.to_json().unwrap();
        let parsed: ReportDiff = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        assert_eq!(parsed.rows[0].winners, 1);
        assert_eq!(parsed.rows[0].losers, 1);
        assert_eq!(parsed.rows[0].query_changes.len(), 2);
    }
}
