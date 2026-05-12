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
    /// per-metric Δ-mean. Fails if the two reports were produced with
    /// different judge fingerprints (silent comparison drift).
    pub fn diff(&self, baseline: &MultiReport) -> Result<ReportDiff> {
        if self.judge_fingerprint != baseline.judge_fingerprint {
            return Err(Error::BaselineMismatch(format!(
                "judge fingerprint mismatch: current={:?} baseline={:?}",
                self.judge_fingerprint, baseline.judge_fingerprint
            )));
        }
        let base_means: BTreeMap<&str, f64> = baseline
            .metrics
            .iter()
            .map(|m| (m.metric.as_str(), m.mean))
            .collect();
        let mut rows = Vec::with_capacity(self.metrics.len());
        for m in &self.metrics {
            let baseline_mean = base_means.get(m.metric.as_str()).copied();
            rows.push(MetricDelta {
                metric: m.metric.clone(),
                current_mean: m.mean,
                baseline_mean,
                delta: baseline_mean.map(|b| m.mean - b),
            });
        }
        Ok(ReportDiff { rows })
    }
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
}

/// Result of [`MultiReport::diff`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDiff {
    /// One row per metric in the current report.
    pub rows: Vec<MetricDelta>,
}

impl ReportDiff {
    /// Render the diff as a Markdown table.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("| metric | current | baseline | Δ |\n");
        out.push_str("|---|---:|---:|---:|\n");
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
                "| {} | {:.4} | {} | {} |\n",
                r.metric, r.current_mean, baseline, delta
            ));
        }
        out
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
}
