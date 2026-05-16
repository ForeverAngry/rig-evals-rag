//! Operator-facing aggregation across one or more [`IngestionDelta`]s.
//!
//! [`IngestionReport`] turns a batch of deltas into reconciled counts plus
//! deterministic JSON and Markdown summaries. Useful for UAT-09 and any
//! offline pipeline that needs a single structured verdict instead of
//! walking raw deltas.

use serde::{Deserialize, Serialize};

use super::types::{DroppedItem, DroppedReason, IngestionDelta};
use crate::error::{Error, Result};

/// Counts for one ingestion track (IoC, proposition, or triple).
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackTotals {
    /// Net-new items emitted by the track.
    pub accepted: u64,
    /// Items the track dropped, by reason.
    pub dropped: u64,
}

impl TrackTotals {
    /// Total items the track inspected.
    #[must_use]
    pub fn extracted(&self) -> u64 {
        self.accepted.saturating_add(self.dropped)
    }
}

/// Counts of dropped items grouped by [`DroppedReason`] kind.
///
/// The enum is `#[non_exhaustive]`, so the report exposes a stable map
/// over reason discriminants (`"duplicate_ioc"`, `"redundant"`,
/// `"duplicate_edge"`). Unknown future variants fold into `other`.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropTotals {
    /// Track 1 duplicates against the IoC baseline.
    pub duplicate_ioc: u64,
    /// Track 3 redundancy hits above the cosine threshold.
    pub redundant: u64,
    /// Track 2 duplicate `(subject, predicate, object)` edges.
    pub duplicate_edge: u64,
    /// Anything else, e.g. variants added after this report shipped.
    pub other: u64,
}

impl DropTotals {
    /// Sum of all drop reasons.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.duplicate_ioc
            .saturating_add(self.redundant)
            .saturating_add(self.duplicate_edge)
            .saturating_add(self.other)
    }
}

/// Aggregated counts plus reconciliation across a batch of deltas.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestionReport {
    /// Number of [`IngestionDelta`]s rolled into this report.
    pub documents: u64,
    /// IoC track totals.
    pub iocs: TrackTotals,
    /// Proposition track totals.
    pub propositions: TrackTotals,
    /// Triple track totals.
    pub triples: TrackTotals,
    /// Dropped-items breakdown by reason.
    pub drops: DropTotals,
}

impl IngestionReport {
    /// Empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a report from an iterator of deltas.
    #[must_use]
    pub fn from_deltas<'a, I>(deltas: I) -> Self
    where
        I: IntoIterator<Item = &'a IngestionDelta>,
    {
        let mut report = Self::new();
        for delta in deltas {
            report.push(delta);
        }
        report
    }

    /// Fold one delta's counts into the report.
    pub fn push(&mut self, delta: &IngestionDelta) {
        self.documents = self.documents.saturating_add(1);
        self.iocs.accepted = self.iocs.accepted.saturating_add(delta.iocs.len() as u64);
        self.propositions.accepted = self
            .propositions
            .accepted
            .saturating_add(delta.propositions.len() as u64);
        self.triples.accepted = self
            .triples
            .accepted
            .saturating_add(delta.triples.len() as u64);

        for dropped in &delta.dropped {
            match dropped.item {
                DroppedItem::Ioc(_) => {
                    self.iocs.dropped = self.iocs.dropped.saturating_add(1);
                }
                DroppedItem::Proposition(_) => {
                    self.propositions.dropped = self.propositions.dropped.saturating_add(1);
                }
                DroppedItem::Triple(_) => {
                    self.triples.dropped = self.triples.dropped.saturating_add(1);
                }
            }
            match dropped.reason {
                DroppedReason::DuplicateIoc => {
                    self.drops.duplicate_ioc = self.drops.duplicate_ioc.saturating_add(1);
                }
                DroppedReason::Redundant { .. } => {
                    self.drops.redundant = self.drops.redundant.saturating_add(1);
                }
                DroppedReason::DuplicateEdge => {
                    self.drops.duplicate_edge = self.drops.duplicate_edge.saturating_add(1);
                }
                // `DroppedReason` is `#[non_exhaustive]`, so a downstream
                // build that picks up a future variant will fold into
                // `other` rather than miscount.
                #[allow(unreachable_patterns)]
                _ => {
                    self.drops.other = self.drops.other.saturating_add(1);
                }
            }
        }
    }

    /// Total items extracted across every track.
    #[must_use]
    pub fn extracted(&self) -> u64 {
        self.iocs
            .extracted()
            .saturating_add(self.propositions.extracted())
            .saturating_add(self.triples.extracted())
    }

    /// Total items accepted across every track.
    #[must_use]
    pub fn accepted(&self) -> u64 {
        self.iocs
            .accepted
            .saturating_add(self.propositions.accepted)
            .saturating_add(self.triples.accepted)
    }

    /// Total items dropped across every track.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.iocs
            .dropped
            .saturating_add(self.propositions.dropped)
            .saturating_add(self.triples.dropped)
    }

    /// Verify the canonical reconciliation invariant:
    /// `extracted = accepted + dropped` and the per-track drop totals
    /// reconcile with the per-reason breakdown.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Ingestion`] when the totals disagree.
    /// Callers can use this as a deterministic CI gate.
    pub fn check_reconciliation(&self) -> Result<()> {
        let track_drops = self.dropped();
        let reason_drops = self.drops.total();
        if track_drops != reason_drops {
            return Err(Error::Ingestion(format!(
                "ingestion report drop totals disagree: tracks={track_drops}, reasons={reason_drops}"
            )));
        }
        let extracted = self.extracted();
        let accepted_plus_dropped = self.accepted().saturating_add(track_drops);
        if extracted != accepted_plus_dropped {
            return Err(Error::Ingestion(format!(
                "ingestion report does not reconcile: extracted={extracted}, accepted+dropped={accepted_plus_dropped}"
            )));
        }
        Ok(())
    }

    /// Render the report as a stable JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Json`] when serialisation fails (should not
    /// happen for the in-crate types).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::from)
    }

    /// Render an operator-readable Markdown summary.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Ingestion Report\n\n");
        out.push_str(&format!("- Documents: {}\n", self.documents));
        out.push_str(&format!("- Extracted: {}\n", self.extracted()));
        out.push_str(&format!("- Accepted: {}\n", self.accepted()));
        out.push_str(&format!("- Dropped: {}\n\n", self.dropped()));
        out.push_str("## Tracks\n\n");
        out.push_str("| Track | Accepted | Dropped | Extracted |\n");
        out.push_str("| --- | ---: | ---: | ---: |\n");
        out.push_str(&format!(
            "| IoCs | {} | {} | {} |\n",
            self.iocs.accepted,
            self.iocs.dropped,
            self.iocs.extracted()
        ));
        out.push_str(&format!(
            "| Propositions | {} | {} | {} |\n",
            self.propositions.accepted,
            self.propositions.dropped,
            self.propositions.extracted()
        ));
        out.push_str(&format!(
            "| Triples | {} | {} | {} |\n\n",
            self.triples.accepted,
            self.triples.dropped,
            self.triples.extracted()
        ));
        out.push_str("## Drop reasons\n\n");
        out.push_str("| Reason | Count |\n");
        out.push_str("| --- | ---: |\n");
        out.push_str(&format!(
            "| duplicate_ioc | {} |\n",
            self.drops.duplicate_ioc
        ));
        out.push_str(&format!("| redundant | {} |\n", self.drops.redundant));
        out.push_str(&format!(
            "| duplicate_edge | {} |\n",
            self.drops.duplicate_edge
        ));
        out.push_str(&format!("| other | {} |\n", self.drops.other));
        out
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::expect_used
)]
mod tests {
    use super::*;
    use crate::ingestion::types::Dropped;
    use crate::ingestion::{Ioc, IocKind, Proposition, Triple};

    fn delta_with_one_of_everything() -> IngestionDelta {
        let mut d = IngestionDelta::new();
        d.iocs.push(Ioc::new(IocKind::Ipv4, "192.0.2.10"));
        d.propositions.push(Proposition::new("the sky is blue"));
        d.triples.push(Triple::new("a", "rel", "b"));
        d.dropped.push(Dropped {
            item: DroppedItem::Ioc(Ioc::new(IocKind::Ipv4, "10.0.0.1")),
            reason: DroppedReason::DuplicateIoc,
        });
        d.dropped.push(Dropped {
            item: DroppedItem::Proposition(Proposition::new("duplicate")),
            reason: DroppedReason::Redundant { similarity: 0.91 },
        });
        d.dropped.push(Dropped {
            item: DroppedItem::Triple(Triple::new("a", "rel", "b")),
            reason: DroppedReason::DuplicateEdge,
        });
        d
    }

    #[test]
    fn from_deltas_reconciles_totals() {
        let deltas = vec![
            delta_with_one_of_everything(),
            delta_with_one_of_everything(),
        ];
        let report = IngestionReport::from_deltas(&deltas);

        assert_eq!(report.documents, 2);
        assert_eq!(report.iocs.accepted, 2);
        assert_eq!(report.iocs.dropped, 2);
        assert_eq!(report.propositions.accepted, 2);
        assert_eq!(report.propositions.dropped, 2);
        assert_eq!(report.triples.accepted, 2);
        assert_eq!(report.triples.dropped, 2);

        assert_eq!(report.drops.duplicate_ioc, 2);
        assert_eq!(report.drops.redundant, 2);
        assert_eq!(report.drops.duplicate_edge, 2);
        assert_eq!(report.drops.other, 0);

        report.check_reconciliation().unwrap();
        assert_eq!(report.extracted(), 12);
        assert_eq!(report.accepted(), 6);
        assert_eq!(report.dropped(), 6);
    }

    #[test]
    fn json_round_trips_and_markdown_is_stable() {
        let report = IngestionReport::from_deltas([&delta_with_one_of_everything()]);
        let json = report.to_json().unwrap();
        let parsed: IngestionReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, report);

        let md = report.to_markdown();
        assert!(md.starts_with("# Ingestion Report"));
        assert!(md.contains("| IoCs | 1 | 1 | 2 |"));
        assert!(md.contains("| duplicate_edge | 1 |"));
    }

    #[test]
    fn empty_report_reconciles() {
        let report = IngestionReport::new();
        report.check_reconciliation().unwrap();
        assert_eq!(report.extracted(), 0);
    }
}
