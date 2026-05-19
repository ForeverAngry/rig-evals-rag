//! Memory evaluation harnesses.
//!
//! The memory harness is backend-neutral: hosts provide a [`MemoryRunner`]
//! that performs writes, reloads, recall, or any other store-specific work,
//! and the harness grades the captured [`MemoryObservation`] against explicit
//! expected / forbidden recall terms.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::report::MetricReport;

/// One memory-evaluation task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTask {
    /// Stable task identifier.
    pub id: String,
    /// Optional text the runner should write before recall.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write: Option<String>,
    /// Recall query to execute after any write/reload behavior.
    pub query: String,
    /// Terms that should appear somewhere in the retrieved evidence.
    #[serde(default)]
    pub expected_terms: Vec<String>,
    /// Terms that must not appear in retrieved evidence.
    #[serde(default)]
    pub forbidden_terms: Vec<String>,
    /// Whether the runner should exercise a reload/reopen boundary.
    #[serde(default)]
    pub require_reload: bool,
    /// Free-form metadata for downstream tooling.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl MemoryTask {
    /// Build a task with a recall query and no assertions.
    pub fn new(id: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            write: None,
            query: query.into(),
            expected_terms: Vec::new(),
            forbidden_terms: Vec::new(),
            require_reload: false,
            metadata: BTreeMap::new(),
        }
    }

    /// Set text the runner should write before recall.
    #[must_use]
    pub fn with_write(mut self, text: impl Into<String>) -> Self {
        self.write = Some(text.into());
        self
    }

    /// Add an expected recall term.
    #[must_use]
    pub fn expect_term(mut self, term: impl Into<String>) -> Self {
        self.expected_terms.push(term.into());
        self
    }

    /// Add a forbidden recall term.
    #[must_use]
    pub fn forbid_term(mut self, term: impl Into<String>) -> Self {
        self.forbidden_terms.push(term.into());
        self
    }

    /// Require the runner to cross a reload/reopen boundary.
    #[must_use]
    pub fn requiring_reload(mut self) -> Self {
        self.require_reload = true;
        self
    }
}

/// A named collection of memory tasks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryTaskSet {
    /// Suite identifier.
    pub id: String,
    /// Tasks in input order.
    pub tasks: Vec<MemoryTask>,
}

impl MemoryTaskSet {
    /// Build an empty suite.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tasks: Vec::new(),
        }
    }

    /// Append a task.
    pub fn push(&mut self, task: MemoryTask) {
        self.tasks.push(task);
    }

    /// Number of tasks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether the suite is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

/// Captured output from a memory backend for one task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryObservation {
    /// Retrieved snippets or structured evidence rendered as text.
    #[serde(default)]
    pub retrieved: Vec<String>,
    /// Whether the runner actually crossed a reload/reopen boundary.
    #[serde(default)]
    pub reloaded: bool,
    /// Optional store item/frame count after the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_count: Option<u64>,
    /// Free-form metadata for downstream tooling.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// Runs a memory task against a concrete backend.
pub trait MemoryRunner: Send + Sync {
    /// Execute one task and return captured evidence for grading.
    fn run<'a>(
        &'a self,
        task: &'a MemoryTask,
    ) -> Pin<Box<dyn Future<Output = Result<MemoryObservation>> + Send + 'a>>;
}

/// One graded memory task result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvalResult {
    /// Task identifier.
    pub task_id: String,
    /// Score in `[0, 1]`.
    pub score: f64,
    /// Whether every assertion passed.
    pub passed: bool,
    /// Captured observation.
    pub observation: MemoryObservation,
    /// Human-readable notes for failed or partial assertions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// Aggregated memory evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvalReport {
    /// Suite identifier.
    pub suite_id: String,
    /// Number of tasks scored.
    pub n_tasks: usize,
    /// Mean score across tasks.
    pub mean_score: f64,
    /// Per-task results.
    pub results: Vec<MemoryEvalResult>,
}

impl MemoryEvalReport {
    /// Convert per-task scores into a generic metric report.
    #[must_use]
    pub fn metric_report(&self) -> MetricReport {
        MetricReport::from_per_query(
            "memory.recall".to_string(),
            self.results
                .iter()
                .map(|result| (result.task_id.clone(), result.score))
                .collect(),
        )
    }
}

/// Drives memory tasks and grades retrieved evidence.
pub struct MemoryHarness<R: MemoryRunner> {
    runner: R,
    concurrency: usize,
}

impl<R: MemoryRunner> MemoryHarness<R> {
    /// Build a harness for a runner.
    pub fn new(runner: R) -> Self {
        Self {
            runner,
            concurrency: 1,
        }
    }

    /// Set maximum concurrent tasks. Values of `0` are clamped to `1`.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Execute and grade every memory task.
    pub async fn run(&self, tasks: &MemoryTaskSet) -> Result<MemoryEvalReport> {
        if tasks.is_empty() {
            return Err(Error::Config("memory task set is empty".into()));
        }

        let rows = stream::iter(tasks.tasks.iter().map(|task| async move {
            let observation = self.runner.run(task).await?;
            Ok::<_, Error>(grade_memory_task(task, observation))
        }))
        .buffer_unordered(self.concurrency)
        .collect::<Vec<_>>()
        .await;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            results.push(row?);
        }
        results.sort_by(|a, b| a.task_id.cmp(&b.task_id));

        let mean_score = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|result| result.score).sum::<f64>() / results.len() as f64
        };

        Ok(MemoryEvalReport {
            suite_id: tasks.id.clone(),
            n_tasks: tasks.len(),
            mean_score,
            results,
        })
    }
}

fn grade_memory_task(task: &MemoryTask, observation: MemoryObservation) -> MemoryEvalResult {
    let joined = observation.retrieved.join("\n").to_lowercase();
    let mut checks = 0usize;
    let mut passed = 0usize;
    let mut notes = Vec::new();

    for term in &task.expected_terms {
        checks = checks.saturating_add(1);
        if joined.contains(&term.to_lowercase()) {
            passed = passed.saturating_add(1);
        } else {
            notes.push(format!("missing expected term `{term}`"));
        }
    }

    for term in &task.forbidden_terms {
        checks = checks.saturating_add(1);
        if joined.contains(&term.to_lowercase()) {
            notes.push(format!("found forbidden term `{term}`"));
        } else {
            passed = passed.saturating_add(1);
        }
    }

    if task.require_reload {
        checks = checks.saturating_add(1);
        if observation.reloaded {
            passed = passed.saturating_add(1);
        } else {
            notes.push("runner did not report reload".to_string());
        }
    }

    let score = if checks == 0 {
        1.0
    } else {
        passed as f64 / checks as f64
    };

    MemoryEvalResult {
        task_id: task.id.clone(),
        score,
        passed: (score - 1.0).abs() < f64::EPSILON,
        observation,
        notes,
    }
}
