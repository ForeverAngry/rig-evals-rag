//! Agent evaluation harnesses.
//!
//! Agent tasks grade captured multi-turn behavior: final answer terms,
//! forbidden leakage, expected tool calls, and turn budgets. Hosts can plug in
//! a real `rig` agent, a `rig-compose` coordinator, or a deterministic fake by
//! implementing [`AgentEvalRunner`].

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::report::MetricReport;

/// One agent-evaluation task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalTask {
    /// Stable task identifier.
    pub id: String,
    /// User prompt or task instruction.
    pub prompt: String,
    /// Terms expected in the final answer.
    #[serde(default)]
    pub expected_output_terms: Vec<String>,
    /// Terms forbidden in the final answer.
    #[serde(default)]
    pub forbidden_output_terms: Vec<String>,
    /// Tool names expected at least once during the run.
    #[serde(default)]
    pub expected_tools: Vec<String>,
    /// Optional maximum assistant-turn count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    /// Free-form metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl AgentEvalTask {
    /// Build a task with no assertions.
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            expected_output_terms: Vec::new(),
            forbidden_output_terms: Vec::new(),
            expected_tools: Vec::new(),
            max_turns: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Add a required output term.
    #[must_use]
    pub fn expect_output(mut self, term: impl Into<String>) -> Self {
        self.expected_output_terms.push(term.into());
        self
    }

    /// Add a forbidden output term.
    #[must_use]
    pub fn forbid_output(mut self, term: impl Into<String>) -> Self {
        self.forbidden_output_terms.push(term.into());
        self
    }

    /// Add an expected tool invocation.
    #[must_use]
    pub fn expect_tool(mut self, tool: impl Into<String>) -> Self {
        self.expected_tools.push(tool.into());
        self
    }

    /// Set the maximum assistant-turn count.
    #[must_use]
    pub fn with_max_turns(mut self, max: usize) -> Self {
        self.max_turns = Some(max);
        self
    }
}

/// A named collection of agent tasks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentEvalTaskSet {
    /// Suite identifier.
    pub id: String,
    /// Tasks in input order.
    pub tasks: Vec<AgentEvalTask>,
}

impl AgentEvalTaskSet {
    /// Build an empty suite.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tasks: Vec::new(),
        }
    }

    /// Append a task.
    pub fn push(&mut self, task: AgentEvalTask) {
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

/// One tool call observed during an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolCall {
    /// Tool name.
    pub name: String,
    /// JSON arguments, if captured.
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// Whether the tool succeeded, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
}

impl AgentToolCall {
    /// Build a call with null arguments and unknown success.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: serde_json::Value::Null,
            ok: None,
        }
    }
}

/// Captured agent behavior for one task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentObservation {
    /// Final user-visible answer.
    pub final_output: String,
    /// Tool calls in observed order.
    #[serde(default)]
    pub tool_calls: Vec<AgentToolCall>,
    /// Assistant turn count, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns: Option<usize>,
    /// Free-form metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// Runs one agent-evaluation task.
pub trait AgentEvalRunner: Send + Sync {
    /// Execute the task and capture observable behavior.
    fn run<'a>(
        &'a self,
        task: &'a AgentEvalTask,
    ) -> Pin<Box<dyn Future<Output = Result<AgentObservation>> + Send + 'a>>;
}

/// One graded agent result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalResult {
    /// Task identifier.
    pub task_id: String,
    /// Score in `[0, 1]`.
    pub score: f64,
    /// Whether every assertion passed.
    pub passed: bool,
    /// Captured observation.
    pub observation: AgentObservation,
    /// Human-readable notes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// Aggregated agent report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalReport {
    /// Suite identifier.
    pub suite_id: String,
    /// Number of tasks scored.
    pub n_tasks: usize,
    /// Mean score across tasks.
    pub mean_score: f64,
    /// Per-task results.
    pub results: Vec<AgentEvalResult>,
}

impl AgentEvalReport {
    /// Convert per-task scores into a generic metric report.
    #[must_use]
    pub fn metric_report(&self) -> MetricReport {
        MetricReport::from_per_query(
            "agent.behavior".to_string(),
            self.results
                .iter()
                .map(|result| (result.task_id.clone(), result.score))
                .collect(),
        )
    }
}

/// Drives agent tasks and grades captured observations.
pub struct AgentHarness<R: AgentEvalRunner> {
    runner: R,
    concurrency: usize,
}

impl<R: AgentEvalRunner> AgentHarness<R> {
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

    /// Execute and grade every task.
    pub async fn run(&self, tasks: &AgentEvalTaskSet) -> Result<AgentEvalReport> {
        if tasks.is_empty() {
            return Err(Error::Config("agent task set is empty".into()));
        }

        let rows = stream::iter(tasks.tasks.iter().map(|task| async move {
            let observation = self.runner.run(task).await?;
            Ok::<_, Error>(grade_agent_task(task, observation))
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

        Ok(AgentEvalReport {
            suite_id: tasks.id.clone(),
            n_tasks: tasks.len(),
            mean_score,
            results,
        })
    }
}

fn grade_agent_task(task: &AgentEvalTask, observation: AgentObservation) -> AgentEvalResult {
    let final_lower = observation.final_output.to_lowercase();
    let mut checks = 0usize;
    let mut passed = 0usize;
    let mut notes = Vec::new();

    for term in &task.expected_output_terms {
        checks = checks.saturating_add(1);
        if final_lower.contains(&term.to_lowercase()) {
            passed = passed.saturating_add(1);
        } else {
            notes.push(format!("missing expected output term `{term}`"));
        }
    }

    for term in &task.forbidden_output_terms {
        checks = checks.saturating_add(1);
        if final_lower.contains(&term.to_lowercase()) {
            notes.push(format!("found forbidden output term `{term}`"));
        } else {
            passed = passed.saturating_add(1);
        }
    }

    for expected_tool in &task.expected_tools {
        checks = checks.saturating_add(1);
        if observation
            .tool_calls
            .iter()
            .any(|call| call.name == *expected_tool)
        {
            passed = passed.saturating_add(1);
        } else {
            notes.push(format!("missing expected tool `{expected_tool}`"));
        }
    }

    if let Some(max_turns) = task.max_turns {
        checks = checks.saturating_add(1);
        match observation.turns {
            Some(turns) if turns <= max_turns => {
                passed = passed.saturating_add(1);
            }
            Some(turns) => notes.push(format!("turns {turns} exceeded max {max_turns}")),
            None => notes.push("runner did not report turn count".to_string()),
        }
    }

    let score = if checks == 0 {
        1.0
    } else {
        passed as f64 / checks as f64
    };

    AgentEvalResult {
        task_id: task.id.clone(),
        score,
        passed: (score - 1.0).abs() < f64::EPSILON,
        observation,
        notes,
    }
}
