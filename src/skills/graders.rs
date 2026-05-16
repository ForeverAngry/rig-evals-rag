//! Off-the-shelf deterministic graders.
//!
//! These cover the "fast, explainable signals" tier from the OpenAI and
//! Anthropic playbooks. They are intentionally simple — substring and
//! tool-call assertions, transcript budgets, and a positive/negative
//! trigger check. Custom graders should implement [`Grader`] directly.

use crate::skills::grader::{Grader, GraderOutcome};
use crate::skills::task::SkillTask;
use crate::skills::transcript::Transcript;

/// Asserts that the final output contains (or does not contain) a substring.
#[derive(Debug, Clone)]
pub struct ContainsGrader {
    id: String,
    needle: String,
    /// When `true`, passes only if the needle is present. When `false`,
    /// passes only if the needle is absent — useful for negative checks
    /// like "agent must not mention internal codenames".
    expect_present: bool,
}

impl ContainsGrader {
    /// Build a grader that requires `needle` to appear in
    /// [`Transcript::final_output`].
    pub fn present(id: impl Into<String>, needle: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            needle: needle.into(),
            expect_present: true,
        }
    }

    /// Build a grader that requires `needle` to be absent from
    /// [`Transcript::final_output`].
    pub fn absent(id: impl Into<String>, needle: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            needle: needle.into(),
            expect_present: false,
        }
    }
}

impl Grader for ContainsGrader {
    fn id(&self) -> &str {
        &self.id
    }

    fn grade(&self, _task: &SkillTask, transcript: &Transcript) -> GraderOutcome {
        let present = transcript.final_output.contains(&self.needle);
        if present == self.expect_present {
            GraderOutcome::pass(&self.id)
        } else if self.expect_present {
            GraderOutcome::fail(&self.id, format!("substring {:?} not found", self.needle))
        } else {
            GraderOutcome::fail(
                &self.id,
                format!("forbidden substring {:?} present", self.needle),
            )
        }
    }
}

/// Asserts that a tool was invoked at least once during the trial.
///
/// This is the deterministic version of Anthropic's `tool_calls` grader
/// shape. Per their guidance, prefer "tool was called with these
/// parameters" over "tools were called in this exact sequence" — sequence
/// assertions are brittle to valid model variation.
#[derive(Debug, Clone)]
pub struct ToolCallGrader {
    id: String,
    tool: String,
    /// Minimum required invocation count. `1` by default.
    min_invocations: usize,
}

impl ToolCallGrader {
    /// Grader that requires at least one invocation of `tool`.
    pub fn at_least_once(id: impl Into<String>, tool: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tool: tool.into(),
            min_invocations: 1,
        }
    }

    /// Grader that requires at least `n` invocations of `tool`.
    pub fn at_least(id: impl Into<String>, tool: impl Into<String>, n: usize) -> Self {
        Self {
            id: id.into(),
            tool: tool.into(),
            min_invocations: n.max(1),
        }
    }
}

impl Grader for ToolCallGrader {
    fn id(&self) -> &str {
        &self.id
    }

    fn grade(&self, _task: &SkillTask, transcript: &Transcript) -> GraderOutcome {
        let count = transcript
            .tool_calls
            .iter()
            .filter(|call| call.name == self.tool)
            .count();
        if count >= self.min_invocations {
            GraderOutcome::pass(&self.id)
        } else {
            GraderOutcome::fail(
                &self.id,
                format!(
                    "tool {:?} invoked {} time(s); required {}",
                    self.tool, count, self.min_invocations
                ),
            )
        }
    }
}

/// Asserts ceilings on transcript turn count and token usage.
///
/// All limits are optional; a `None` field is not checked. Mirrors the
/// "command count and thrashing" / "token budget" tier from OpenAI's
/// roadmap for extending skill evals.
#[derive(Debug, Clone, Default)]
pub struct TranscriptBudget {
    id: String,
    /// Maximum allowed turns. `None` disables the check.
    pub max_turns: Option<usize>,
    /// Maximum allowed tool calls in the trial. `None` disables.
    pub max_tool_calls: Option<usize>,
    /// Maximum allowed total tokens (`input + output`). `None` disables.
    pub max_total_tokens: Option<u64>,
    /// Maximum allowed cost in USD (requires
    /// [`Usage::cost_usd`](crate::skills::Usage::cost_usd)). `None` disables.
    pub max_cost_usd: Option<f64>,
}

impl TranscriptBudget {
    /// Build a budget grader with the given id and no enforced limits.
    /// Use the public fields to set the ones you care about.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }
}

impl Grader for TranscriptBudget {
    fn id(&self) -> &str {
        &self.id
    }

    fn grade(&self, _task: &SkillTask, transcript: &Transcript) -> GraderOutcome {
        let mut violations = Vec::new();
        if let (Some(limit), Some(turns)) = (self.max_turns, transcript.turns)
            && turns > limit
        {
            violations.push(format!("turns {turns} > {limit}"));
        }
        if let Some(limit) = self.max_tool_calls {
            let n = transcript.tool_calls.len();
            if n > limit {
                violations.push(format!("tool_calls {n} > {limit}"));
            }
        }
        if let (Some(limit), Some(usage)) = (self.max_total_tokens, transcript.usage.as_ref()) {
            let total = usage.total_tokens();
            if total > limit {
                violations.push(format!("total_tokens {total} > {limit}"));
            }
        }
        if let (Some(limit), Some(cost)) = (
            self.max_cost_usd,
            transcript.usage.as_ref().and_then(|u| u.cost_usd),
        ) && cost > limit
        {
            violations.push(format!("cost_usd {cost} > {limit}"));
        }

        if violations.is_empty() {
            GraderOutcome::pass(&self.id)
        } else {
            GraderOutcome::fail(&self.id, violations.join("; "))
        }
    }
}

/// Verifies that the agent's router selected the expected skill (or
/// correctly declined to, for negative controls).
///
/// Behaviour:
///
/// - If [`SkillTask::should_trigger`] is `Some(expected)` and the
///   transcript's [`Transcript::skill_invoked`] matches, pass.
/// - If `should_trigger` is `None` and the transcript's
///   `skill_invoked` is also `None`, pass — the agent correctly chose not
///   to delegate.
/// - If the transcript does not populate `skill_invoked` at all (runner
///   does not expose routing), the grader returns a [`skipped`](GraderOutcome::skipped)
///   outcome rather than failing the task.
#[derive(Debug, Clone)]
pub struct TriggerGrader {
    id: String,
}

impl TriggerGrader {
    /// Build a trigger grader with the given id.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl Default for TriggerGrader {
    fn default() -> Self {
        Self::new("trigger")
    }
}

impl Grader for TriggerGrader {
    fn id(&self) -> &str {
        &self.id
    }

    fn grade(&self, task: &SkillTask, transcript: &Transcript) -> GraderOutcome {
        match (
            task.should_trigger.as_deref(),
            transcript.skill_invoked.as_deref(),
        ) {
            (Some(expected), Some(actual)) if expected == actual => GraderOutcome::pass(&self.id),
            (Some(expected), Some(actual)) => GraderOutcome::fail(
                &self.id,
                format!("expected skill {expected:?}, got {actual:?}"),
            ),
            (Some(expected), None) => GraderOutcome::fail(
                &self.id,
                format!("expected skill {expected:?}, runner did not record routing"),
            ),
            (None, Some(actual)) => GraderOutcome::fail(
                &self.id,
                format!("expected no skill to fire, got {actual:?}"),
            ),
            (None, None) => {
                GraderOutcome::skipped(&self.id, "no expected skill and no observed routing")
            }
        }
    }
}
