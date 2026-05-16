//! Deterministic check contract.

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::skills::task::SkillTask;
use crate::skills::transcript::Transcript;

/// Outcome of one grader applied to one trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraderOutcome {
    /// Grader identifier (matches [`Grader::id`]).
    pub id: String,
    /// Continuous score in `[0.0, 1.0]`. `1.0` is best.
    pub score: f64,
    /// Boolean pass/fail derived by the grader. Typically `score >= 1.0`,
    /// but graders are free to set a stricter or looser threshold.
    pub passed: bool,
    /// Free-form explanation surfaced in reports and transcripts.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
}

impl GraderOutcome {
    /// Build a passing outcome with score `1.0`.
    pub fn pass(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            score: 1.0,
            passed: true,
            notes: String::new(),
        }
    }

    /// Build a failing outcome with score `0.0`.
    pub fn fail(id: impl Into<String>, notes: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            score: 0.0,
            passed: false,
            notes: notes.into(),
        }
    }

    /// Build a partial-credit outcome. `score` is clamped to `[0.0, 1.0]`.
    pub fn partial(id: impl Into<String>, score: f64, notes: impl Into<String>) -> Self {
        let score = score.clamp(0.0, 1.0);
        Self {
            id: id.into(),
            score,
            passed: score >= 1.0,
            notes: notes.into(),
        }
    }

    /// Build a skipped outcome — score `1.0`, passed, with a note. Used by
    /// graders whose preconditions are not met (e.g. a trigger grader run
    /// against a transcript whose runner did not populate
    /// [`Transcript::skill_invoked`](crate::skills::Transcript::skill_invoked)).
    pub fn skipped(id: impl Into<String>, notes: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            score: 1.0,
            passed: true,
            notes: notes.into(),
        }
    }
}

/// Deterministic check applied to a captured [`Transcript`].
///
/// Graders are synchronous and pure: given the same task and transcript
/// they must return the same outcome. Stateful or asynchronous checks
/// (LLM rubrics, network calls, environment probes) belong upstream of the
/// grader — populate the relevant [`Transcript`] fields in the
/// [`AgentRunner`](crate::skills::AgentRunner) and grade the captured
/// result.
pub trait Grader: Send + Sync {
    /// Stable identifier referenced from [`SkillTask::graders`].
    fn id(&self) -> &str;

    /// Apply the check.
    fn grade(&self, task: &SkillTask, transcript: &Transcript) -> GraderOutcome;
}

/// Async-capable check applied to a captured [`Transcript`].
///
/// `AsyncGrader` is the trait the [`SkillHarness`](crate::skills::SkillHarness)
/// actually drives. It exists so that LLM-rubric judges (e.g. the
/// [`RagasJudgeGrader`](crate::skills::RagasJudgeGrader) wrapper around a
/// `RagasMetric`) can sit alongside the cheap deterministic [`Grader`]s in
/// the same registry.
///
/// Every type that implements the synchronous [`Grader`] trait
/// automatically implements [`AsyncGrader`] via a blanket impl, so you only
/// need to implement this trait directly when you actually need to `.await`
/// during grading.
///
/// The boxed-future return shape mirrors
/// [`DynRagasMetric`](crate::ragas::DynRagasMetric) — see that module for
/// the rationale (object-safety with bounded concurrency).
pub trait AsyncGrader: Send + Sync {
    /// Stable identifier referenced from [`SkillTask::graders`].
    fn id(&self) -> &str;

    /// Apply the check. Implementations should be deterministic in their
    /// public scoring contract: the same `(task, transcript)` should
    /// produce the same outcome distribution, even when the judge itself
    /// is a stochastic LLM. Use sampling or temperature controls inside
    /// the grader if you need reproducibility.
    fn grade<'a>(
        &'a self,
        task: &'a SkillTask,
        transcript: &'a Transcript,
    ) -> Pin<Box<dyn Future<Output = GraderOutcome> + Send + 'a>>;
}

impl<G: Grader + ?Sized> AsyncGrader for G {
    fn id(&self) -> &str {
        Grader::id(self)
    }

    fn grade<'a>(
        &'a self,
        task: &'a SkillTask,
        transcript: &'a Transcript,
    ) -> Pin<Box<dyn Future<Output = GraderOutcome> + Send + 'a>> {
        let outcome = Grader::grade(self, task, transcript);
        Box::pin(async move { outcome })
    }
}
