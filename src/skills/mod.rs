//! Skill / agent evaluation primitives (feature `skills`).
//!
//! While [`crate::harness::RetrievalHarness`] scores a vector store against
//! a labelled query set, the [`SkillHarness`] in this module scores an
//! *agent* (or a single skill loaded into one) against a labelled task set.
//!
//! The shape follows the vocabulary established by Anthropic's
//! [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
//! and OpenAI's
//! [Testing Agent Skills Systematically with Evals](https://developers.openai.com/blog/eval-skills):
//!
//! - A [`SkillTask`] is one labelled prompt with a `should_trigger` flag and
//!   a set of grader ids to run against the resulting transcript.
//! - A [`Transcript`] is the captured output of a single trial: final text,
//!   tool calls, token usage, elapsed time, and an optional skill-selection
//!   marker.
//! - A [`Grader`] is a deterministic check over a [`Transcript`]. Concrete
//!   graders ship in this module (see [`ContainsGrader`], [`ToolCallGrader`],
//!   [`TranscriptBudget`], [`TriggerGrader`]).
//! - An [`AgentRunner`] is user-supplied: it owns whatever agent / harness
//!   you want to evaluate, and returns one [`Transcript`] per `(task, trial)`.
//! - [`SkillHarness`] drives the matrix `tasks × trials`, applies every
//!   grader, and aggregates results into a [`SkillEvalReport`] that reuses
//!   the existing [`MetricReport`](crate::report::MetricReport) and
//!   [`ReliabilityReport`](crate::report::ReliabilityReport) infrastructure.
//!
//! ## Scope
//!
//! Phase 1 is deterministic-only. LLM-rubric judging is intentionally out of
//! scope for this module — pair the existing [`crate::ragas`] judges with a
//! custom [`Grader`] impl if you need it today.

mod grader;
mod graders;
mod groundedness;
mod harness;
#[cfg(feature = "ragas")]
mod ragas_grader;
mod runner;
mod task;
mod transcript;

pub use grader::{AsyncGrader, Grader, GraderOutcome};
pub use graders::{ContainsGrader, ToolCallGrader, TranscriptBudget, TriggerGrader};
pub use groundedness::{
    DocumentExtractorFn, GroundednessQueryFn, GroundednessScorerFn, RetrievalGroundednessGrader,
    default_document_extractor, default_query_fn, default_scorer,
};
pub use harness::{SkillEvalReport, SkillHarness, TrialRow};
#[cfg(feature = "ragas")]
pub use ragas_grader::{RagasInputsFn, RagasJudgeGrader, default_inputs_fn};
pub use runner::AgentRunner;
pub use task::{SkillTask, SkillTaskSet};
pub use transcript::{ToolCall, Transcript, Usage};
