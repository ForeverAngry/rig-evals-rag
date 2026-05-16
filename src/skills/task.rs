//! Task definitions for skill evaluation.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::error::{Error, Result};

/// One labelled evaluation case.
///
/// A task is a single prompt with success criteria expressed as the list of
/// [`Grader`](crate::skills::Grader) ids to run against the resulting
/// transcript. `should_trigger` optionally records which skill the agent
/// should select when the prompt is dispatched; the
/// [`TriggerGrader`](crate::skills::TriggerGrader) consumes this field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTask {
    /// Stable identifier (matches `query_id` in reports).
    pub id: String,
    /// The user-visible prompt to send to the agent.
    pub prompt: String,
    /// If `Some`, the skill the coordinator is expected to invoke. When the
    /// agent runner populates [`Transcript::skill_invoked`](crate::skills::Transcript::skill_invoked),
    /// the [`TriggerGrader`](crate::skills::TriggerGrader) compares it to
    /// this field. When `None`, the task is a negative control — see
    /// `should_trigger=false` in the OpenAI playbook.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub should_trigger: Option<String>,
    /// Grader ids to apply. Each id must resolve in the harness's grader
    /// registry, otherwise the harness returns
    /// [`Error::Config`](crate::error::Error::Config).
    #[serde(default)]
    pub graders: Vec<String>,
    /// Free-form metadata. Surfaces in reports for downstream tooling.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl SkillTask {
    /// Build a task with no graders and no trigger expectation.
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            should_trigger: None,
            graders: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Set the expected skill name for trigger evaluation.
    #[must_use]
    pub fn with_should_trigger(mut self, skill: impl Into<String>) -> Self {
        self.should_trigger = Some(skill.into());
        self
    }

    /// Append a grader id.
    #[must_use]
    pub fn with_grader(mut self, grader_id: impl Into<String>) -> Self {
        self.graders.push(grader_id.into());
        self
    }
}

/// A named, ordered collection of [`SkillTask`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillTaskSet {
    /// Human-readable identifier for the suite (e.g. `"setup-demo-app.v1"`).
    pub id: String,
    /// Tasks in input order.
    pub tasks: Vec<SkillTask>,
}

impl SkillTaskSet {
    /// Build an empty task set with the given id.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tasks: Vec::new(),
        }
    }

    /// Load a JSONL skill-task suite from disk, inferring the suite id from
    /// the file stem. Each non-empty line must deserialize into a
    /// [`SkillTask`].
    ///
    /// For explicit suite ids, use [`SkillTaskSet::load_jsonl_with_id`].
    pub fn load_jsonl<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .ok_or_else(|| {
                Error::Config(format!("could not infer skill suite id from path {path:?}"))
            })?;
        Self::load_jsonl_with_id(id, path)
    }

    /// Load a JSONL skill-task suite from disk with an explicit suite id.
    /// Each non-empty line must deserialize into a [`SkillTask`].
    pub fn load_jsonl_with_id<P: AsRef<Path>>(id: impl Into<String>, path: P) -> Result<Self> {
        let path = path.as_ref();
        debug!(?path, "loading skill task set");
        let text = std::fs::read_to_string(path)?;
        Self::from_jsonl_str(id, &text)
    }

    /// Parse a JSONL skill-task suite from a string. Each non-empty line is
    /// decoded into a [`SkillTask`]. The 1-indexed line number is included
    /// in any parse error.
    pub fn from_jsonl_str(id: impl Into<String>, text: &str) -> Result<Self> {
        let mut tasks = Vec::new();
        for (idx, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let task: SkillTask =
                serde_json::from_str(line).map_err(|source| Error::DatasetParse {
                    line: idx + 1,
                    source,
                })?;
            tasks.push(task);
        }
        Ok(Self {
            id: id.into(),
            tasks,
        })
    }

    /// Serialize the suite as JSONL, one [`SkillTask`] per line.
    pub fn to_jsonl_string(&self) -> Result<String> {
        let mut out = String::new();
        for task in &self.tasks {
            out.push_str(&serde_json::to_string(task)?);
            out.push('\n');
        }
        Ok(out)
    }

    /// Write the suite to disk as JSONL, one [`SkillTask`] per line.
    pub fn save_jsonl<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        debug!(?path, suite = %self.id, tasks = self.tasks.len(), "saving skill task set");
        std::fs::write(path, self.to_jsonl_string()?)?;
        Ok(())
    }

    /// Append a task.
    pub fn push(&mut self, task: SkillTask) {
        self.tasks.push(task);
    }

    /// Number of tasks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether the set contains any tasks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn parses_skill_task_jsonl() {
        let text = r#"{"id":"t1","prompt":"say hello","graders":["greets"],"metadata":{"tier":"smoke"}}
        {"id":"t2","prompt":"route this","should_trigger":"search","graders":["trigger"]}

        "#;
        let suite = SkillTaskSet::from_jsonl_str("demo.v1", text).unwrap();
        assert_eq!(suite.id, "demo.v1");
        assert_eq!(suite.len(), 2);
        assert_eq!(suite.tasks[0].graders, vec!["greets"]);
        assert_eq!(suite.tasks[0].metadata.get("tier").unwrap(), "smoke");
        assert_eq!(suite.tasks[1].should_trigger.as_deref(), Some("search"));
    }

    #[test]
    fn reports_line_on_skill_task_parse_error() {
        let text = "{\"id\":\"t1\",\"prompt\":\"a\"}\nnot json\n";
        let err = SkillTaskSet::from_jsonl_str("bad", text).unwrap_err();
        match err {
            Error::DatasetParse { line, .. } => assert_eq!(line, 2),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn round_trips_skill_task_jsonl_file() {
        let mut suite = SkillTaskSet::new("roundtrip.v1");
        suite.push(SkillTask::new("t1", "say hello").with_grader("greets"));
        suite.push(SkillTask::new("t2", "route").with_should_trigger("search"));

        let file = tempfile::NamedTempFile::new().unwrap();
        suite.save_jsonl(file.path()).unwrap();
        let loaded = SkillTaskSet::load_jsonl_with_id("loaded.v1", file.path()).unwrap();

        assert_eq!(loaded.id, "loaded.v1");
        assert_eq!(loaded.tasks.len(), 2);
        assert_eq!(loaded.tasks[0].id, "t1");
        assert_eq!(loaded.tasks[1].should_trigger.as_deref(), Some("search"));
    }

    #[test]
    fn load_jsonl_infers_suite_id_from_file_stem() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("smoke_suite.jsonl");
        std::fs::write(&path, "{\"id\":\"t1\",\"prompt\":\"hello\"}\n").unwrap();

        let loaded = SkillTaskSet::load_jsonl(&path).unwrap();
        assert_eq!(loaded.id, "smoke_suite");
        assert_eq!(loaded.tasks.len(), 1);
    }
}
