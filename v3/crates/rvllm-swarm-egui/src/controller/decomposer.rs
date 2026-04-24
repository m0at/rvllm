// Goal -> subtask decomposition. Phase-1 uses a deterministic mock; Phase-2
// invokes the master agent via the native rvllm backend.

use crate::state::Persona;

#[derive(Clone, Debug)]
pub struct DecomposedGoal {
    pub plan_summary: String,
    pub subtasks: Vec<DecomposedSubtask>,
}

#[derive(Clone, Debug)]
pub struct DecomposedSubtask {
    pub summary: String,
    pub persona: Persona,
    pub depends_on_indices: Vec<usize>,
    pub exit_criteria: String,
}

/// A deterministic six-step decomposition that doesn't call any LLM. Good
/// for demoing the pipeline end-to-end and for the `swarm-cli --mock` tests.
pub fn mock_decompose(goal: &str) -> DecomposedGoal {
    let short: String = goal.chars().take(40).collect();
    let summary = if goal.len() > 40 {
        format!("{short}…")
    } else {
        short
    };

    let subtasks = vec![
        DecomposedSubtask {
            summary: format!("scaffold code for: {summary}"),
            persona: Persona::Runtime,
            depends_on_indices: vec![],
            exit_criteria: "cargo check passes".into(),
        },
        DecomposedSubtask {
            summary: format!("write kernels for: {summary}"),
            persona: Persona::Kernels,
            depends_on_indices: vec![0],
            exit_criteria: "file kernels/foo.cu exists".into(),
        },
        DecomposedSubtask {
            summary: format!("add tests for: {summary}"),
            persona: Persona::Tests,
            depends_on_indices: vec![0],
            exit_criteria: "cargo test passes".into(),
        },
        DecomposedSubtask {
            summary: format!("document: {summary}"),
            persona: Persona::Docs,
            depends_on_indices: vec![0],
            exit_criteria: "file docs/foo.md contains foo".into(),
        },
        DecomposedSubtask {
            summary: format!("benchmark: {summary}"),
            persona: Persona::Misc,
            depends_on_indices: vec![1, 2],
            exit_criteria: "human approval".into(),
        },
        DecomposedSubtask {
            summary: format!("review + finalise: {summary}"),
            persona: Persona::Runtime,
            depends_on_indices: vec![1, 2, 3],
            exit_criteria: "cargo clippy passes".into(),
        },
    ];

    DecomposedGoal {
        plan_summary: format!("6-step default plan for: {summary}"),
        subtasks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_decompose_shape() {
        let d = mock_decompose("write a haiku about worktrees");
        assert_eq!(d.subtasks.len(), 6);
        assert_eq!(d.subtasks[0].depends_on_indices, vec![] as Vec<usize>);
        assert!(d.subtasks[5].depends_on_indices.contains(&1));
    }
}
