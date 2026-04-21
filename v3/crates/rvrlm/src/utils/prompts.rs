use crate::types::{CodeBlock, Prompt, ReplResult};

pub const RLM_SYSTEM_PROMPT: &str = "You are rvRLM, a recursive inference runtime that can decompose work, inspect state, and return a final answer when the task is complete.";

pub fn build_user_prompt(prompt: &Prompt) -> String {
    match prompt {
        Prompt::Text(text) => text.clone(),
        Prompt::Structured(value) => value.to_string(),
    }
}

pub fn build_iteration_prompt(
    root_prompt: &Prompt,
    previous_response: Option<&str>,
    executed_blocks: &[CodeBlock],
) -> String {
    let mut sections = vec![
        RLM_SYSTEM_PROMPT.to_owned(),
        format!("ROOT_PROMPT:\n{}", build_user_prompt(root_prompt)),
        "If you are done, emit a line starting with `FINAL_ANSWER:`.".to_owned(),
        "If you need computation, emit Rust-native Rhai code in fenced code blocks.".to_owned(),
        "Set `FINAL_VAR = ...` inside code if the executed block computes the final answer."
            .to_owned(),
    ];

    if let Some(response) = previous_response {
        sections.push(format!("PREVIOUS_MODEL_RESPONSE:\n{response}"));
    }

    if !executed_blocks.is_empty() {
        let rendered_blocks = executed_blocks
            .iter()
            .enumerate()
            .map(|(index, block)| {
                format!(
                    "CODE_BLOCK_{}:\n```rhai\n{}\n```\nRESULT:\n{}",
                    index + 1,
                    block.code,
                    format_repl_result(&block.result),
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        sections.push(format!("EXECUTION_HISTORY:\n{rendered_blocks}"));
    }

    sections.push("Respond with either more code or `FINAL_ANSWER: ...`.".to_owned());
    sections.join("\n\n")
}

fn format_repl_result(result: &ReplResult) -> String {
    let locals = if result.locals.is_empty() {
        "{}".to_owned()
    } else {
        serde_json::to_string_pretty(&result.locals).unwrap_or_else(|_| "{}".to_owned())
    };

    format!(
        "stdout:\n{}\n\nstderr:\n{}\n\nlocals:\n{}\n\nfinal_answer:\n{}",
        result.stdout,
        result.stderr,
        locals,
        result.final_answer.as_deref().unwrap_or("null"),
    )
}
