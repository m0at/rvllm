use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;

use crate::error::{Result, RlmError};
use crate::types::{EnvironmentKind, Prompt, ReplResult, RlmChatCompletion};

pub const RESERVED_TOOL_NAMES: &[&str] = &[
    "llm_query",
    "llm_query_batched",
    "rlm_query",
    "rlm_query_batched",
    "FINAL_VAR",
    "SHOW_VARS",
    "context",
    "history",
];

pub trait Tool: Send + Sync {
    fn description(&self) -> Option<&str> {
        None
    }

    fn call(&self, input: &Value) -> Result<Value>;
}

pub struct FnTool {
    description: Option<String>,
    handler: Box<dyn Fn(&Value) -> Result<Value> + Send + Sync>,
}

impl FnTool {
    pub fn new<F>(description: Option<String>, handler: F) -> Self
    where
        F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
    {
        Self {
            description,
            handler: Box::new(handler),
        }
    }
}

impl Tool for FnTool {
    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn call(&self, input: &Value) -> Result<Value> {
        (self.handler)(input)
    }
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn insert<T>(&mut self, name: impl Into<String>, tool: T) -> Result<()>
    where
        T: Tool + 'static,
    {
        let name = name.into();
        if RESERVED_TOOL_NAMES.contains(&name.as_str()) {
            return Err(RlmError::InvalidConfig(format!(
                "tool name `{name}` is reserved"
            )));
        }
        self.tools.insert(name, Arc::new(tool));
        Ok(())
    }

    pub fn call(&self, name: &str, input: &Value) -> Result<Value> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| RlmError::Environment(format!("unknown tool `{name}`")))?;
        tool.call(input)
    }

    pub fn descriptions(&self) -> BTreeMap<String, Option<String>> {
        self.tools
            .iter()
            .map(|(name, tool)| (name.clone(), tool.description().map(str::to_owned)))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum HostCall {
    Llm(Prompt),
    Rlm(Prompt),
}

pub struct ExecutionCallbacks<'a> {
    pub host_call: &'a mut dyn FnMut(HostCall) -> Result<RlmChatCompletion>,
}

pub trait Environment {
    fn kind(&self) -> EnvironmentKind;
    fn fork(&self) -> Result<Box<dyn Environment>>;
    fn setup(&mut self) -> Result<()>;
    fn load_context(&mut self, context: Prompt) -> Result<()>;
    fn execute_code(
        &mut self,
        code: &str,
        callbacks: &mut ExecutionCallbacks<'_>,
    ) -> Result<ReplResult>;
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::error::RlmError;

    use super::{FnTool, ToolRegistry};

    #[test]
    fn registry_rejects_reserved_names() {
        let mut registry = ToolRegistry::default();
        let error = registry
            .insert("llm_query", FnTool::new(None, |_input| Ok(json!(null))))
            .expect_err("reserved name should fail");

        match error {
            RlmError::InvalidConfig(message) => {
                assert!(message.contains("reserved"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn registry_calls_tools_and_reports_descriptions() {
        let mut registry = ToolRegistry::default();
        registry
            .insert(
                "echo",
                FnTool::new(Some("returns the input".to_owned()), |input| {
                    Ok(input.clone())
                }),
            )
            .expect("tool insert should succeed");

        let value = registry
            .call("echo", &json!({"depth": 2}))
            .expect("tool call should succeed");
        let descriptions = registry.descriptions();

        assert_eq!(value, json!({"depth": 2}));
        assert_eq!(
            descriptions.get("echo"),
            Some(&Some("returns the input".to_owned()))
        );
    }
}
