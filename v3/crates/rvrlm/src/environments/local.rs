use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rhai::{Array, Dynamic, Engine, EvalAltResult, ImmutableString, Map, Position, Scope};
use serde_json::{Number, Value};

use crate::error::{Result, RlmError};
use crate::types::{EnvironmentKind, Prompt, ReplResult};

use super::{Environment, ExecutionCallbacks, HostCall, ToolRegistry};

type RhaiResult = std::result::Result<Dynamic, Box<EvalAltResult>>;

#[derive(Default)]
pub struct LocalEnvironment {
    initialized: bool,
    contexts: Vec<Prompt>,
    history: Vec<Prompt>,
    tools: ToolRegistry,
    scope: Scope<'static>,
}

impl LocalEnvironment {
    pub fn new(tools: ToolRegistry) -> Self {
        Self {
            initialized: false,
            contexts: Vec::new(),
            history: Vec::new(),
            tools,
            scope: Scope::new(),
        }
    }

    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }

    pub fn history_count(&self) -> usize {
        self.history.len()
    }

    pub fn tool_descriptions(&self) -> BTreeMap<String, Option<String>> {
        self.tools.descriptions()
    }

    fn set_scope_value(&mut self, name: &str, value: Dynamic) {
        if self.scope.contains(name) {
            self.scope.set_value(name, value);
        } else {
            self.scope.push_dynamic(name, value);
        }
    }
}

impl Environment for LocalEnvironment {
    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Local
    }

    fn fork(&self) -> Result<Box<dyn Environment>> {
        Ok(Box::new(Self::new(self.tools.clone())))
    }

    fn setup(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    fn load_context(&mut self, context: Prompt) -> Result<()> {
        if !self.initialized {
            self.setup()?;
        }
        self.history.push(context.clone());
        self.contexts.push(context);
        Ok(())
    }

    fn execute_code(
        &mut self,
        code: &str,
        callbacks: &mut ExecutionCallbacks<'_>,
    ) -> Result<ReplResult> {
        if !self.initialized {
            self.setup()?;
        }

        self.set_scope_value(
            "context",
            json_to_dynamic(self.contexts.last().cloned().unwrap_or_default().as_value()),
        );
        self.set_scope_value(
            "history",
            Dynamic::from_array(
                self.history
                    .iter()
                    .map(|prompt| json_to_dynamic(prompt.as_value()))
                    .collect(),
            ),
        );
        self.set_scope_value("FINAL_VAR", Dynamic::UNIT);

        let stdout = Arc::new(Mutex::new(String::new()));
        let stderr = Arc::new(Mutex::new(String::new()));
        let llm_calls = Arc::new(Mutex::new(Vec::new()));
        let locals_snapshot = Arc::new(Mutex::new(Map::new()));
        let callbacks_addr = callbacks as *mut ExecutionCallbacks<'_> as *mut () as usize;

        let mut engine = Engine::new();
        {
            let stdout = Arc::clone(&stdout);
            engine.on_print(move |line| {
                let mut out = stdout.lock().expect("stdout lock");
                out.push_str(line);
                out.push('\n');
            });
        }
        {
            let stderr = Arc::clone(&stderr);
            engine.on_debug(move |line, _src, _pos| {
                let mut err = stderr.lock().expect("stderr lock");
                err.push_str(line);
                err.push('\n');
            });
        }
        register_host_query(
            &mut engine,
            "llm_query",
            HostCallKind::Llm,
            callbacks_addr,
            Arc::clone(&llm_calls),
        );
        register_host_query(
            &mut engine,
            "rlm_query",
            HostCallKind::Rlm,
            callbacks_addr,
            Arc::clone(&llm_calls),
        );
        register_host_query_batched(
            &mut engine,
            "llm_query_batched",
            HostCallKind::Llm,
            callbacks_addr,
            Arc::clone(&llm_calls),
        );
        register_host_query_batched(
            &mut engine,
            "rlm_query_batched",
            HostCallKind::Rlm,
            callbacks_addr,
            Arc::clone(&llm_calls),
        );
        register_tools(
            &mut engine,
            self.tools.clone(),
            Arc::clone(&llm_calls),
            callbacks_addr,
        );
        {
            let locals_snapshot = Arc::clone(&locals_snapshot);
            engine.register_fn("SHOW_VARS", move || -> Map {
                locals_snapshot
                    .lock()
                    .expect("locals snapshot lock")
                    .clone()
            });
        }

        let started = Instant::now();
        if let Err(error) = engine.eval_with_scope::<Dynamic>(&mut self.scope, code) {
            let mut err = stderr.lock().expect("stderr lock");
            if !err.is_empty() && !err.ends_with('\n') {
                err.push('\n');
            }
            err.push_str(&error.to_string());
        }

        let locals_map = visible_locals(&self.scope);
        *locals_snapshot.lock().expect("locals snapshot lock") = locals_map.clone();

        let final_answer = self
            .scope
            .get_value::<Dynamic>("FINAL_VAR")
            .filter(|value| !value.is_unit())
            .map(dynamic_to_text);

        let stdout_text = stdout.lock().expect("stdout lock").clone();
        let stderr_text = stderr.lock().expect("stderr lock").clone();
        let locals = dynamic_map_to_btreemap(&locals_map)?;
        let calls = llm_calls.lock().expect("llm_calls lock").clone();

        Ok(ReplResult {
            stdout: stdout_text,
            stderr: stderr_text,
            locals,
            execution_time_secs: started.elapsed().as_secs_f64(),
            llm_calls: calls,
            final_answer,
        })
    }
}

#[derive(Clone, Copy)]
enum HostCallKind {
    Llm,
    Rlm,
}

fn register_host_query(
    engine: &mut Engine,
    name: &str,
    kind: HostCallKind,
    callbacks_addr: usize,
    llm_calls: Arc<Mutex<Vec<crate::types::RlmChatCompletion>>>,
) {
    engine.register_fn(name, move |input: Dynamic| -> RhaiResult {
        let prompt = dynamic_to_prompt(&input)?;
        let completion = unsafe {
            let callbacks_ptr = callbacks_addr as *mut ExecutionCallbacks<'static>;
            match kind {
                HostCallKind::Llm => ((*callbacks_ptr).host_call)(HostCall::Llm(prompt)),
                HostCallKind::Rlm => ((*callbacks_ptr).host_call)(HostCall::Rlm(prompt)),
            }
        }
        .map_err(rhai_runtime_error)?;
        llm_calls
            .lock()
            .expect("llm_calls lock")
            .push(completion.clone());
        Ok(Dynamic::from(completion.response))
    });
}

fn register_host_query_batched(
    engine: &mut Engine,
    name: &str,
    kind: HostCallKind,
    callbacks_addr: usize,
    llm_calls: Arc<Mutex<Vec<crate::types::RlmChatCompletion>>>,
) {
    engine.register_fn(
        name,
        move |inputs: Array| -> std::result::Result<Array, Box<EvalAltResult>> {
            inputs
                .into_iter()
                .map(|input| {
                    let prompt = dynamic_to_prompt(&input)?;
                    let completion = unsafe {
                        let callbacks_ptr = callbacks_addr as *mut ExecutionCallbacks<'static>;
                        match kind {
                            HostCallKind::Llm => {
                                ((*callbacks_ptr).host_call)(HostCall::Llm(prompt))
                            }
                            HostCallKind::Rlm => {
                                ((*callbacks_ptr).host_call)(HostCall::Rlm(prompt))
                            }
                        }
                    }
                    .map_err(rhai_runtime_error)?;
                    llm_calls
                        .lock()
                        .expect("llm_calls lock")
                        .push(completion.clone());
                    Ok(Dynamic::from(completion.response))
                })
                .collect()
        },
    );
}

fn register_tools(
    engine: &mut Engine,
    tools: ToolRegistry,
    llm_calls: Arc<Mutex<Vec<crate::types::RlmChatCompletion>>>,
    _callbacks_addr: usize,
) {
    for name in tools.descriptions().into_keys() {
        let registry = tools.clone();
        let tool_name = name.clone();
        let llm_calls = Arc::clone(&llm_calls);
        engine.register_fn(name.as_str(), move |input: Dynamic| -> RhaiResult {
            let value = dynamic_to_json(&input).map_err(rhai_runtime_error)?;
            let output = registry
                .call(&tool_name, &value)
                .map_err(rhai_runtime_error)?;
            llm_calls
                .lock()
                .expect("llm_calls lock")
                .push(tool_completion(&tool_name, value.clone(), output.clone()));
            Ok(json_to_dynamic(output))
        });
    }
}

fn visible_locals(scope: &Scope<'_>) -> Map {
    let mut locals = Map::new();
    for (name, _constant, value) in scope.iter_raw() {
        if name.starts_with('_')
            || matches!(
                name,
                "context"
                    | "history"
                    | "llm_query"
                    | "llm_query_batched"
                    | "rlm_query"
                    | "rlm_query_batched"
                    | "SHOW_VARS"
            )
        {
            continue;
        }
        locals.insert(name.into(), value.clone());
    }
    locals
}

fn dynamic_to_prompt(value: &Dynamic) -> std::result::Result<Prompt, Box<EvalAltResult>> {
    match dynamic_to_json(value) {
        Ok(Value::String(text)) => Ok(Prompt::from(text)),
        Ok(other) => Ok(Prompt::from(other)),
        Err(error) => Err(rhai_runtime_error(error)),
    }
}

fn dynamic_to_json(value: &Dynamic) -> Result<Value> {
    if value.is_unit() {
        return Ok(Value::Null);
    }
    if let Some(text) = value.clone().try_cast::<ImmutableString>() {
        return Ok(Value::String(text.to_string()));
    }
    if let Some(boolean) = value.clone().try_cast::<bool>() {
        return Ok(Value::Bool(boolean));
    }
    if let Some(integer) = value.clone().try_cast::<i64>() {
        return Ok(Value::Number(Number::from(integer)));
    }
    if let Some(float) = value.clone().try_cast::<f64>() {
        let Some(number) = Number::from_f64(float) else {
            return Err(RlmError::Environment(format!(
                "non-finite float cannot be converted to json: {float}"
            )));
        };
        return Ok(Value::Number(number));
    }
    if let Some(array) = value.clone().try_cast::<Array>() {
        let items = array
            .into_iter()
            .map(|item| dynamic_to_json(&item))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Value::Array(items));
    }
    if let Some(map) = value.clone().try_cast::<Map>() {
        let mut object = serde_json::Map::new();
        for (key, item) in map {
            object.insert(key.to_string(), dynamic_to_json(&item)?);
        }
        return Ok(Value::Object(object));
    }
    Ok(Value::String(format!("{value:?}")))
}

fn json_to_dynamic(value: Value) -> Dynamic {
    match value {
        Value::Null => Dynamic::UNIT,
        Value::Bool(boolean) => Dynamic::from(boolean),
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                Dynamic::from(integer)
            } else if let Some(float) = number.as_f64() {
                Dynamic::from(float)
            } else if let Some(unsigned) = number.as_u64() {
                Dynamic::from(unsigned as i64)
            } else {
                Dynamic::UNIT
            }
        }
        Value::String(text) => Dynamic::from(text),
        Value::Array(items) => {
            Dynamic::from_array(items.into_iter().map(json_to_dynamic).collect())
        }
        Value::Object(object) => {
            let mut map = Map::new();
            for (key, item) in object {
                map.insert(key.into(), json_to_dynamic(item));
            }
            Dynamic::from_map(map)
        }
    }
}

fn dynamic_map_to_btreemap(map: &Map) -> Result<BTreeMap<String, Value>> {
    map.iter()
        .map(|(key, value)| Ok((key.to_string(), dynamic_to_json(value)?)))
        .collect()
}

fn dynamic_to_text(value: Dynamic) -> String {
    match dynamic_to_json(&value) {
        Ok(Value::String(text)) => text,
        Ok(other) => other.to_string(),
        Err(_) => format!("{value:?}"),
    }
}

fn rhai_runtime_error(error: impl ToString) -> Box<EvalAltResult> {
    EvalAltResult::ErrorRuntime(error.to_string().into(), Position::NONE).into()
}

fn tool_completion(name: &str, input: Value, output: Value) -> crate::types::RlmChatCompletion {
    let response = match &output {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    crate::types::RlmChatCompletion {
        root_model: format!("tool:{name}"),
        prompt: Prompt::from(input),
        response,
        usage_summary: crate::types::UsageSummary::default(),
        execution_time_secs: 0.0,
        metadata: Some(serde_json::json!({"tool": name, "output": output})),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use crate::environments::{Environment, FnTool, HostCall, ToolRegistry};
    use crate::types::{Prompt, RlmChatCompletion, UsageSummary};

    use super::LocalEnvironment;

    #[test]
    fn load_context_tracks_context_and_history() {
        let mut env = LocalEnvironment::default();
        Environment::load_context(&mut env, Prompt::from("recursive prompt"))
            .expect("load context should succeed");

        assert_eq!(env.context_count(), 1);
        assert_eq!(env.history_count(), 1);
    }

    #[test]
    fn execute_code_runs_in_persistent_worker_and_makes_host_calls() {
        let mut env = LocalEnvironment::default();
        Environment::load_context(&mut env, Prompt::from("solve it"))
            .expect("load context should succeed");

        let prompts = Arc::new(Mutex::new(Vec::new()));
        let seen_prompts = Arc::clone(&prompts);
        let mut host_call = move |call: HostCall| -> crate::error::Result<RlmChatCompletion> {
            let prompt = match call {
                HostCall::Llm(prompt) | HostCall::Rlm(prompt) => prompt,
            };
            seen_prompts.lock().expect("lock").push(prompt.as_text());
            Ok(RlmChatCompletion {
                root_model: "scripted".to_owned(),
                prompt: prompt.clone(),
                response: "tool-free answer".to_owned(),
                usage_summary: UsageSummary::default(),
                execution_time_secs: 0.0,
                metadata: None,
            })
        };

        let first = Environment::execute_code(
            &mut env,
            r#"let x = 2; print(x); let answer = llm_query("hello from rust"); FINAL_VAR = answer;"#,
            &mut crate::environments::ExecutionCallbacks {
                host_call: &mut host_call,
            },
        )
        .expect("execute_code should succeed");
        let second = Environment::execute_code(
            &mut env,
            "print(x + 1);",
            &mut crate::environments::ExecutionCallbacks {
                host_call: &mut host_call,
            },
        )
        .expect("execute_code should succeed");

        assert_eq!(first.stdout, "2\n");
        assert_eq!(first.final_answer.as_deref(), Some("tool-free answer"));
        assert_eq!(second.stdout, "3\n");
        assert_eq!(
            prompts.lock().expect("lock").as_slice(),
            &["hello from rust".to_owned()]
        );
    }

    #[test]
    fn execute_code_can_call_registered_tools() {
        let mut tools = ToolRegistry::default();
        tools
            .insert(
                "lookup",
                FnTool::new(Some("resolves a value".to_owned()), |input| {
                    Ok(json!({"echo": input}))
                }),
            )
            .expect("tool insert should succeed");

        let mut env = LocalEnvironment::new(tools);
        let mut host_call = |_call: HostCall| -> crate::error::Result<RlmChatCompletion> {
            panic!("tool test should not issue model calls")
        };

        let result = Environment::execute_code(
            &mut env,
            r#"let value = lookup(#{ depth: 2 }); print(value); FINAL_VAR = value;"#,
            &mut crate::environments::ExecutionCallbacks {
                host_call: &mut host_call,
            },
        )
        .expect("execute_code should succeed");

        assert!(result.stdout.contains("depth"));
        assert_eq!(
            result.final_answer.as_deref(),
            Some(r#"{"echo":{"depth":2}}"#)
        );
    }

    #[test]
    fn tool_descriptions_expose_registered_tools() {
        let mut tools = ToolRegistry::default();
        tools
            .insert(
                "lookup",
                FnTool::new(Some("resolves a value".to_owned()), |_input| {
                    Ok(json!("ok"))
                }),
            )
            .expect("tool insert should succeed");

        let env = LocalEnvironment::new(tools);
        assert_eq!(
            env.tool_descriptions().get("lookup"),
            Some(&Some("resolves a value".to_owned()))
        );
    }
}
