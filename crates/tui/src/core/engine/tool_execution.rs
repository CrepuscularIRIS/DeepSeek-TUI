//! Low-level tool execution helpers for the engine turn loop.
//!
//! This module keeps the mechanics of MCP dispatch, execution locking, and
//! parallel-tool fanout out of `engine.rs`; the turn loop still owns planning,
//! approval, and how tool results are written back into session state.

use std::{fs::OpenOptions, io::Write, sync::Arc, time::Duration};

use super::*;

/// RAII guard that pauses the TUI's terminal-state ownership for the duration
/// of an interactive tool, then restores it on drop.
///
/// Background: interactive tools (anything that needs the raw TTY — external
/// editor, `exec_shell` with stdin, etc.) need the TUI to leave alt-screen,
/// disable raw mode, and release mouse capture so the child sees a normal
/// terminal. The TUI listens for `Event::PauseEvents` / `Event::ResumeEvents`
/// and runs `pause_terminal` / `resume_terminal` in response.
///
/// Earlier code sent `PauseEvents` before tool execution and `ResumeEvents`
/// after. That worked on the happy path, but if the tool's future was dropped
/// — Ctrl+C cancellation, sub-agent abort, parent task cancelled while the
/// tool was awaiting — the second `await` never reached and `ResumeEvents`
/// was never sent. It also let interactive children start before the UI had
/// actually left alt-screen/raw mode. Both failures strand the TUI in a
/// regular shell scrollback: the parent shell scrollbar takes over, mouse
/// wheel scrolls the host terminal instead of the transcript, and the TUI
/// renders at the bottom of cooked-mode output.
///
/// `Drop` runs synchronously and can't await, so we first use `try_send` on a
/// **clone of the event channel** to push `ResumeEvents` non-blockingly. If the
/// channel is full we enqueue the resume on the active Tokio runtime instead of
/// dropping it; otherwise a burst of engine events can strand the UI in the
/// paused terminal state.
pub(super) struct InteractiveTerminalGuard {
    tx: Option<mpsc::Sender<Event>>,
}

impl InteractiveTerminalGuard {
    /// Send `PauseEvents` and arm the guard. If `interactive` is false the
    /// guard is a no-op — `Drop` will skip the resume.
    pub(super) async fn engage(tx: mpsc::Sender<Event>, interactive: bool) -> Self {
        if !interactive {
            return Self { tx: None };
        }
        // Best-effort: if the receiver is gone the TUI has already shut down
        // and there's nothing to restore. If the event is delivered, wait for
        // the UI to actually release the terminal before starting the child.
        let ack = Arc::new(tokio::sync::Notify::new());
        match tx
            .send(Event::PauseEvents {
                ack: Some(ack.clone()),
            })
            .await
        {
            Ok(()) => {
                if tokio::time::timeout(Duration::from_millis(750), ack.notified())
                    .await
                    .is_err()
                {
                    tracing::warn!(
                        target: "engine.tool_execution",
                        "InteractiveTerminalGuard: timed out waiting for terminal pause ack; \
                         continuing with interactive tool"
                    );
                }
            }
            Err(err) => {
                tracing::debug!(
                    target: "engine.tool_execution",
                    ?err,
                    "InteractiveTerminalGuard: event channel closed before PauseEvents"
                );
            }
        }
        Self { tx: Some(tx) }
    }
}

impl Drop for InteractiveTerminalGuard {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            match tx.try_send(Event::ResumeEvents) {
                Ok(()) => {}
                Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                    match tokio::runtime::Handle::try_current() {
                        Ok(handle) => {
                            handle.spawn(async move {
                                if let Err(err) = tx.send(event).await {
                                    tracing::warn!(
                                        target: "engine.tool_execution",
                                        ?err,
                                        "InteractiveTerminalGuard: async send(ResumeEvents) failed; \
                                         terminal may stay in paused state until the next \
                                         pause/resume cycle"
                                    );
                                }
                            });
                        }
                        Err(err) => {
                            tracing::warn!(
                                target: "engine.tool_execution",
                                ?err,
                                "InteractiveTerminalGuard: event channel full and no Tokio runtime \
                                 available to queue ResumeEvents; terminal may stay paused until \
                                 the next pause/resume cycle"
                            );
                        }
                    }
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    tracing::debug!(
                        target: "engine.tool_execution",
                        "InteractiveTerminalGuard: event channel closed before ResumeEvents"
                    );
                }
            }
        }
    }
}

pub(super) fn emit_tool_audit(event: serde_json::Value) {
    let Some(path) = std::env::var_os("DEEPSEEK_TOOL_AUDIT_LOG") else {
        return;
    };
    let line = match serde_json::to_string(&event) {
        Ok(line) => line,
        Err(_) => return,
    };
    let path = PathBuf::from(path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

fn deferred_tool_schema_hydration(
    tool_name: &str,
    tool_input: &serde_json::Value,
    schema: &serde_json::Value,
) -> Option<ToolResult> {
    let schema_obj = schema.as_object()?;
    let input_obj = tool_input.as_object()?;
    let properties = schema_obj.get("properties")?.as_object()?;

    let mut expected_fields: Vec<String> = properties.keys().cloned().collect();
    expected_fields.sort();

    let required_fields: Vec<String> = schema_obj
        .get("required")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let mut received_fields: Vec<String> = input_obj.keys().cloned().collect();
    received_fields.sort();

    let expected_lookup: std::collections::BTreeSet<&str> =
        expected_fields.iter().map(String::as_str).collect();
    let received_lookup: std::collections::BTreeSet<&str> =
        received_fields.iter().map(String::as_str).collect();

    let missing_fields: Vec<String> = required_fields
        .iter()
        .filter(|field| !received_lookup.contains(field.as_str()))
        .cloned()
        .collect();

    let unexpected_fields: Vec<String> = received_fields
        .iter()
        .filter(|field| !expected_lookup.contains(field.as_str()))
        .cloned()
        .collect();

    if missing_fields.is_empty() && unexpected_fields.is_empty() {
        return None;
    }

    let likely_corrections: Vec<serde_json::Value> = unexpected_fields
        .iter()
        .filter_map(|unexpected| {
            suggest_field_correction(unexpected, properties).map(|expected| {
                serde_json::json!({
                    "from": unexpected,
                    "to": expected,
                })
            })
        })
        .collect();

    let mut content = format!(
        "Tool `{tool_name}` was deferred and has now been loaded.\n\n\
         The tool was not executed.\n\n\
         Expected fields:\n"
    );
    for field in &expected_fields {
        let required = if required_fields.iter().any(|required| required == field) {
            "required"
        } else {
            "optional"
        };
        content.push_str(&format!("  - {field} ({required})\n"));
    }
    if expected_fields.is_empty() {
        content.push_str("  - none\n");
    }

    content.push_str("\nReceived fields:\n");
    if received_fields.is_empty() {
        content.push_str("  - none\n");
    } else {
        for field in &received_fields {
            content.push_str(&format!("  - {field}\n"));
        }
    }

    content.push_str("\nMissing required fields:\n");
    if missing_fields.is_empty() {
        content.push_str("  - none\n");
    } else {
        for field in &missing_fields {
            content.push_str(&format!("  - {field}\n"));
        }
    }

    content.push_str("\nUnexpected fields:\n");
    if unexpected_fields.is_empty() {
        content.push_str("  - none\n");
    } else {
        for field in &unexpected_fields {
            content.push_str(&format!("  - {field}\n"));
        }
    }

    if !likely_corrections.is_empty() {
        content.push_str("\nLikely corrections:\n");
        for correction in &likely_corrections {
            let from = correction
                .get("from")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let to = correction.get("to").and_then(|v| v.as_str()).unwrap_or("");
            content.push_str(&format!("  - {from} -> {to}\n"));
        }
    }

    content.push_str("\nRetry with the loaded schema.");

    Some(ToolResult::error(content).with_metadata(serde_json::json!({
        "kind": "deferred_tool_schema_hydration",
        "tool_name": tool_name,
        "expected_fields": expected_fields,
        "required_fields": required_fields,
        "received_fields": received_fields,
        "missing_fields": missing_fields,
        "unexpected_fields": unexpected_fields,
        "likely_corrections": likely_corrections,
    })))
}

fn suggest_field_correction(
    unexpected: &str,
    properties: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    let unexpected_normalized = normalize_field_name(unexpected);
    let unexpected_tokens = tokenise_field_name(unexpected);

    let mut best: Option<(usize, String)> = None;
    for (candidate, schema) in properties {
        let candidate_normalized = normalize_field_name(candidate);
        let candidate_tokens = tokenise_field_name(candidate);

        let mut score = edit_distance(&unexpected_normalized, &candidate_normalized);

        if candidate_normalized.contains(&unexpected_normalized)
            || unexpected_normalized.contains(&candidate_normalized)
        {
            score = score.saturating_sub(2);
        }

        if obvious_alias_match(&unexpected_tokens, &candidate_tokens, schema) {
            score = 0;
        }

        match &best {
            Some((best_score, _)) if score >= *best_score => {}
            _ => best = Some((score, candidate.clone())),
        }
    }

    best.and_then(|(score, candidate)| (score <= 3).then_some(candidate))
}

fn obvious_alias_match(
    unexpected_tokens: &[String],
    candidate_tokens: &[String],
    schema: &serde_json::Value,
) -> bool {
    let description = schema
        .get("description")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();

    let unexpected_has_oldish = unexpected_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "old" | "before" | "previous" | "source" | "from"
        )
    });
    let unexpected_has_newish = unexpected_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "new" | "after" | "replacement" | "target" | "to"
        )
    });

    let candidate_is_search_like = candidate_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "search" | "find" | "match" | "pattern" | "query"
        )
    }) || description.contains("search")
        || description.contains("find")
        || description.contains("match");

    let candidate_is_replace_like = candidate_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "replace" | "replacement" | "substitute" | "write" | "content"
        )
    }) || description.contains("replace")
        || description.contains("substitute");

    (unexpected_has_oldish && candidate_is_search_like)
        || (unexpected_has_newish && candidate_is_replace_like)
}

fn normalize_field_name(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn tokenise_field_name(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.to_ascii_lowercase())
        .collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let mut prev: Vec<usize> = (0..=b.chars().count()).collect();
    let mut curr = vec![0; prev.len()];

    for (i, a_ch) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, b_ch) in b.chars().enumerate() {
            let substitution = prev[j] + usize::from(a_ch != b_ch);
            let insertion = curr[j] + 1;
            let deletion = prev[j + 1] + 1;
            curr[j + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b.chars().count()]
}

impl Engine {
    pub(super) async fn execute_mcp_tool_with_pool(
        pool: Arc<AsyncMutex<McpPool>>,
        name: &str,
        input: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let mut pool = pool.lock().await;
        let result = pool
            .call_tool(name, input)
            .await
            .map_err(|e| ToolError::execution_failed(format!("MCP tool failed: {e}")))?;
        let content = serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string());
        Ok(ToolResult::success(content))
    }

    pub(super) async fn execute_parallel_tool(
        &mut self,
        input: serde_json::Value,
        tool_registry: Option<&crate::tools::ToolRegistry>,
        tool_exec_lock: Arc<RwLock<()>>,
    ) -> Result<ToolResult, ToolError> {
        let calls = parse_parallel_tool_calls(&input)?;
        let mcp_pool = if calls.iter().any(|(tool, _)| McpPool::is_mcp_tool(tool)) {
            Some(self.ensure_mcp_pool().await?)
        } else {
            None
        };
        let Some(registry) = tool_registry else {
            return Err(ToolError::not_available(
                "tool registry unavailable for multi_tool_use.parallel",
            ));
        };

        let mut tasks = FuturesUnordered::new();
        for (tool_name, tool_input) in calls {
            if tool_name == MULTI_TOOL_PARALLEL_NAME {
                return Err(ToolError::invalid_input(
                    "multi_tool_use.parallel cannot call itself",
                ));
            }
            if McpPool::is_mcp_tool(&tool_name) {
                if !mcp_tool_is_parallel_safe(&tool_name) {
                    return Err(ToolError::invalid_input(format!(
                        "Tool '{tool_name}' is an MCP tool and cannot run in parallel. \
                         Allowed MCP tools: list_mcp_resources, list_mcp_resource_templates, \
                         mcp_read_resource, read_mcp_resource, mcp_get_prompt."
                    )));
                }
            } else {
                let Some(spec) = registry.get(&tool_name) else {
                    return Err(ToolError::not_available(format!(
                        "tool '{tool_name}' is not registered"
                    )));
                };
                if !spec.is_read_only() {
                    return Err(ToolError::invalid_input(format!(
                        "Tool '{tool_name}' is not read-only and cannot run in parallel"
                    )));
                }
                if spec.approval_requirement() != ApprovalRequirement::Auto {
                    return Err(ToolError::invalid_input(format!(
                        "Tool '{tool_name}' requires approval and cannot run in parallel"
                    )));
                }
                if !spec.supports_parallel() {
                    return Err(ToolError::invalid_input(format!(
                        "Tool '{tool_name}' does not support parallel execution"
                    )));
                }
            }

            let registry_ref = registry;
            let lock = tool_exec_lock.clone();
            let tx_event = self.tx_event.clone();
            let mcp_pool = mcp_pool.clone();
            tasks.push(async move {
                let result = Engine::execute_tool_with_lock(
                    lock,
                    true,
                    false,
                    tx_event,
                    tool_name.clone(),
                    tool_input.clone(),
                    Some(registry_ref),
                    mcp_pool,
                    None,
                )
                .await;
                (tool_name, result)
            });
        }

        let mut results = Vec::new();
        while let Some((tool_name, result)) = tasks.next().await {
            match result {
                Ok(output) => {
                    let mut error = None;
                    if !output.success {
                        error = Some(output.content.clone());
                    }
                    results.push(ParallelToolResultEntry {
                        tool_name,
                        success: output.success,
                        content: output.content,
                        error,
                    });
                }
                Err(err) => {
                    let message = format!("{err}");
                    results.push(ParallelToolResultEntry {
                        tool_name,
                        success: false,
                        content: format!("Error: {message}"),
                        error: Some(message),
                    });
                }
            }
        }

        ToolResult::json(&ParallelToolResult { results })
            .map_err(|e| ToolError::execution_failed(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_tool_with_lock(
        lock: Arc<RwLock<()>>,
        supports_parallel: bool,
        interactive: bool,
        tx_event: mpsc::Sender<Event>,
        tool_name: String,
        tool_input: serde_json::Value,
        registry: Option<&crate::tools::ToolRegistry>,
        mcp_pool: Option<Arc<AsyncMutex<McpPool>>>,
        context_override: Option<crate::tools::ToolContext>,
    ) -> Result<ToolResult, ToolError> {
        if let Some(registry) = registry
            && let Some(tool) = registry.get(&tool_name)
            && tool.defer_loading()
            && let Some(result) =
                deferred_tool_schema_hydration(&tool_name, &tool_input, &tool.input_schema())
        {
            return Ok(result);
        }

        let _guard = if supports_parallel {
            ToolExecGuard::Read(lock.read().await)
        } else {
            ToolExecGuard::Write(lock.write().await)
        };

        // RAII pause/resume: ensures `Event::ResumeEvents` always fires on
        // drop, even if the tool future is cancelled mid-await. See
        // `InteractiveTerminalGuard` doc-comment for the regression this
        // closes (parent terminal scrollback hijacking the TUI after a
        // cancelled interactive tool).
        let _terminal = InteractiveTerminalGuard::engage(tx_event, interactive).await;

        if McpPool::is_mcp_tool(&tool_name) {
            if let Some(pool) = mcp_pool {
                Engine::execute_mcp_tool_with_pool(pool, &tool_name, tool_input).await
            } else {
                Err(ToolError::not_available(format!(
                    "tool '{tool_name}' is not registered"
                )))
            }
        } else if let Some(registry) = registry {
            registry
                .execute_full_with_context(&tool_name, tool_input, context_override.as_ref())
                .await
        } else {
            Err(ToolError::not_available(format!(
                "tool '{tool_name}' is not registered"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    /// Tests in this module mutate `DEEPSEEK_TOOL_AUDIT_LOG` which is
    /// process-global; serialise through this guard so the parallel
    /// runner doesn't observe interleaved env mutations.
    static AUDIT_TEST_GUARD: Mutex<()> = Mutex::new(());

    fn audit_test_guard() -> std::sync::MutexGuard<'static, ()> {
        AUDIT_TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner())
    }

    struct DeferredToolProbe {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl crate::tools::spec::ToolSpec for DeferredToolProbe {
        fn name(&self) -> &'static str {
            "edit_file"
        }

        fn description(&self) -> &'static str {
            "Probe tool for deferred-tool schema hydration tests."
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "search": { "type": "string" },
                    "replace": { "type": "string" }
                },
                "required": ["path", "search", "replace"]
            })
        }

        fn capabilities(&self) -> Vec<crate::tools::spec::ToolCapability> {
            vec![
                crate::tools::spec::ToolCapability::WritesFiles,
                crate::tools::spec::ToolCapability::Sandboxable,
            ]
        }

        fn defer_loading(&self) -> bool {
            true
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _context: &crate::tools::spec::ToolContext,
        ) -> Result<crate::tools::spec::ToolResult, crate::tools::spec::ToolError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(crate::tools::spec::ToolResult::success("executed"))
        }
    }

    #[tokio::test]
    async fn terminal_guard_queues_resume_when_event_channel_is_full() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send(Event::status("filler")).expect("fill channel");

        drop(InteractiveTerminalGuard { tx: Some(tx) });

        assert!(matches!(rx.recv().await, Some(Event::Status { .. })));
        let resumed = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("queued resume event")
            .expect("event channel still open");
        assert!(matches!(resumed, Event::ResumeEvents));
    }

    #[tokio::test]
    async fn terminal_guard_waits_for_pause_ack_before_returning() {
        let (tx, mut rx) = mpsc::channel(4);
        let task = tokio::spawn(InteractiveTerminalGuard::engage(tx, true));

        let event = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("pause event")
            .expect("event channel still open");
        let ack = match event {
            Event::PauseEvents { ack: Some(ack) } => ack,
            other => panic!("expected PauseEvents with ack, got {other:?}"),
        };

        tokio::task::yield_now().await;
        assert!(!task.is_finished(), "guard returned before pause ack");

        ack.notify_one();
        let guard = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("guard returned after ack")
            .expect("guard task joined");

        drop(guard);
        let resumed = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("resume event")
            .expect("event channel still open");
        assert!(matches!(resumed, Event::ResumeEvents));
    }

    #[tokio::test]
    async fn deferred_tool_with_guessed_args_returns_schema_hydration_hint() {
        let calls = Arc::new(AtomicUsize::new(0));
        let tool = Arc::new(DeferredToolProbe {
            calls: calls.clone(),
        });
        let workspace = tempfile::tempdir().expect("tempdir");
        let context = crate::tools::spec::ToolContext::new(workspace.path().to_path_buf());
        let mut registry = crate::tools::ToolRegistry::new(context);
        registry.register(tool);

        let lock = Arc::new(tokio::sync::RwLock::new(()));
        let (tx, _rx) = mpsc::channel(1);
        let result = Engine::execute_tool_with_lock(
            lock,
            false,
            false,
            tx,
            "edit_file".to_string(),
            json!({
                "path": "src/foo.rs",
                "old_string": "...",
                "new_string": "..."
            }),
            Some(&registry),
            None,
            None,
        )
        .await
        .expect("deferred tool should return a synthetic result");

        assert!(!result.success);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(result.content.contains("The tool was not executed"));
        assert!(result.content.contains("Expected fields"));
        assert!(result.content.contains("old_string -> search"));
        assert!(result.content.contains("new_string -> replace"));
    }

    #[test]
    fn emit_tool_audit_writes_jsonl_line_when_env_var_set() {
        let _g = audit_test_guard();
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("audit.log");
        // SAFETY: serialised by the guard above.
        unsafe {
            std::env::set_var("DEEPSEEK_TOOL_AUDIT_LOG", &path);
        }

        emit_tool_audit(json!({
            "event": "tool.spillover",
            "tool_id": "call-abc",
            "tool_name": "exec_shell",
            "path": "/tmp/foo.txt",
        }));
        emit_tool_audit(json!({
            "event": "tool.result",
            "tool_id": "call-xyz",
            "success": true,
        }));

        let body = std::fs::read_to_string(&path).expect("audit log written");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "two emits → two lines");

        // Each line round-trips as JSON, has the expected event key.
        let first: serde_json::Value = serde_json::from_str(lines[0]).expect("first line is JSON");
        assert_eq!(
            first.get("event").and_then(|v| v.as_str()),
            Some("tool.spillover")
        );
        assert_eq!(
            first.get("tool_id").and_then(|v| v.as_str()),
            Some("call-abc")
        );

        let second: serde_json::Value =
            serde_json::from_str(lines[1]).expect("second line is JSON");
        assert_eq!(
            second.get("event").and_then(|v| v.as_str()),
            Some("tool.result")
        );

        // SAFETY: cleanup under the guard.
        unsafe {
            std::env::remove_var("DEEPSEEK_TOOL_AUDIT_LOG");
        }
    }

    #[test]
    fn emit_tool_audit_is_noop_when_env_var_unset() {
        let _g = audit_test_guard();
        // SAFETY: serialised by the guard above.
        unsafe {
            std::env::remove_var("DEEPSEEK_TOOL_AUDIT_LOG");
        }
        // Should not panic and should not create any file. We can't
        // assert "no file written" without knowing where one might be
        // written, but the contract is "do nothing", which we verify
        // by ensuring the call returns without error.
        emit_tool_audit(json!({"event": "noop", "x": 1}));
        // Successful return is the assertion.
    }

    #[test]
    fn emit_tool_audit_creates_parent_directory() {
        let _g = audit_test_guard();
        let tmp = tempfile::tempdir().expect("tempdir");
        // Path with a parent that doesn't exist yet — the writer
        // should create it.
        let nested = tmp.path().join("nested").join("dir").join("audit.log");
        // SAFETY: serialised by the guard above.
        unsafe {
            std::env::set_var("DEEPSEEK_TOOL_AUDIT_LOG", &nested);
        }
        emit_tool_audit(json!({"event": "test"}));
        assert!(nested.exists(), "writer should mkdir -p the parent chain");

        // SAFETY: cleanup under the guard.
        unsafe {
            std::env::remove_var("DEEPSEEK_TOOL_AUDIT_LOG");
        }
    }
}
