from __future__ import annotations

import json

from .agent import AgentSpec
from .types import RunInput, SessionState, StepRecord

_MAX_RECENT_EVENTS = 8
_MAX_TOOL_OUTPUT_CHARS = 10000
_MAX_FINAL_OUTPUT_CHARS = 10000
_MAX_RECENT_CONTEXT_CHARS = 12000


def _tool_text(tools: list[dict[str, str]]) -> str:
    return "\n".join(f"- {item['name']}: {item['description']}" for item in tools) or "- none"


def _agent_text(items: list[dict[str, str]]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item['name']}: {item['description']}" for item in items)


def _budget_text(run: RunInput, st: SessionState) -> str:
    cap = run.max_steps
    remaining = "unlimited" if cap is None else str(max(cap - st.step_count, 0))
    return "\n".join(
        [
            "Step budget:",
            f"- Steps used: {st.step_count}",
            f"- Max steps: {cap if cap is not None else 'unlimited'}",
            f"- Steps remaining: {remaining}",
        ]
    )


def _event_text(recs: list[StepRecord]) -> str:
    parts = []
    recent = recs[-_MAX_RECENT_EVENTS:]
    skipped = max(len(recs) - len(recent), 0)
    if skipped:
        parts.append(f"... {skipped} earlier events omitted ...")
    for rec in recent:
        if rec.kind == "tool":
            name = rec.payload.get("name", "tool")
            args = rec.payload.get("args", {})
            title = rec.payload.get("title", "")
            out = rec.payload.get("output", "")
            block = [f"tool {name}"]
            if args:
                block.append(f"args: {json.dumps(args, ensure_ascii=False, sort_keys=True)}")
            if title:
                block.append(f"title: {title}")
            if out:
                block.append("output:")
                block.append(_truncate(out, _MAX_TOOL_OUTPUT_CHARS))
            parts.append("\n".join(block))
            continue
        if rec.kind == "final":
            parts.append(f"candidate\n{_truncate(rec.payload.get('text', ''), _MAX_FINAL_OUTPUT_CHARS)}")
            continue
        if rec.kind == "error":
            parts.append(f"error\n{rec.payload.get('message', '')}")
    text = "\n\n".join(parts)
    return _truncate(text, _MAX_RECENT_CONTEXT_CHARS)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... [{omitted} chars omitted] ..."


def _system_prompt(
    run: RunInput,
    tools: list[dict[str, str]],
    lsp_ok: bool,
    *,
    agent: AgentSpec,
    available_agents: list[dict[str, str]],
    force_plain_text: bool,
) -> str:
    sections = [
        "\n".join(
            [
                "You are a coding agent running in a local CLI runtime.",
                f"Current agent: {agent.name}",
                agent.prompt.strip(),
            ]
        ).strip(),
        "\n".join(
            [
                "Environment:",
                f"- Current cwd: {run.cwd}",
                f"- LSP available: {'yes' if lsp_ok else 'no'}",
                f"- Parent session: {run.parent_id or 'none'}",
            ]
        ),
        "\n".join(
            [
                "Workflow:",
                "1. Classify the task as read-only, investigation, planning, or editing.",
                "2. Keep the todo list current when the work has multiple distinct steps.",
                "3. Prefer the most specific tool for the next action, and treat static-analysis tools as the default first choice for code understanding.",
                "4. Stop gathering context once the answer or required change is already supported by evidence.",
                "5. Use the task tool only for bounded side work that can be delegated cleanly.",
            ]
        ),
        "\n".join(
            [
                "Tool policy:",
                "- For commit-inspection tasks, use git_commit_show first to get structured message, files, and patch in one step.",
                "- Use git_read_file to inspect a file as it existed in a historical commit when that path no longer exists in the current worktree.",
                "- When you need exact historical code, prefer commit-scoped access such as git_read_file or bash commands like `git show <commit>:<path>` over reading the current worktree version.",
                "- For symbol lookup, definitions, references, file structure, diagnostics, call flow, patch mapping, and impact estimation, use the static-analysis tools first.",
                "- Prefer symbol_search, symbol_definition, symbol_references, file_outline, and syntax_diagnostics for structured code analysis.",
                "- Use function_callers, function_callees, call_chain, patch_symbol_map, and impact_analysis for call-graph and blast-radius analysis.",
                "- For commit tasks, prefer patch_symbol_map(commit_id=...) over copying a raw patch string into tool arguments.",
                "- Use bash for git, filesystem inspection, broad text search, builds, tests, or other shell-only actions.",
                "- Use write, edit, multiedit, or patch for file changes.",
                "- Do not use bash as the first step for code understanding if a static-analysis tool can answer the question.",
                "- Use lsp only as a lower-level fallback when the static-analysis tools cannot express the query or returned insufficient detail.",
                "- Use todoread/todowrite when the work benefits from an explicit checklist.",
                "- Prefer bash workdir over `cd <dir> && <command>` patterns.",
            ]
        ),
        "\n".join(
            [
                "Static-analysis priority order:",
                "- If the task is about classifying or reviewing a git commit, start with git_commit_show, then call patch_symbol_map with commit_id if symbol mapping would help.",
                "- If you need file context from the target commit and the current worktree path is missing or renamed, use git_read_file with commit_id instead of inspecting the current worktree path.",
                "- If you need a historical file or path-specific diff and the dedicated tools are insufficient, use bash to run a commit-scoped git command rather than reading the current worktree path.",
                "- If the task asks what symbol exists, where it is defined, who references it, or what a file contains structurally, start with symbol_search, symbol_definition, symbol_references, or file_outline.",
                "- If the task asks about callers, callees, call paths, changed symbols, or blast radius, start with function_callers, function_callees, call_chain, patch_symbol_map, or impact_analysis.",
                "- Only fall back to bash-based text search after the structured tools fail, are unavailable, or the problem is intentionally broad and text-oriented.",
            ]
        ),
        "\n".join(
            [
                "Loop prevention:",
                "- Do not repeat an identical tool call if the previous result already answers the question.",
                "- Do not keep reformatting or re-running equivalent bash commands to confirm the same fact.",
                "- After a useful tool result, continue from that result instead of restarting the investigation.",
                "- Do not fall back to bash or lsp just to re-derive facts that a static-analysis tool already returned.",
                "- For single-answer classification or review tasks, once the main conclusion is supported, respond immediately instead of collecting marginal extra context.",
                "- Watch the remaining step budget and avoid exploratory calls when the budget is low.",
                "- If only one step remains, prefer a plain-text response unless one last tool call is strictly necessary.",
                "- If a read-only or planning task is answered, summarize the evidence and respond in plain text.",
            ]
        ),
        "\n".join(
            [
                "Completion rules:",
                "- Return tool_call only when another concrete action is required and the tool name is listed under Available tools.",
                "- When the task is complete, respond with plain text only.",
                "- For read-only questions, the plain-text response should directly answer the user and cite the key evidence briefly.",
                "- For editing tasks, the plain-text response should summarize what changed and what was verified.",
                "- For subagents, return only the result needed by the caller, not extra conversation.",
            ]
        ),
        "\n".join(
            [
                "Current mode:",
                "- Recent tool attempts were blocked as repetitive. Do not request any more tools.",
                "- Use the evidence already collected and respond with plain text only.",
            ]
        )
        if force_plain_text
        else "",
        "\n".join(
            [
                "Available subagents:",
                _agent_text(available_agents),
            ]
        ),
        "\n".join(
            [
                "Available tools:",
                _tool_text(tools),
            ]
        ),
    ]
    return "\n\n".join(item for item in sections if item.strip())


def build_messages(
    run: RunInput,
    st: SessionState,
    recs: list[StepRecord],
    tools: list[dict[str, str]],
    lsp_ok: bool,
    *,
    agent: AgentSpec,
    available_agents: list[dict[str, str]],
    force_plain_text: bool = False,
) -> list[dict[str, str]]:
    sys = _system_prompt(
        run,
        tools,
        lsp_ok,
        agent=agent,
        available_agents=available_agents,
        force_plain_text=force_plain_text,
    )
    todo = "\n".join(f"- [{item.status}] {item.content}" for item in st.todo) or "- [pending] no todo"
    usr = "\n".join(
        [
            f"Task: {run.task}",
            f"Session title: {st.title or run.title or 'Untitled'}",
            "",
            "Todo:",
            todo,
            "",
            _budget_text(run, st),
            "",
            "Summary:",
            st.summary or "No summary yet.",
            "",
            "Recent context:",
            _event_text(recs) or "No recent events.",
        ]
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]
