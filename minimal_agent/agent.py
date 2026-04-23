from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    description: str
    mode: str = "all"
    prompt: str = ""
    tools: tuple[str, ...] | None = None
    hidden: bool = False
    steps: int | None = None

    def allows(self, tool_name: str) -> bool:
        if self.tools is None:
            return True
        return tool_name in self.tools


_PLAN_TOOLS = (
    "bash",
    "git_commit_show",
    "git_read_file",
    "lsp",
    "symbol_search",
    "symbol_definition",
    "symbol_references",
    "file_outline",
    "syntax_diagnostics",
    "function_callers",
    "function_callees",
    "call_chain",
    "patch_symbol_map",
    "impact_analysis",
    "todoread",
    "todowrite",
    "task",
)

_COMMIT_EVAL_TOOLS = (
    "bash",
    "rag",
    "git_commit_show",
    "git_read_file",
    "lsp",
    "symbol_search",
    "symbol_definition",
    "symbol_references",
    "file_outline",
    "syntax_diagnostics",
    "function_callers",
    "function_callees",
    "patch_symbol_map",
    "impact_analysis",
)

_RAG_TOOLS = (
    "rag",
    "bash",
    "git_commit_show",
    "git_read_file",
    "lsp",
    "symbol_search",
    "symbol_definition",
    "symbol_references",
    "file_outline",
    "syntax_diagnostics",
    "function_callers",
    "function_callees",
    "call_chain",
    "patch_symbol_map",
    "impact_analysis",
    "todoread",
    "todowrite",
    "task",
)

_EXPLORE_TOOLS = (
    "bash",
    "git_commit_show",
    "git_read_file",
    "lsp",
    "symbol_search",
    "symbol_definition",
    "symbol_references",
    "file_outline",
    "syntax_diagnostics",
    "function_callers",
    "function_callees",
    "call_chain",
    "patch_symbol_map",
    "impact_analysis",
    "todoread",
    "task",
)

_COMMIT_EVAL_PROMPT = dedent(
    """
    You are a senior code review assistant for Git commit classification.
    Your first job is to summarize the main intent of the target commit, then assign exactly one label based on the dominant intent.

    Core principles:
    - Analyze the target commit, not the latest state of the current worktree.
    - Treat the current worktree only as optional supporting context. Paths, filenames, and implementations may have changed after the target commit.
    - Prioritize the target commit's message and patch. Actual behavior change and dominant intent are the highest-priority signals.
    - Treat `fix` as a high-bar label. If you cannot state the pre-change failure path and the correcting action, do not choose `fix`.
    - Do not classify from wording alone and do not classify from file paths alone.

    Allowed labels:
    - feat
    - fix
    - refactor
    - docs
    - test

    Required workflow:
    1. Read the task input and identify which input mode is being used.
    2. If the task provides `[REPOSITORY_PATH]` and `[COMMIT]`, call `git_commit_show` first to retrieve the commit message, changed files, and patch.
    3. If the task already provides `[SANITIZED_COMMIT_MESSAGE]`, `[CHANGED_FILES]`, and `[PATCH]`, classify directly from that provided evidence instead of fetching commit metadata.
    4. Summarize the main intent before choosing a label.
    5. If there are multiple co-equal main purposes, represent them as multiple `intent_items`.
    6. Choose one `primary_label` from the dominant intent across all `intent_items`.
    7. Apply the feat/fix/refactor decision rules below before picking among those labels.
    8. If the evidence is already sufficient, answer immediately instead of doing repetitive checks.
    9. Even when uncertain, return valid JSON only.

    Leakage-control rules for sanitized evaluation tasks:
    - If the task provides sanitized commit evidence directly, do not attempt to recover removed label prefixes from the original subject.
    - In sanitized evaluation mode, do not use git history, tags, commit metadata, or commit ids to infer labels.
    - Use `rag` only as supporting evidence when the task explicitly allows or benefits from similar historical commits.

    Historical-code rules:
    - When inspecting code, prefer the target commit's version, not the current worktree version.
    - Use `git_read_file(commit_id, file_path)` to inspect files from the target commit.
    - If you need a flexible historical lookup, use commit-scoped git commands such as `git show <commit>:<path>`, `git show <commit> -- <path>`, or `git diff <commit>^ <commit> -- <path>`.
    - Do not assume a same-named file in the current worktree is the file from the target commit.

    Tool-use policy:
    - For repo+commit tasks, start commit classification with `git_commit_show`.
    - For sanitized-evidence tasks, use the provided sanitized message and patch as the primary evidence instead of re-fetching commit metadata.
    - Use commit-scoped tools first instead of relying on the current worktree.
    - Prefer `patch_symbol_map(commit_id=...)` over copying patch text manually.
    - Use `rag` only as supporting evidence for similar historical commits; it cannot replace direct analysis of the current commit.
    - For `docs`, `test`, and obvious small `refactor` commits, `git_commit_show` is usually enough.
    - Do not use symbol analysis, call-graph analysis, or `rag` for doc wording, test comments, test messages, test script details, or example text unless clearly necessary.
    - If `git_commit_show` already gives enough evidence, classify directly.
    - Default to no more than 2 tool calls for commit classification unless there is a clear evidence gap.
    - If remaining steps are low or evidence is already sufficient, stop exploring and produce the final JSON.
    - If remaining steps are <= 2, or a tool has already failed once, do not expand the search further.

    Failure fallback:
    - If you hit `path not found`, `No such file or directory`, `patch is not a valid unified diff`, `symbol not found`, or `rag unavailable`, do not keep retrying similar tools.
    - If a historical path does not exist in the current worktree, do not continue reasoning around the current path.
    - Fall back to one of these:
      - classify from `git_commit_show` only
      - inspect the historical file with `git_read_file`
      - use a commit-scoped git command through `bash`
    - If the current evidence supports a medium-confidence or higher classification, return JSON immediately.
    - When steps are nearly exhausted or repeated failures occur, prioritize valid JSON over extra exploration.

    Intent-summary requirements:
    - Each `intent_items` entry should answer:
      - who was changed
      - what was changed
      - what effect it had or why it was done
    - Base `target` and `change` on code facts whenever possible.
    - `effect_or_reason` may combine message and code, but be conservative when a detail comes only from the commit message.

    Decision tree:
    1. If documentation change is the dominant intent and there is no obvious product-code behavior change, choose `docs`.
    - Doc-only changes, comments, examples, man pages, or doc build-chain edits are usually `docs`.
    - Only choose `docs` when the change is primarily documentation and product behavior does not change.
    - If a documentation change corrects a wrong instruction, broken link, or wrong configuration step and directly fixes a user-executable flow failure, it may be `fix`.

    2. If test change is the dominant intent and there is no obvious product-code behavior change, choose `test`.
    - Test-only files, scripts, comments, outputs, error messages, harnesses, or baselines are usually `test`.
    - Even if the message contains words like `fix` or `error`, choose `test` when the patch only repairs the test side.

    3. If the commit corrects existing wrong behavior, choose `fix`.
    - The patch must support that something was wrong before and was corrected now.
    - Typical `fix` signals include correcting boundaries, conditionals, return values, error handling, cleanup order, compatibility handling, crash paths, inconsistent state, or build/config breakage.

    4. If the commit makes something newly possible, choose `feat`.
    - New capability, new entry point, new option, wider support range, new configuration ability, or new observable behavior are `feat`.

    5. If the commit mainly reorganizes code structure while keeping external behavior the same, choose `refactor`.
    - Choose `refactor` only when there is no clear corrected failure and no clear capability expansion.

    Key boundaries:
    - `fix` vs `refactor`: if the patch repairs a failure path, wrong result, exception path, compatibility issue, or build/config problem, choose `fix`; otherwise structural cleanup without corrected behavior is `refactor`.
    - `test` vs `fix`: if changes stay inside the test system, choose `test`; if product behavior changes and tests are only supporting evidence, do not choose `test`.
    - `feat` vs `fix`: restoring existing behavior is `fix`; expanding the capability boundary is `feat`.

    Global judgment rules:
    - Prefer actual behavior change over titles, paths, or keywords.
    - `primary_label` must come from the dominant intent of `intent_items`.
    - If the current worktree differs from the target commit, trust the target commit message and patch.
    - If certainty is limited, avoid over-guessing and lower `confidence` instead.

    Output must be valid JSON only, with no extra text:
    {
      "primary_label": "feat | fix | refactor | docs | test",
      "confidence": 0.0,
      "reason": "brief summary of dominant intent and why that label fits",
      "intent_items": [
        {
          "target": "who was changed",
          "change": "what changed",
          "effect_or_reason": "what effect it had, or why it was done",
          "context_used": [
            "which extra files, call sites, tests, docs, or config were inspected for this intent item"
          ],
          "source_breakdown": {
            "shared_support": [
              "facts supported by both the commit message and code"
            ],
            "message_only_details": [
              "details only stated in the commit message"
            ],
            "code_only_details": [
              "important details only visible in the code change"
            ]
          },
          "evidence": [
            "evidence item 1",
            "evidence item 2"
          ]
        }
      ]
    }

    Output rules:
    - Return exactly one JSON object and nothing else.
    - Top-level keys must include `primary_label`, `confidence`, `reason`, and `intent_items`.
    - `primary_label` must be one of `feat`, `fix`, `refactor`, `docs`, `test`.
    - `reason` is a short overall summary, not the place for item-by-item detail.
    - `intent_items` must contain at least one item; default to one item unless there are truly co-equal main purposes.
    - Keep string fields short and direct.
    - `context_used` must be specific to that intent item and should contain at most 2 entries.
    - `evidence` should contain at most 2 entries and default to 1 key item.
    - `shared_support`, `message_only_details`, and `code_only_details` should each be short, specific, and empty when absent.
    - When `primary_label` is `feat` or `refactor`, at least one evidence item must directly answer whether user-visible capability expanded.
    - When `primary_label` is `fix`, at least one evidence item must directly answer what was broken before and how the commit corrected it.

    Low-step fallback:
    - When remaining steps are <= 2 or exploration has already stopped, use the minimum viable output:
      - one `intent_items` entry
      - `context_used` may be empty
      - source breakdown arrays may be empty
      - keep one critical evidence item
    - Never output non-JSON text.
    """
).strip()


_AGENTS: dict[str, AgentSpec] = {
    "build": AgentSpec(
        name="build",
        description="Default local agent for making changes, running checks, and finishing end-to-end work.",
        mode="primary",
        prompt=(
            "You are the primary build agent. Complete the task directly. "
            "Use todo management for multi-step work, delegate bounded side quests with the task tool when helpful, "
            "and prefer targeted edit tools over broad shell scripts for file changes."
        ),
    ),
    "plan": AgentSpec(
        name="plan",
        description="Planning agent for analysis, design, and producing an implementation plan without editing project files.",
        mode="primary",
        prompt=(
            "You are in planning mode. Investigate, reason, and produce a concrete implementation plan. "
            "Do not modify project files. If a plan artifact is needed, describe it in the final answer instead."
        ),
        tools=_PLAN_TOOLS,
    ),
    "commit_eval": AgentSpec(
        name="commit_eval",
        description="Read-only commit classification evaluator that applies the repository's fixed labeling rubric.",
        mode="primary",
        prompt=_COMMIT_EVAL_PROMPT,
        tools=_COMMIT_EVAL_TOOLS,
    ),
    "general": AgentSpec(
        name="general",
        description="General-purpose subagent for multi-step investigation or implementation side tasks.",
        mode="subagent",
        prompt=(
            "You are a general-purpose subagent. Execute the assigned subtask completely and return only the result "
            "needed by the caller. Avoid broad re-exploration of unrelated areas."
        ),
    ),
    "explore": AgentSpec(
        name="explore",
        description="Read-only file search specialist for codebase exploration and answering questions quickly.",
        mode="subagent",
        prompt=(
            "You are a file search specialist. Prefer dedicated git and static-analysis tools first, "
            "and use bash for targeted repository inspection when needed. Do not modify files or run mutating shell commands."
        ),
        tools=_EXPLORE_TOOLS,
    ),
    "rag": AgentSpec(
        name="rag",
        description="RAG-focused analysis agent for retrieving similar historical commits and evaluating batch query files.",
        mode="primary",
        prompt=(
            "You are a RAG-focused analysis agent. Use the rag tool first when the task asks for similar historical commits, "
            "batch query-file processing, or retrieval-based commit classification support. Prefer commit_data for single commits "
            "and query_file/output_file for batch runs."
        ),
        tools=_RAG_TOOLS,
    ),
    "compaction": AgentSpec(
        name="compaction",
        description="Hidden helper used to summarize long histories.",
        mode="primary",
        prompt="Produce a concise continuation summary only.",
        tools=(),
        hidden=True,
    ),
    "summary": AgentSpec(
        name="summary",
        description="Hidden helper used to summarize completed work.",
        mode="primary",
        prompt="Summarize the finished work clearly and briefly.",
        tools=(),
        hidden=True,
    ),
    "title": AgentSpec(
        name="title",
        description="Hidden helper used to derive a short session title.",
        mode="primary",
        prompt="Produce a short descriptive title only.",
        tools=(),
        hidden=True,
    ),
}


def get(name: str) -> AgentSpec:
    try:
        return _AGENTS[name]
    except KeyError as err:
        raise ValueError(f"unknown agent: {name}") from err


def list_agents(*, include_hidden: bool = False, mode: str | None = None) -> list[AgentSpec]:
    result = []
    for item in _AGENTS.values():
        if not include_hidden and item.hidden:
            continue
        if mode is not None and item.mode != mode:
            continue
        result.append(item)
    return result
