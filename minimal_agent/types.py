from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field


Phase = Literal["analyze", "act", "verify", "finish", "stop"]
Priority = Literal["high", "medium", "low"]
TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]
AgentName = Literal["build", "plan", "commit_eval", "general", "explore", "rag", "compaction", "summary", "title"]


class RunInput(BaseModel):
    task: str
    cwd: Path
    resume: str | None = None
    max_steps: int | None = None
    agent: str = "build"
    parent_id: str | None = None
    title: str | None = None


class VerifyResult(BaseModel):
    ok: bool
    checks: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    next_action: str = ""


class TodoItem(BaseModel):
    content: str
    status: TodoStatus = "pending"
    priority: Priority = "medium"


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    title: str
    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class LspRequest(BaseModel):
    operation: Literal[
        "go_to_definition",
        "find_references",
        "hover",
        "document_symbol",
        "workspace_symbol",
        "go_to_implementation",
    ]
    file_path: str = "."
    line: int = 1
    character: int = 1
    query: str | None = None


class LspResult(BaseModel):
    ok: bool
    items: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class StepRecord(BaseModel):
    kind: str
    ts: float
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    description: str
    prompt: str
    subagent_type: str
    task_id: str | None = None
    command: str | None = None
    max_steps: int | None = None


class SessionState(BaseModel):
    id: str
    cwd: Path
    agent: str = "build"
    title: str = ""
    parent_id: str | None = None
    phase: Phase = "analyze"
    step_count: int = 0
    retry_count: int = 0
    repeat_count: int = 0
    todo: list[TodoItem] = Field(default_factory=list)
    summary: str = ""
    candidate: str = ""
    last_error: str = ""
    verification: VerifyResult | None = None


class RunResult(BaseModel):
    status: Literal["finished", "stopped"]
    summary: str
    files: list[str] = Field(default_factory=list)
    verification: VerifyResult
    session_id: str
    next_action: str = ""
    agent: str = "build"


@dataclass(slots=True)
class ToolCtx:
    run: RunInput
    st: SessionState
    project: Any
    cfg: Any
    store: Any
    lsp: Any
    analysis: Any | None = None
    emit: Callable[[dict[str, Any]], None] | None = None
    run_subtask: Callable[[TaskRequest], RunResult] | None = None
    available_agents: list[dict[str, str]] | None = None


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    execute: Callable[[ToolCtx, BaseModel], ToolResult]
