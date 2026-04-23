#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import json
import re

from rich.console import Console
from rich.table import Table
import typer

from minimal_agent.agent import get as get_agent
from minimal_agent.analysis import AnalysisService
from minimal_agent.config import AgentConfig
from minimal_agent.lsp.server import LspPool
from minimal_agent.model import OpenAIModelAdapter
from minimal_agent.project import Project
from minimal_agent.runtime.loop import run as run_loop
from minimal_agent.session.store import SessionStore
from minimal_agent.tool.registry import ToolRegistry
from minimal_agent.types import RunInput


app = typer.Typer(help="Run the minimal agent from a task file.")
console = Console()
_REPOSITORY_PATH = re.compile(r"^\[REPOSITORY_PATH\]\s*$", re.MULTILINE)


def _repository_path_from_task(task: str, base_dir: Path) -> Path | None:
    lines = task.splitlines()
    for idx, line in enumerate(lines):
        if not _REPOSITORY_PATH.match(line.strip()):
            continue
        if idx + 1 >= len(lines):
            return None
        raw = lines[idx + 1].strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path.resolve()
    return None


def _state_root_for_repository(repo_path: Path, script_dir: Path) -> Path:
    key = hashlib.sha1(str(repo_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return script_dir / ".agent" / "external" / key


@app.command()
def main(
    task_file: Path = typer.Argument(Path("task.txt"), exists=True, dir_okay=False, readable=True),
    cwd: Path = typer.Option(Path("."), "--cwd", exists=True, file_okay=False, dir_okay=True),
    resume: str | None = typer.Option(None, "--resume"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    agent: str = typer.Option("build", "--agent"),
) -> None:
    task = task_file.read_text(encoding="utf-8").strip()
    if not task:
        raise typer.BadParameter(f"task file is empty: {task_file}")

    script_dir = Path(__file__).resolve().parent
    task_repo = _repository_path_from_task(task, task_file.parent)
    project_cwd = task_repo or cwd
    if not project_cwd.exists() or not project_cwd.is_dir():
        raise typer.BadParameter(f"repository path from task is not a directory: {project_cwd}")
    state_root = _state_root_for_repository(project_cwd, script_dir) if task_repo is not None else None
    project = Project.load(project_cwd, state_root=state_root)
    cfg = AgentConfig.load(
        project.cwd,
        env_search_paths=[script_dir, project.root, project.cwd],
        config_search_paths=[project.root, project.cwd],
    )
    if max_steps is None:
        max_steps = cfg.limits.max_steps
    get_agent(agent)
    store = SessionStore(project.sessions_dir)
    st = store.load(resume) if resume else None
    tools = ToolRegistry()
    lsp = LspPool(cfg, project.root)
    analysis = AnalysisService(cfg, project, lsp)
    model = OpenAIModelAdapter(cfg.model)
    data = run_loop(
        run=RunInput(task=task, cwd=project.cwd, resume=resume, max_steps=max_steps, agent=agent),
        cfg=cfg,
        project=project,
        store=store,
        model=model,
        tools=tools,
        lsp=lsp,
        analysis=analysis,
        st=st,
        emit=_print_event,
    )

    table = Table(title="minimal-agent-cli")
    table.add_column("field")
    table.add_column("value")
    table.add_row("task_file", str(task_file))
    table.add_row("status", data.status)
    table.add_row("session_id", data.session_id)
    table.add_row("agent", data.agent)
    table.add_row("next_action", data.next_action)
    table.add_row("files", ", ".join(data.files) or "-")
    table.add_row("verification", "ok" if data.verification.ok else "failed")
    console.print(table)
    console.print(data.summary)
    if data.status != "finished":
        raise typer.Exit(code=1)


def _print_event(event: dict[str, object]) -> None:
    kind = str(event.get("type", "event"))
    prefix = ""
    if event.get("subtask"):
        prefix = f"[subtask:{event.get('subtask_agent', '?')}] "
    if kind == "session":
        console.rule(f"{prefix}Session {event['session_id']}")
        console.print("task file: loaded")
        console.print(f"task: {event['task']}")
        console.print(f"cwd: {event['cwd']}")
        console.print(f"agent: {event.get('agent', 'build')}")
        return
    if kind == "subtask_start":
        console.rule(f"Subtask Start: {event.get('description', '')}")
        console.print(f"agent: {event.get('agent', '')}")
        return
    if kind == "subtask_finish":
        console.rule(f"Subtask Finish: {event.get('description', '')}")
        console.print(f"agent: {event.get('agent', '')}")
        console.print(f"status: {event.get('status', '')}")
        console.print(f"task_id: {event.get('task_id', '')}")
        return
    if kind == "todo":
        console.rule(f"{prefix}Todo")
        for item in event.get("items", []):
            if isinstance(item, dict):
                console.print(f"- [{item.get('status', 'pending')}] {item.get('content', '')}")
        return
    if kind == "summary":
        console.rule(f"{prefix}Compacted Summary")
        console.print(event.get("text", ""))
        return
    if kind == "reasoning":
        console.rule(f"{prefix}Reasoning")
        console.print(event.get("text", ""))
        return
    if kind == "tool_call":
        console.rule(f"{prefix}Tool Call: {event['name']}")
        console.print(json.dumps(event.get("args", {}), ensure_ascii=False, indent=2))
        return
    if kind == "tool_result":
        console.rule(f"{prefix}Tool Result: {event['name']}")
        console.print(event.get("output", ""))
        return
    if kind == "final":
        console.rule(f"{prefix}Final Draft")
        console.print(event.get("text", ""))
        return
    if kind == "verify_start":
        console.rule(f"{prefix}Verify")
        console.print("running verification")
        return
    if kind == "verify_result":
        console.print(f"verification: {'ok' if event.get('ok') else 'failed'}")
        checks = event.get("checks", [])
        failures = event.get("failures", [])
        if checks:
            console.print("checks:")
            for item in checks:
                console.print(f"- {item}")
        if failures:
            console.print("failures:")
            for item in failures:
                console.print(f"- {item}")
        return
    if kind == "error":
        console.rule(f"{prefix}Error")
        console.print(event.get("message", ""))


if __name__ == "__main__":
    app()
