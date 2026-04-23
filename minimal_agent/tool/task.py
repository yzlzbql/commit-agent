from __future__ import annotations

from pydantic import BaseModel

from ..types import TaskRequest, ToolCtx, ToolResult, ToolSpec


class TaskArgs(BaseModel):
    description: str
    prompt: str
    subagent_type: str
    task_id: str | None = None
    command: str | None = None
    max_steps: int | None = None


def spec() -> ToolSpec:
    return ToolSpec(name="task", description="Run a bounded subtask with a specialized subagent", input_model=TaskArgs, execute=run)


def run(ctx: ToolCtx, args: TaskArgs) -> ToolResult:
    if ctx.run_subtask is None:
        raise RuntimeError("subtask runner unavailable")
    if ctx.available_agents is not None:
        allowed = {item["name"] for item in ctx.available_agents}
        if args.subagent_type not in allowed:
            raise ValueError(f"unknown subagent_type: {args.subagent_type}")
    result = ctx.run_subtask(
        TaskRequest(
            description=args.description,
            prompt=args.prompt,
            subagent_type=args.subagent_type,
            task_id=args.task_id,
            command=args.command,
            max_steps=args.max_steps,
        )
    )
    output = "\n".join(
        [
            f"task_id: {result.session_id}",
            "",
            "<task_result>",
            result.summary.strip(),
            "</task_result>",
        ]
    )
    return ToolResult(
        title=args.description,
        output=output,
        metadata={"session_id": result.session_id, "status": result.status, "agent": result.agent},
    )
