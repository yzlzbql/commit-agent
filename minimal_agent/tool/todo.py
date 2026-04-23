from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ..types import TodoItem, ToolCtx, ToolResult, ToolSpec


class TodoReadArgs(BaseModel):
    pass


class TodoWriteArgs(BaseModel):
    todos: list[TodoItem] = Field(default_factory=list)


def read_spec() -> ToolSpec:
    return ToolSpec(name="todoread", description="Read the current todo list", input_model=TodoReadArgs, execute=read_run)


def write_spec() -> ToolSpec:
    return ToolSpec(name="todowrite", description="Replace the current todo list", input_model=TodoWriteArgs, execute=write_run)


def read_run(ctx: ToolCtx, _args: TodoReadArgs) -> ToolResult:
    todos = ctx.store.load_todo(ctx.st.id)
    return ToolResult(
        title=f"{len([item for item in todos if item.status != 'completed'])} todos",
        output=json.dumps([item.model_dump() for item in todos], ensure_ascii=False, indent=2),
        metadata={"todos": [item.model_dump() for item in todos]},
    )


def write_run(ctx: ToolCtx, args: TodoWriteArgs) -> ToolResult:
    ctx.st.todo = list(args.todos)
    ctx.store.save_todo(ctx.st.id, ctx.st.todo)
    return ToolResult(
        title=f"{len([item for item in args.todos if item.status != 'completed'])} todos",
        output=json.dumps([item.model_dump() for item in args.todos], ensure_ascii=False, indent=2),
        metadata={"todo_items": [item.model_dump() for item in args.todos]},
    )
