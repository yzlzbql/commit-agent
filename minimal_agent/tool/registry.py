from __future__ import annotations

from . import bash, edit, git_commit, git_read_file, lsp, multiedit, patch, rag, static_analysis, task, todo, write
from ..sanitize import scrub_tool_result
from ..types import ToolCall, ToolCtx, ToolResult


class ToolRegistry:
    def __init__(self):
        self.specs = {
            item.name: item
            for item in [
                bash.spec(),
                git_commit.spec(),
                git_read_file.spec(),
                patch.spec(),
                write.spec(),
                edit.spec(),
                multiedit.spec(),
                lsp.spec(),
                rag.spec(),
                *static_analysis.specs(),
                todo.read_spec(),
                todo.write_spec(),
                task.spec(),
            ]
        }

    def schemas(self, allowed: set[str] | None = None) -> list[dict[str, object]]:
        return [
            {
                "name": item.name,
                "description": item.description,
                "schema": item.input_model.model_json_schema(),
            }
            for item in self.specs.values()
            if allowed is None or item.name in allowed
        ]

    def execute(self, ctx: ToolCtx, call: ToolCall) -> ToolResult:
        spec = self.specs.get(call.name)
        if spec is None:
            raise ValueError(f"unknown tool: {call.name}")
        args = spec.input_model.model_validate(call.args)
        return scrub_tool_result(spec.execute(ctx, args))

    def lsp_available(self, path: object) -> bool:
        return bool(path)
