from __future__ import annotations

from pydantic import BaseModel, Field

from ..types import ToolCtx, ToolResult, ToolSpec
from .edit import EditArgs, run as edit_run


class MultiEditArgs(BaseModel):
    file_path: str
    edits: list[EditArgs] = Field(default_factory=list)


def spec() -> ToolSpec:
    return ToolSpec(
        name="multiedit",
        description="Apply multiple exact-string edit operations to one file in order",
        input_model=MultiEditArgs,
        execute=run,
    )


def run(ctx: ToolCtx, args: MultiEditArgs) -> ToolResult:
    if not args.edits:
        raise ValueError("edits cannot be empty")
    last: ToolResult | None = None
    results = []
    for item in args.edits:
        last = edit_run(
            ctx,
            EditArgs(
                file_path=args.file_path,
                old_string=item.old_string,
                new_string=item.new_string,
                replace_all=item.replace_all,
            ),
        )
        results.append(last.metadata)
    assert last is not None
    last.metadata["results"] = results
    return last
