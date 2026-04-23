from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ..policy import ensure_path, is_git_path
from ..types import ToolCtx, ToolResult, ToolSpec


class ListArgs(BaseModel):
    path: str = "."
    limit: int = 200


def spec() -> ToolSpec:
    return ToolSpec(name="list", description="List files and directories in a directory", input_model=ListArgs, execute=run)


def run(ctx: ToolCtx, args: ListArgs) -> ToolResult:
    path = (ctx.project.cwd / args.path).resolve()
    ensure_path(ctx.project.root, path)
    if not path.exists():
        raise ValueError(f"path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"path is not a directory: {path}")
    items = sorted(
        item.name + ("/" if item.is_dir() else "")
        for item in path.iterdir()
        if not is_git_path(ctx.project.root, item)
    )
    chunk = items[: args.limit]
    if len(items) > args.limit:
        chunk.append("")
        chunk.append(f"(truncated: showing first {args.limit} entries of {len(items)})")
    return ToolResult(
        title=f"List {ctx.project.relpath(path)}",
        output="\n".join(chunk),
        metadata={"path": ctx.project.relpath(path), "count": len(items)},
    )
