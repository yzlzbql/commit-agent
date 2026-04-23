from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ..policy import ensure_path, is_git_path
from ..types import ToolCtx, ToolResult, ToolSpec


class ReadArgs(BaseModel):
    file_path: str
    offset: int = 1
    limit: int = 200


def spec() -> ToolSpec:
    return ToolSpec(name="read", description="Read a file or list a directory", input_model=ReadArgs, execute=run)


def run(ctx: ToolCtx, args: ReadArgs) -> ToolResult:
    path = (ctx.project.cwd / args.file_path).resolve()
    ensure_path(ctx.project.root, path)
    if not path.exists():
        raise ValueError(f"path not found: {path}")
    if path.is_dir():
        items = sorted(
            item.name + ("/" if item.is_dir() else "")
            for item in path.iterdir()
            if not is_git_path(ctx.project.root, item)
        )
        start = max(args.offset - 1, 0)
        chunk = items[start : start + args.limit]
        return ToolResult(
            title=f"Read {ctx.project.relpath(path)}",
            output="\n".join(chunk),
            metadata={"path": ctx.project.relpath(path)},
        )
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ValueError(f"binary file rejected: {path}")
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    start = max(args.offset - 1, 0)
    chunk = lines[start : start + args.limit]
    body = "\n".join(f"{idx}: {line}" for idx, line in enumerate(chunk, start=args.offset))
    return ToolResult(
        title=f"Read {ctx.project.relpath(path)}",
        output=body,
        metadata={"path": ctx.project.relpath(path)},
    )
