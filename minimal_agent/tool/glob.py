from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel

from ..policy import ensure_path, is_git_path
from ..types import ToolCtx, ToolResult, ToolSpec


class GlobArgs(BaseModel):
    pattern: str
    path: str = "."
    limit: int = 100


def spec() -> ToolSpec:
    return ToolSpec(name="glob", description="Match files by glob pattern", input_model=GlobArgs, execute=run)


def run(ctx: ToolCtx, args: GlobArgs) -> ToolResult:
    root = (ctx.project.cwd / args.path).resolve()
    ensure_path(ctx.project.root, root)
    if not root.is_dir():
        raise ValueError(f"path is not a directory: {root}")
    matches: list[tuple[int, str]] = []
    for item in root.rglob("*"):
        if not item.exists():
            continue
        if is_git_path(ctx.project.root, item):
            continue
        rel = item.relative_to(root).as_posix()
        if fnmatch(rel, args.pattern) or fnmatch(item.name, args.pattern):
            matches.append((int(item.stat().st_mtime), str(item.resolve())))
    matches.sort(key=lambda item: item[0], reverse=True)
    rows = [path for _, path in matches[: args.limit]]
    if len(matches) > args.limit:
        rows.extend(["", f"(truncated: showing first {args.limit} results of {len(matches)})"])
    return ToolResult(
        title=args.pattern,
        output="\n".join(rows) if rows else "No files found",
        metadata={"count": len(matches), "truncated": len(matches) > args.limit},
    )
