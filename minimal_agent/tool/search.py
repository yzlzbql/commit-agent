from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from pydantic import BaseModel

from ..policy import ensure_path, is_git_path
from ..types import ToolCtx, ToolResult, ToolSpec


class SearchArgs(BaseModel):
    query: str
    mode: str = "content"
    root: str = "."
    limit: int = 50


def spec() -> ToolSpec:
    return ToolSpec(name="search", description="Search file paths or file contents", input_model=SearchArgs, execute=run)


def run(ctx: ToolCtx, args: SearchArgs) -> ToolResult:
    root = (ctx.project.cwd / args.root).resolve()
    ensure_path(ctx.project.root, root)
    if not root.exists():
        raise ValueError(f"path not found: {root}")
    if shutil.which("rg"):
        out = _rg(root, args)
    else:
        out = _walk(ctx.project.root, root, args)
    return ToolResult(
        title=f"Search {args.mode}",
        output="\n".join(out),
        metadata={"count": len(out)},
    )


def _rg(root: Path, args: SearchArgs) -> list[str]:
    if args.mode == "path":
        res = subprocess.run(
            ["rg", "--files", "--glob", "!.git", "--glob", "!.git/**", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )
        rows = [line for line in res.stdout.splitlines() if args.query.lower() in line.lower()]
        return rows[: args.limit]
    res = subprocess.run(
        ["rg", "-n", "--hidden", "--no-messages", "--glob", "!.git", "--glob", "!.git/**", args.query, str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.splitlines()[: args.limit]


def _walk(project_root: Path, root: Path, args: SearchArgs) -> list[str]:
    result = []
    for path in root.rglob("*"):
        if len(result) >= args.limit:
            break
        if path.is_dir():
            continue
        if is_git_path(project_root, path):
            continue
        if args.mode == "path":
            if args.query.lower() in path.as_posix().lower():
                result.append(path.as_posix())
            continue
        try:
            for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if args.query.lower() in line.lower():
                    result.append(f"{path.as_posix()}:{idx}:{line}")
                    if len(result) >= args.limit:
                        break
        except UnicodeDecodeError:
            continue
    return result
