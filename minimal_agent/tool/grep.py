from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

from pydantic import BaseModel

from ..policy import ensure_path, is_git_path
from ..types import ToolCtx, ToolResult, ToolSpec


class GrepArgs(BaseModel):
    pattern: str
    path: str = "."
    include: str | None = None
    limit: int = 100


def spec() -> ToolSpec:
    return ToolSpec(name="grep", description="Search file contents with a regex pattern", input_model=GrepArgs, execute=run)


def run(ctx: ToolCtx, args: GrepArgs) -> ToolResult:
    root = (ctx.project.cwd / args.path).resolve()
    ensure_path(ctx.project.root, root)
    if root.is_file():
        rows = _grep_file(root, args)
    elif shutil.which("rg"):
        rows = _rg(root, args)
    else:
        rows = _walk(ctx.project.root, root, args)
    return ToolResult(
        title=args.pattern,
        output="\n".join(rows) if rows else "No files found",
        metadata={"matches": len(rows), "path": ctx.project.relpath(root)},
    )


def _rg(root: Path, args: GrepArgs) -> list[str]:
    cmd = ["rg", "-nH", "--hidden", "--no-messages", "--glob", "!.git", "--glob", "!.git/**", args.pattern]
    if args.include:
        cmd.extend(["--glob", args.include])
    cmd.append(str(root))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode not in {0, 1, 2}:
        raise RuntimeError(res.stderr.strip() or "ripgrep failed")
    return res.stdout.splitlines()[: args.limit]


def _walk(project_root: Path, root: Path, args: GrepArgs) -> list[str]:
    pat = re.compile(args.pattern)
    rows: list[str] = []
    for item in root.rglob("*"):
        if len(rows) >= args.limit:
            break
        if item.is_dir():
            continue
        if is_git_path(project_root, item):
            continue
        if args.include and not item.match(args.include):
            continue
        try:
            for idx, line in enumerate(item.read_text(encoding="utf-8").splitlines(), start=1):
                if pat.search(line):
                    rows.append(f"{item}:{idx}:{line}")
                    if len(rows) >= args.limit:
                        break
        except (UnicodeDecodeError, re.error):
            continue
    return rows


def _grep_file(path: Path, args: GrepArgs) -> list[str]:
    if shutil.which("rg"):
        res = subprocess.run(["rg", "-nH", "--no-messages", args.pattern, str(path)], capture_output=True, text=True, check=False)
        if res.returncode not in {0, 1, 2}:
            raise RuntimeError(res.stderr.strip() or "ripgrep failed")
        return res.stdout.splitlines()[: args.limit]
    pat = re.compile(args.pattern)
    rows: list[str] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if pat.search(line):
            rows.append(f"{path}:{idx}:{line}")
            if len(rows) >= args.limit:
                break
    return rows
