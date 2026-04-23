from __future__ import annotations

from pathlib import PurePosixPath
import subprocess

from pydantic import BaseModel

from ..types import ToolCtx, ToolResult, ToolSpec


class GitReadFileArgs(BaseModel):
    commit_id: str
    file_path: str
    offset: int = 1
    limit: int = 200


def spec() -> ToolSpec:
    return ToolSpec(
        name="git_read_file",
        description=(
            "Read one file as it existed in a specific historical commit. "
            "Use this when a commit changed a path that no longer exists in the current worktree."
        ),
        input_model=GitReadFileArgs,
        execute=run,
    )


def run(ctx: ToolCtx, args: GitReadFileArgs) -> ToolResult:
    commit = _verify_commit(ctx, args.commit_id.strip())
    git_path = _normalize_git_path(args.file_path)
    blob = _read_blob(ctx, commit, git_path)
    if b"\x00" in blob:
        raise ValueError(f"binary file rejected: {git_path}@{commit}")
    text = blob.decode("utf-8", errors="replace")
    lines = text.splitlines()
    start = max(args.offset - 1, 0)
    chunk = lines[start : start + args.limit]
    body = "\n".join(f"{idx}: {line}" for idx, line in enumerate(chunk, start=args.offset))
    return ToolResult(
        title=f"Read {git_path}@{commit[:12]}",
        output=body,
        metadata={"commit": commit, "path": git_path},
    )


def _verify_commit(ctx: ToolCtx, commit_id: str) -> str:
    if not commit_id:
        raise ValueError("E_INPUT_INVALID: commit_id must not be empty")
    res = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{commit_id}^{{commit}}"],
        cwd=ctx.project.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        msg = res.stderr.strip() or res.stdout.strip() or "invalid commit"
        raise ValueError(f"E_INPUT_INVALID: {msg}")
    return res.stdout.strip()


def _normalize_git_path(file_path: str) -> str:
    raw = (file_path or "").strip()
    if not raw:
        raise ValueError("E_INPUT_INVALID: file_path must not be empty")
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise ValueError(f"E_INPUT_INVALID: file_path must be repo-relative: {raw}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"E_INPUT_INVALID: invalid file_path: {raw}")
    return path.as_posix()


def _read_blob(ctx: ToolCtx, commit: str, git_path: str) -> bytes:
    spec = f"{commit}:{git_path}"
    res = subprocess.run(
        ["git", "show", "--no-textconv", "--end-of-options", spec],
        cwd=ctx.project.root,
        capture_output=True,
        check=False,
    )
    if res.returncode != 0:
        msg = (res.stderr or res.stdout).decode("utf-8", errors="replace").strip() or "path not found in commit"
        raise ValueError(f"E_INPUT_INVALID: {msg}")
    return res.stdout
