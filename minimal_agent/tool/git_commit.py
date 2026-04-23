from __future__ import annotations

import json
import subprocess

from pydantic import BaseModel

from ..sanitize import sanitize_commit_body, sanitize_commit_message, sanitize_commit_subject
from ..types import ToolCtx, ToolResult, ToolSpec


class GitCommitShowArgs(BaseModel):
    commit: str


def spec() -> ToolSpec:
    return ToolSpec(
        name="git_commit_show",
        description=(
            "Show one commit as structured JSON with sanitized message, changed files, and unified patch. "
            "Use this as the preferred first step for commit inspection or commit classification instead of repeating "
            "multiple git show variants."
        ),
        input_model=GitCommitShowArgs,
        execute=run,
    )


def run(ctx: ToolCtx, args: GitCommitShowArgs) -> ToolResult:
    commit = args.commit.strip()
    if not commit:
        raise ValueError("E_INPUT_INVALID: commit must not be empty")
    if not (ctx.project.root / ".git").exists():
        raise ValueError(f"E_INPUT_INVALID: not a git repository: {ctx.project.root}")

    canonical = _git(
        ctx,
        ["git", "rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}"],
        "invalid commit",
    ).strip()
    meta_raw = _git(
        ctx,
        ["git", "show", "--no-patch", "--format=%H%x00%s%x00%b", "--end-of-options", canonical],
        "failed to read commit metadata",
    )
    parts = meta_raw.split("\x00", 2)
    if len(parts) != 3:
        raise RuntimeError("E_INTERNAL: failed to parse git commit metadata")
    _, raw_subject, raw_body = parts
    raw_body = raw_body.rstrip()
    subject = sanitize_commit_subject(raw_subject)
    body = sanitize_commit_body(raw_body)
    message = sanitize_commit_message(raw_subject if not raw_body else f"{raw_subject}\n\n{raw_body}")

    files_raw = _git(
        ctx,
        ["git", "diff-tree", "--root", "--no-commit-id", "--name-status", "-r", "--find-renames", "--end-of-options", canonical],
        "failed to list changed files",
    )
    files = [_parse_name_status(line) for line in files_raw.splitlines() if line.strip()]

    patch = _git(
        ctx,
        ["git", "show", "--format=", "--patch", "--no-ext-diff", "--no-color", "--find-renames", "--end-of-options", canonical],
        "failed to read commit patch",
    )

    data = {
        "ok": True,
        "tool": "git_commit_show",
        "version": "v0",
        "data": {
            "commit": canonical,
            "subject": subject,
            "body": body,
            "message": message,
            "files": files,
            "patch": patch,
        },
    }
    return ToolResult(
        title=f"git_commit_show {canonical[:12]}",
        output=json.dumps(data, ensure_ascii=False, indent=2),
        metadata={
            "ok": True,
            "tool": "git_commit_show",
            "commit": canonical,
            "subject": subject,
            "file_count": len(files),
        },
    )


def _git(ctx: ToolCtx, cmd: list[str], error_prefix: str) -> str:
    res = subprocess.run(
        cmd,
        cwd=ctx.project.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        stderr = res.stderr.strip() or res.stdout.strip() or error_prefix
        if res.returncode == 128:
            raise ValueError(f"E_INPUT_INVALID: {stderr}")
        raise RuntimeError(f"E_INTERNAL: {stderr}")
    return res.stdout


def _parse_name_status(line: str) -> dict[str, str | None]:
    parts = line.split("\t")
    if len(parts) < 2:
        return {"status": parts[0].strip() if parts else "?", "path": line.strip(), "old_path": None}
    status = parts[0].strip()
    if status.startswith(("R", "C")) and len(parts) >= 3:
        return {"status": status, "path": parts[2], "old_path": parts[1]}
    return {"status": status, "path": parts[1], "old_path": None}
