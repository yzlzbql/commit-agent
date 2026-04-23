from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re

from .types import RunInput, SessionState, ToolCall


_BAD = [
    re.compile(r"\brm\s+-rf\s+/"),
    re.compile(r"(^|\s)sudo(\s|$)"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"),
]
_REDIRECT_SINKS = {Path("/dev/null"), Path("/dev/stdout"), Path("/dev/stderr")}


def ensure_path(root: Path, path: str | Path) -> None:
    item = Path(path).resolve()
    if root != item and root not in item.parents:
        raise ValueError(f"path outside project boundary: {item}")
    if is_git_path(root, item):
        raise ValueError(f"path inside .git is not readable: {item}")


def is_git_path(root: Path, path: str | Path) -> bool:
    item = Path(path).resolve()
    git_dir = (root / ".git").resolve()
    return item == git_dir or git_dir in item.parents


def ensure_shell(root: Path, cwd: Path, cmd: str) -> None:
    ensure_path(root, cwd)
    if any(pat.search(cmd) for pat in _BAD):
        raise ValueError("dangerous shell command rejected")
    if ">" in cmd:
        parts = [item.strip() for item in cmd.split(">")[1:] if item.strip()]
        for item in parts:
            tgt = item.split()[0]
            if tgt.startswith("&"):
                continue
            resolved = (cwd / tgt).resolve()
            if resolved in _REDIRECT_SINKS:
                continue
            ensure_path(root, resolved)


def ensure_steps(run: RunInput, st: SessionState) -> None:
    cap = run.max_steps or 0
    if cap and st.step_count >= cap:
        raise RuntimeError("step budget exceeded")


def ensure_retry(limit: int, st: SessionState) -> None:
    if st.retry_count >= limit:
        raise RuntimeError("retry budget exceeded")


def ensure_repeat(limit: int, calls: list[ToolCall], call: ToolCall) -> None:
    if limit <= 0:
        return
    key = json.dumps({"name": call.name, "args": call.args}, sort_keys=True)
    cnt = Counter(json.dumps({"name": item.name, "args": item.args}, sort_keys=True) for item in calls)
    if cnt[key] >= limit:
        raise RuntimeError("repeat tool call budget exceeded")
