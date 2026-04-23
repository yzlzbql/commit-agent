from __future__ import annotations

import json
import re
import shlex

from ..types import StepRecord, ToolCall


_SPACE = re.compile(r"\s+")
_DISPLAY_PIPE = re.compile(r"\|\s*(?:cat|head|tail)\b.*$")
_CD_PREFIX = re.compile(r"^\s*cd\s+.+?\s*&&\s*", re.DOTALL)
_GIT_SHOW_FLAGS = {"--stat", "--patch", "--no-patch", "--name-only", "--name-status", "--no-stat"}
_GIT_SHOW_VALUE_FLAGS = {"--pretty", "--format", "--unified", "-U"}
_GREP_VALUE_FLAGS = {"-A", "-B", "-C", "--after-context", "--before-context", "--context", "--color"}
_GREP_DISPLAY_FLAGS = {
    "-n",
    "-H",
    "-h",
    "--line-number",
    "--with-filename",
    "--no-filename",
    "--heading",
}


def check(recs: list[StepRecord], call: ToolCall, threshold: int) -> str:
    if threshold <= 0:
        return ""
    recent = [rec for rec in recs if rec.kind == "tool"]
    if len(recent) < threshold:
        return ""
    want = _fingerprint(call)
    window = recent[-threshold:]
    if all(_fingerprint(ToolCall(name=item.payload.get("name", ""), args=item.payload.get("args", {}))) == want for item in window):
        return (
            "You already executed this tool call repeatedly. Use the information you have, "
            "respond with the plain-text answer now, or choose a materially different next step."
        )
    return ""


def _fingerprint(call: ToolCall) -> str:
    args = dict(call.args)
    if call.name == "bash":
        args["command"] = _normalize_bash_command(str(args.get("command", "")))
    return json.dumps({"name": call.name, "args": args}, ensure_ascii=False, sort_keys=True)


def _normalize_bash_command(cmd: str) -> str:
    cmd = _DISPLAY_PIPE.sub("", cmd)
    cmd = _CD_PREFIX.sub("", cmd).strip()
    cmd = _SPACE.sub(" ", cmd).strip()
    if not cmd:
        return cmd
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return cmd
    if len(parts) >= 3 and parts[0] == "git" and parts[1] == "show":
        return _normalize_git_show(parts)
    if parts and parts[0] == "grep":
        return _normalize_grep(parts)
    return " ".join(parts)


def _normalize_git_show(parts: list[str]) -> str:
    normalized = ["git", "show"]
    positional: list[str] = []
    extras: list[str] = []
    idx = 2
    while idx < len(parts):
        token = parts[idx]
        if token == "--":
            extras.extend(parts[idx:])
            break
        if token in _GIT_SHOW_FLAGS:
            idx += 1
            continue
        if token in _GIT_SHOW_VALUE_FLAGS:
            idx += 2
            continue
        if token.startswith("-U") and token != "-U":
            idx += 1
            continue
        if any(token.startswith(f"{flag}=") for flag in _GIT_SHOW_VALUE_FLAGS):
            idx += 1
            continue
        if token.startswith("-"):
            extras.append(token)
            idx += 1
            continue
        positional.append(token)
        idx += 1
    normalized.extend(positional)
    normalized.extend(extras)
    return " ".join(normalized)


def _normalize_grep(parts: list[str]) -> str:
    normalized = ["grep"]
    idx = 1
    while idx < len(parts):
        token = parts[idx]
        if token in _GREP_DISPLAY_FLAGS:
            idx += 1
            continue
        if token in _GREP_VALUE_FLAGS:
            idx += 2
            continue
        if re.match(r"^-[ABC]\d+$", token):
            idx += 1
            continue
        if any(token.startswith(f"{flag}=") for flag in _GREP_VALUE_FLAGS if flag.startswith("--")):
            idx += 1
            continue
        normalized.append(token)
        idx += 1
    return " ".join(normalized)
