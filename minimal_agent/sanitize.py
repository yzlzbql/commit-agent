from __future__ import annotations

import json
import re
from typing import Any

from .commit_labels import LEGACY_COMMIT_LABELS
from .types import ToolResult


_LABEL_ALT = "|".join(sorted(LEGACY_COMMIT_LABELS, key=len, reverse=True))
_CONVENTIONAL_PREFIX = re.compile(
    rf"^\s*(?:{_LABEL_ALT})(?:\([^)]+\))?!?:\s*",
    re.IGNORECASE,
)
_SENSITIVE_LINE_PATTERNS = (
    re.compile(r"^\s*category\s*:\s*.+$", re.IGNORECASE),
    re.compile(r"^\s*fixes\s*:\s*.+$", re.IGNORECASE),
    re.compile(r"^\s*cve\s*:\s*.+$", re.IGNORECASE),
    re.compile(r"^\s*bugzilla\s*:\s*.+$", re.IGNORECASE),
)
_COMMIT_TEXT_KEYS = {"subject", "body", "message", "commit_message"}


def sanitize_commit_subject(subject: str) -> str:
    value = (subject or "").strip()
    if not value:
        return ""
    stripped = _CONVENTIONAL_PREFIX.sub("", value, count=1).strip()
    return stripped or "Message intentionally sanitized."


def sanitize_commit_body(body: str) -> str:
    if not body:
        return ""
    kept: list[str] = []
    for line in body.splitlines():
        if any(pattern.match(line) for pattern in _SENSITIVE_LINE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def sanitize_commit_message(message: str) -> str:
    raw = (message or "").strip()
    if not raw:
        return ""
    first, *rest = raw.splitlines()
    subject = sanitize_commit_subject(first)
    body = sanitize_commit_body("\n".join(rest))
    if not body:
        return subject
    return f"{subject}\n{body}".strip()


def scrub_text(text: str) -> str:
    if not text:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        if any(pattern.match(line) for pattern in _SENSITIVE_LINE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def scrub_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {item_key: scrub_value(item_value, key=item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [scrub_value(item, key=key) for item in value]
    if not isinstance(value, str):
        return value
    if key in _COMMIT_TEXT_KEYS:
        return sanitize_commit_message(value)
    return scrub_text(value)


def scrub_tool_result(result: ToolResult) -> ToolResult:
    output = result.output
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        scrubbed_output = scrub_text(output)
    else:
        scrubbed_output = json.dumps(scrub_value(parsed), ensure_ascii=False, indent=2)
    return ToolResult(
        title=scrub_text(result.title),
        output=scrubbed_output,
        metadata=scrub_value(result.metadata),
        attachments=scrub_value(result.attachments),
    )
