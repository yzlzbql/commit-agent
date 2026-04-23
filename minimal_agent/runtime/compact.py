from __future__ import annotations

import json

from ..types import StepRecord


def _trim(text: str, size: int = 800) -> str:
    if len(text) <= size:
        return text
    return text[:size] + "\n..."


def apply(recs: list[StepRecord], keep: int = 6) -> tuple[list[StepRecord], str]:
    rich = [rec for rec in recs if rec.kind in {"tool", "final"}]
    if len(rich) <= keep:
        return recs, ""
    cut = rich[:-keep]
    tail = rich[-keep:]
    completed = []
    discoveries = []
    for rec in cut:
        if rec.kind == "tool":
            completed.append(f"- tool `{rec.payload.get('name', 'tool')}`")
            args = rec.payload.get("args", {})
            title = rec.payload.get("title", "")
            if args:
                completed.append(f"  - args: {json.dumps(args, ensure_ascii=False, sort_keys=True)}")
            if title:
                completed.append(f"  - title: {_trim(title, 180)}")
            out = rec.payload.get("output", "")
            if out:
                discoveries.append(f"- {_trim(out, 180)}")
        if rec.kind == "final":
            completed.append(f"- final candidate: {_trim(rec.payload.get('text', ''), 180)}")
    lines = [
        "## Compacted history",
        "",
        "### Completed",
        "",
        *(completed or ["- none"]),
        "",
        "### Discoveries",
        "",
        *(discoveries[:8] or ["- none"]),
    ]
    keep_ids = {id(rec) for rec in tail}
    fresh = [rec for rec in recs if rec.kind not in {"tool", "final"} or id(rec) in keep_ids]
    return fresh, "\n".join(lines).strip()
