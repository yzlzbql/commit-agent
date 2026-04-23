from __future__ import annotations

from ..types import StepRecord


def build(task: str, recs: list[StepRecord], files: list[str], next_step: str) -> str:
    done = []
    stat = []
    discoveries = []
    for rec in recs:
        if rec.kind == "tool":
            done.append(f"- {rec.payload.get('name', 'tool')}: {rec.payload.get('title', '')}")
            out = str(rec.payload.get("output", "")).strip()
            if out:
                discoveries.append(f"- {rec.payload.get('name', 'tool')}: {out[:180]}")
        if rec.kind == "error":
            stat.append(f"- error: {rec.payload.get('message', '')}")
        if rec.kind == "verify":
            stat.append(f"- verify: {'ok' if rec.payload.get('ok') else 'failed'}")
        if rec.kind == "final":
            stat.append("- final draft prepared")
    rel = "\n".join(f"- {item}" for item in files) or "- none"
    comp = "\n".join(done) or "- none"
    cur = "\n".join(stat) or "- active"
    notes = "\n".join(discoveries[:8]) or "- none"
    return "\n".join(
        [
            "## Goal",
            "",
            task,
            "",
            "## Discoveries",
            "",
            notes,
            "",
            "## Completed",
            "",
            comp,
            "",
            "## Current status",
            "",
            cur,
            "",
            "## Relevant files",
            "",
            rel,
            "",
            "## Next step",
            "",
            next_step or "Continue acting or verify.",
            "",
        ]
    )
