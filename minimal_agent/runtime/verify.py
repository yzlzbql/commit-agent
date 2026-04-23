from __future__ import annotations

from pathlib import Path
import subprocess

from ..config import AgentConfig
from ..project import Project
from ..session.store import SessionStore
from ..types import SessionState, VerifyResult


def run(project: Project, cfg: AgentConfig, store: SessionStore, st: SessionState) -> VerifyResult:
    checks = []
    fails = []
    events = store.events(st.id)
    for cmd in cfg.verify.commands:
        res = subprocess.run(
            cmd,
            cwd=project.root,
            shell=True,
            capture_output=True,
            text=True,
        )
        checks.append(f"$ {cmd}")
        if res.returncode != 0:
            out = (res.stdout + "\n" + res.stderr).strip()
            fails.append(out or f"command failed: {cmd}")
            return VerifyResult(ok=False, checks=checks, failures=fails, next_action="Fix failing verification")
    if checks:
        return VerifyResult(ok=True, checks=checks, next_action="Complete")
    files = store.files(st.id)
    if files:
        checks.append("detected changed files from tool events")
        return VerifyResult(ok=True, checks=checks, next_action="Complete")
    for item in _artifacts(project.root):
        checks.append(f"detected artifact: {item}")
        return VerifyResult(ok=True, checks=checks, next_action="Complete")
    if st.candidate.strip() and _read_only(events):
        checks.append("accepted final answer for read-only task")
        return VerifyResult(ok=True, checks=checks, next_action="Complete")
    return VerifyResult(
        ok=False,
        checks=checks,
        failures=["no verifiable outcome detected"],
        next_action="Produce a concrete result or configure verify.commands",
    )


def _artifacts(root: Path) -> list[str]:
    result = []
    for name in ["dist", "build", "coverage", ".pytest_cache"]:
        path = root / name
        if path.exists():
            result.append(name)
    return result


def _read_only(events: list[object]) -> bool:
    for rec in events:
        if getattr(rec, "kind", "") != "tool":
            continue
        name = rec.payload.get("name", "")
        if name == "patch":
            return False
    return True
