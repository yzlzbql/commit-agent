from __future__ import annotations

from pathlib import Path
import time
from uuid import uuid4

from ..types import RunInput, SessionState, StepRecord, TodoItem


class SessionStore:
    def __init__(self, root: Path):
        self.root = root

    def create(self, run: RunInput) -> SessionState:
        sid = uuid4().hex[:12]
        st = SessionState(
            id=sid,
            cwd=run.cwd,
            agent=run.agent,
            title=run.title or run.task[:80],
            parent_id=run.parent_id,
        )
        dir = self.root / sid
        dir.mkdir(parents=True, exist_ok=True)
        self.save_state(st)
        self.append_event(
            sid,
            StepRecord(
                kind="task",
                ts=time.time(),
                payload={
                    "task": run.task,
                    "cwd": str(run.cwd),
                    "agent": run.agent,
                    "parent_id": run.parent_id,
                    "title": st.title,
                },
            ),
        )
        self.save_todo(sid, [])
        self.save_summary(sid, "")
        return st

    def load(self, sid: str) -> SessionState:
        path = self.root / sid / "session.json"
        return SessionState.model_validate_json(path.read_text())

    def save_state(self, st: SessionState) -> None:
        path = self.root / st.id / "session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(st.model_dump_json(indent=2))

    def append_event(self, sid: str, rec: StepRecord) -> None:
        path = self.root / sid / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(rec.model_dump_json())
            fh.write("\n")

    def events(self, sid: str) -> list[StepRecord]:
        path = self.root / sid / "events.jsonl"
        if not path.exists():
            return []
        return [StepRecord.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]

    def save_todo(self, sid: str, items: list[TodoItem]) -> None:
        path = self.root / sid / "todo.json"
        path.write_text("[" + ",\n".join(item.model_dump_json() for item in items) + "]")

    def load_todo(self, sid: str) -> list[TodoItem]:
        path = self.root / sid / "todo.json"
        if not path.exists():
            return []
        return [TodoItem.model_validate(item) for item in __import__("json").loads(path.read_text())]

    def save_summary(self, sid: str, text: str) -> None:
        path = self.root / sid / "summary.md"
        path.write_text(text)

    def load_summary(self, sid: str) -> str:
        path = self.root / sid / "summary.md"
        if not path.exists():
            return ""
        return path.read_text()

    def files(self, sid: str) -> list[str]:
        result = []
        for rec in self.events(sid):
            if rec.kind != "tool":
                continue
            for item in rec.payload.get("files", []):
                if item not in result:
                    result.append(item)
        return result
