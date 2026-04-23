from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _root(cwd: Path) -> Path:
    cur = cwd.resolve()
    found = None
    for item in [cur, *cur.parents]:
        if (item / ".git").exists():
            found = item
    return found or cur


@dataclass(slots=True)
class Project:
    cwd: Path
    root: Path
    agent_dir: Path
    sessions_dir: Path

    @classmethod
    def load(cls, cwd: str | Path, *, state_root: str | Path | None = None) -> "Project":
        path = Path(cwd).resolve()
        root = _root(path)
        agent_dir = Path(state_root).resolve() if state_root is not None else root / ".agent"
        sessions_dir = agent_dir / "sessions"
        agent_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return cls(cwd=path, root=root, agent_dir=agent_dir, sessions_dir=sessions_dir)

    def contains(self, path: str | Path) -> bool:
        item = Path(path).resolve()
        return self.root == item or self.root in item.parents

    def relpath(self, path: str | Path) -> str:
        item = Path(path).resolve()
        try:
            return item.relative_to(self.root).as_posix()
        except ValueError:
            return str(item)

    def session_dir(self, sid: str) -> Path:
        return self.sessions_dir / sid
