from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from unidiff import PatchSet

from ..policy import ensure_path
from ..types import ToolCtx, ToolResult, ToolSpec


class PatchArgs(BaseModel):
    patch: str


def spec() -> ToolSpec:
    return ToolSpec(name="patch", description="Apply a unified diff patch", input_model=PatchArgs, execute=run)


def run(ctx: ToolCtx, args: PatchArgs) -> ToolResult:
    patch = PatchSet(args.patch.splitlines(keepends=True))
    if not patch:
        raise ValueError("empty patch")
    files = []
    for item in patch:
        path = _path(ctx.project.root, item.path)
        ensure_path(ctx.project.root, path)
        _apply(path, item)
        files.append(ctx.project.relpath(path))
        if ctx.analysis is not None:
            ctx.analysis.invalidate([path])
        ctx.lsp.refresh(path)
    diag = []
    for item in files:
        diag.extend(ctx.lsp.diagnostics(ctx.project.root / item))
    return ToolResult(
        title="Apply patch",
        output="\n".join(files),
        metadata={"files": files, "diagnostics": diag},
    )


def _path(root: Path, raw: str) -> Path:
    path = raw
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return (root / path).resolve()


def _apply(path: Path, item: object) -> None:
    src = path.read_text() if path.exists() else ""
    old = src.splitlines(keepends=True)
    out: list[str] = []
    pos = 0
    for hunk in item:
        start = max(hunk.source_start - 1, 0)
        out.extend(old[pos:start])
        idx = start
        for line in hunk:
            if line.is_context:
                if idx >= len(old) or old[idx] != line.value:
                    raise ValueError(f"patch context mismatch for {path}")
                out.append(old[idx])
                idx += 1
            elif line.is_removed:
                if idx >= len(old) or old[idx] != line.value:
                    raise ValueError(f"patch remove mismatch for {path}")
                idx += 1
            elif line.is_added:
                out.append(line.value)
        pos = idx
    out.extend(old[pos:])
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(out)
    if getattr(item, "is_removed_file", False):
        if path.exists():
            path.unlink()
        return
    path.write_text(text)
