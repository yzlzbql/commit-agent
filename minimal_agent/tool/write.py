from __future__ import annotations

from difflib import unified_diff
from pathlib import Path

from pydantic import BaseModel

from ..policy import ensure_path
from ..types import ToolCtx, ToolResult, ToolSpec


class WriteArgs(BaseModel):
    file_path: str
    content: str


def spec() -> ToolSpec:
    return ToolSpec(name="write", description="Write the full contents of a file", input_model=WriteArgs, execute=run)


def run(ctx: ToolCtx, args: WriteArgs) -> ToolResult:
    path = _resolve(ctx, args.file_path)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args.content, encoding="utf-8")
    if ctx.analysis is not None:
        ctx.analysis.invalidate([path])
    ctx.lsp.refresh(path)
    diff = _diff(path, old, args.content)
    diagnostics = ctx.lsp.diagnostics(path)
    return ToolResult(
        title=f"Write {ctx.project.relpath(path)}",
        output=_write_output(diagnostics),
        metadata={"files": [ctx.project.relpath(path)], "diff": diff, "diagnostics": diagnostics},
    )


def _resolve(ctx: ToolCtx, file_path: str) -> Path:
    path = Path(file_path)
    if not path.is_absolute():
        path = (ctx.project.cwd / path).resolve()
    ensure_path(ctx.project.root, path)
    return path


def _diff(path: Path, before: str, after: str) -> str:
    return "".join(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
    )


def _write_output(diagnostics: list[dict]) -> str:
    if not diagnostics:
        return "Wrote file successfully."
    return "Wrote file successfully.\n\nDiagnostics:\n" + "\n".join(str(item) for item in diagnostics[:20])
