from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ..policy import ensure_path
from ..types import ToolCtx, ToolResult, ToolSpec
from .write import _diff, _write_output


class EditArgs(BaseModel):
    file_path: str
    old_string: str
    new_string: str
    replace_all: bool = False


def spec() -> ToolSpec:
    return ToolSpec(name="edit", description="Replace text in a file using an exact old/new string pair", input_model=EditArgs, execute=run)


def run(ctx: ToolCtx, args: EditArgs) -> ToolResult:
    path = _resolve(ctx, args.file_path)
    if args.old_string == args.new_string:
        raise ValueError("old_string and new_string must differ")
    if args.old_string == "":
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        if before:
            raise ValueError("old_string may be empty only when creating a new file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.new_string, encoding="utf-8")
        if ctx.analysis is not None:
            ctx.analysis.invalidate([path])
        ctx.lsp.refresh(path)
        diagnostics = ctx.lsp.diagnostics(path)
        return ToolResult(
            title=f"Edit {ctx.project.relpath(path)}",
            output=_write_output(diagnostics),
            metadata={"files": [ctx.project.relpath(path)], "diff": _diff(path, before, args.new_string), "diagnostics": diagnostics},
        )

    if not path.exists():
        raise ValueError(f"file not found: {path}")
    before = path.read_text(encoding="utf-8")
    after = _replace(before, args.old_string, args.new_string, args.replace_all)
    path.write_text(after, encoding="utf-8")
    if ctx.analysis is not None:
        ctx.analysis.invalidate([path])
    ctx.lsp.refresh(path)
    diagnostics = ctx.lsp.diagnostics(path)
    return ToolResult(
        title=f"Edit {ctx.project.relpath(path)}",
        output=_write_output(diagnostics),
        metadata={"files": [ctx.project.relpath(path)], "diff": _diff(path, before, after), "diagnostics": diagnostics},
    )


def _resolve(ctx: ToolCtx, file_path: str) -> Path:
    path = Path(file_path)
    if not path.is_absolute():
        path = (ctx.project.cwd / path).resolve()
    ensure_path(ctx.project.root, path)
    return path


def _replace(content: str, old: str, new: str, replace_all: bool) -> str:
    count = content.count(old)
    if count == 0:
        raise ValueError("old_string not found in file")
    if count > 1 and not replace_all:
        raise ValueError("old_string appears multiple times; set replace_all=true to replace every occurrence")
    if replace_all:
        return content.replace(old, new)
    return content.replace(old, new, 1)
