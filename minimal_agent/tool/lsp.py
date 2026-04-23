from __future__ import annotations

from pydantic import BaseModel

from ..policy import ensure_path
from ..types import LspRequest, ToolCtx, ToolResult, ToolSpec


class LspArgs(BaseModel):
    operation: str
    file_path: str = "."
    line: int = 1
    character: int = 1
    query: str | None = None


def spec() -> ToolSpec:
    return ToolSpec(name="lsp", description="Run a local LSP query", input_model=LspArgs, execute=run)


def run(ctx: ToolCtx, args: LspArgs) -> ToolResult:
    path = (ctx.project.cwd / args.file_path).resolve()
    ensure_path(ctx.project.root, path)
    res = ctx.lsp.call(
        LspRequest(
            operation=args.operation,  # type: ignore[arg-type]
            file_path=args.file_path,
            line=args.line,
            character=args.character,
            query=args.query,
        )
    )
    body = res.model_dump_json(indent=2)
    return ToolResult(title=f"LSP {args.operation}", output=body, metadata={"ok": res.ok})
