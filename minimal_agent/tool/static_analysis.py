from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ..policy import ensure_path
from ..types import ToolCtx, ToolResult, ToolSpec


class CommonArgs(BaseModel):
    path: str | None = None
    language: str | None = None
    include_generated: bool = False
    timeout_ms: int | None = None


class SymbolSearchArgs(CommonArgs):
    query: str
    kind: str | None = None
    limit: int = 100
    fuzzy: bool = True


class SymbolDefinitionArgs(CommonArgs):
    symbol: str
    path_hint: str | None = None
    prefer_exact: bool = True


class SymbolReferencesArgs(CommonArgs):
    symbol: str
    definition_file: str | None = None
    include_declaration: bool = True
    limit: int = 500


class FileOutlineArgs(BaseModel):
    file_path: str
    max_depth: int = 4
    include_private: bool = True
    timeout_ms: int | None = None


class SyntaxDiagnosticsArgs(CommonArgs):
    target: str
    max_files: int = 200
    severity_min: str | None = None


class FunctionCallersArgs(CommonArgs):
    function: str
    limit: int = 500


class FunctionCalleesArgs(CommonArgs):
    function: str
    definition_file: str | None = None
    include_external: bool = False


class CallChainArgs(CommonArgs):
    function: str
    max_depth: int = 3
    max_branches: int = 20
    direction: str = "forward"


class PatchSymbolMapArgs(BaseModel):
    patch: str | None = None
    commit_id: str | None = None
    repo_root: str | None = None
    include_context_symbols: bool = True
    timeout_ms: int | None = None


class ImpactAnalysisArgs(BaseModel):
    changed_symbols: list[str] = Field(default_factory=list)
    patch: str | None = None
    max_depth: int = 2
    include_tests: bool = True
    timeout_ms: int | None = None


def specs() -> list[ToolSpec]:
    return [
        ToolSpec(name="symbol_search", description="Search symbols across the workspace with structured results", input_model=SymbolSearchArgs, execute=run_symbol_search),
        ToolSpec(
            name="symbol_definition",
            description="Resolve the canonical definition for a symbol; use when you need the exact function, type, or variable definition touched by a change",
            input_model=SymbolDefinitionArgs,
            execute=run_symbol_definition,
        ),
        ToolSpec(
            name="symbol_references",
            description="Find references to a resolved symbol; use after resolving a definition to see where that symbol is used",
            input_model=SymbolReferencesArgs,
            execute=run_symbol_references,
        ),
        ToolSpec(
            name="file_outline",
            description="Return the structured symbol outline for a file; use to quickly understand file structure before reading full file contents",
            input_model=FileOutlineArgs,
            execute=run_file_outline,
        ),
        ToolSpec(name="syntax_diagnostics", description="Collect syntax diagnostics for a file or directory", input_model=SyntaxDiagnosticsArgs, execute=run_syntax_diagnostics),
        ToolSpec(
            name="function_callers",
            description="Find direct callers of a function; use when you need to confirm which code paths invoke the changed logic",
            input_model=FunctionCallersArgs,
            execute=run_function_callers,
        ),
        ToolSpec(
            name="function_callees",
            description="Find direct callees used by a function; use when you need to confirm what the changed logic calls into",
            input_model=FunctionCalleesArgs,
            execute=run_function_callees,
        ),
        ToolSpec(name="call_chain", description="Compute a bounded call chain from an entry function", input_model=CallChainArgs, execute=run_call_chain),
        ToolSpec(
            name="patch_symbol_map",
            description=(
                "Map patch hunks to impacted symbols and ranges; use on a commit patch to see which functions or types were actually touched "
                "before reading more code. Prefer passing commit_id for git commit analysis so the runtime can fetch the raw patch directly "
                "instead of relying on a model-copied patch string."
            ),
            input_model=PatchSymbolMapArgs,
            execute=run_patch_symbol_map,
        ),
        ToolSpec(
            name="impact_analysis",
            description="Estimate impacted symbols, blast radius, and candidate tests from changes; use when you need likely downstream impact instead of manual broad search",
            input_model=ImpactAnalysisArgs,
            execute=run_impact_analysis,
        ),
    ]


def run_symbol_search(ctx: ToolCtx, args: SymbolSearchArgs) -> ToolResult:
    _ensure_optional_path(ctx, args.path)
    env = ctx.analysis.symbol_search(
        query=args.query,
        kind=args.kind,
        path=args.path,
        language=args.language,
        limit=args.limit,
        fuzzy=args.fuzzy,
        include_generated=args.include_generated,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def run_symbol_definition(ctx: ToolCtx, args: SymbolDefinitionArgs) -> ToolResult:
    _ensure_optional_path(ctx, args.path)
    _ensure_optional_path(ctx, args.path_hint)
    env = ctx.analysis.symbol_definition(
        symbol=args.symbol,
        path_hint=args.path_hint or args.path,
        language=args.language,
        prefer_exact=args.prefer_exact,
        include_generated=args.include_generated,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def run_symbol_references(ctx: ToolCtx, args: SymbolReferencesArgs) -> ToolResult:
    _ensure_optional_path(ctx, args.path)
    _ensure_optional_path(ctx, args.definition_file)
    env = ctx.analysis.symbol_references(
        symbol=args.symbol,
        definition_file=args.definition_file,
        include_declaration=args.include_declaration,
        path=args.path,
        language=args.language,
        limit=args.limit,
        include_generated=args.include_generated,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def run_file_outline(ctx: ToolCtx, args: FileOutlineArgs) -> ToolResult:
    _ensure_required_path(ctx, args.file_path)
    env = ctx.analysis.file_outline(
        file_path=args.file_path,
        max_depth=args.max_depth,
        include_private=args.include_private,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def run_syntax_diagnostics(ctx: ToolCtx, args: SyntaxDiagnosticsArgs) -> ToolResult:
    _ensure_required_path(ctx, args.target)
    env = ctx.analysis.syntax_diagnostics(
        target=args.target,
        language=args.language,
        max_files=args.max_files,
        severity_min=args.severity_min,
        include_generated=args.include_generated,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def run_function_callers(ctx: ToolCtx, args: FunctionCallersArgs) -> ToolResult:
    _ensure_optional_path(ctx, args.path)
    env = ctx.analysis.function_callers(
        function=args.function,
        path=args.path,
        language=args.language,
        limit=args.limit,
        include_generated=args.include_generated,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def run_function_callees(ctx: ToolCtx, args: FunctionCalleesArgs) -> ToolResult:
    _ensure_optional_path(ctx, args.path)
    _ensure_optional_path(ctx, args.definition_file)
    env = ctx.analysis.function_callees(
        function=args.function,
        definition_file=args.definition_file or args.path,
        include_external=args.include_external,
        language=args.language,
        include_generated=args.include_generated,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def run_call_chain(ctx: ToolCtx, args: CallChainArgs) -> ToolResult:
    _ensure_optional_path(ctx, args.path)
    env = ctx.analysis.call_chain(
        function=args.function,
        max_depth=args.max_depth,
        max_branches=args.max_branches,
        direction=args.direction,  # type: ignore[arg-type]
        path=args.path,
        language=args.language,
        include_generated=args.include_generated,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def run_patch_symbol_map(ctx: ToolCtx, args: PatchSymbolMapArgs) -> ToolResult:
    env = ctx.analysis.patch_symbol_map(
        patch=args.patch,
        commit_id=args.commit_id,
        repo_root=args.repo_root,
        include_context_symbols=args.include_context_symbols,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def run_impact_analysis(ctx: ToolCtx, args: ImpactAnalysisArgs) -> ToolResult:
    env = ctx.analysis.impact_analysis(
        changed_symbols=args.changed_symbols,
        patch=args.patch,
        max_depth=args.max_depth,
        include_tests=args.include_tests,
        timeout_ms=args.timeout_ms,
    )
    return _result(env)


def _result(env: dict[str, object]) -> ToolResult:
    metadata = {
        "ok": env["ok"],
        "tool": env["tool"],
        "stats": env["stats"],
        "warnings": env["warnings"],
    }
    return ToolResult(
        title=str(env["tool"]),
        output=json.dumps(env, indent=2, ensure_ascii=False),
        metadata=metadata,
    )


def _ensure_optional_path(ctx: ToolCtx, value: str | None) -> None:
    if not value:
        return
    ensure_path(ctx.project.root, (ctx.project.cwd / value).resolve())


def _ensure_required_path(ctx: ToolCtx, value: str) -> None:
    ensure_path(ctx.project.root, (ctx.project.cwd / value).resolve())
