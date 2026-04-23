from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import json
import re
import shlex
import shutil
import subprocess
import time
from typing import Any, Iterable, Literal

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from ..policy import ensure_path


_C_EXTENSIONS = {".c"}
_CPP_EXTENSIONS = {".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"}
_HEADER_EXTENSIONS = {".h", ".hh", ".hpp", ".hxx"}
_PYTHON_EXTENSIONS = {".py"}
_GO_EXTENSIONS = {".go"}
_RUST_EXTENSIONS = {".rs"}
_JS_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTENSIONS = {".ts", ".tsx"}
_SUPPORTED_EXTENSIONS = _C_EXTENSIONS | _CPP_EXTENSIONS | _HEADER_EXTENSIONS | _PYTHON_EXTENSIONS | _GO_EXTENSIONS | _RUST_EXTENSIONS | _JS_EXTENSIONS | _TS_EXTENSIONS
_GENERATED_PARTS = {".agent", ".git", ".hg", ".svn", "__pycache__", "node_modules", "dist", "build", "target", ".venv", "venv"}
_NON_RETRYABLE_CODES = {
    "E_INPUT_INVALID",
    "E_PATH_FORBIDDEN",
    "E_PATH_NOT_FOUND",
    "E_LSP_UNAVAILABLE",
    "E_SYMBOL_NOT_FOUND",
    "E_AMBIGUOUS_SYMBOL",
    "E_UNSUPPORTED_LANGUAGE",
}
_PYTHON_KINDS = {"function", "method", "class", "variable"}
_CONTROL_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch", "new", "delete", "throw"}
_PYTHON_CALL_SKIP = {"def", "class", "return", "if", "elif", "while", "for", "with", "print"}
_COMPILER_DIAG = re.compile(r"^(.*?):(\d+):(?:(\d+):)?\s*(fatal error|error|warning|note):\s*(.*)$")


@dataclass(slots=True)
class SymbolRecord:
    name: str
    kind: str
    file_path: str
    abs_path: Path
    line: int
    character: int
    container: str | None
    scope_kind: str | None
    signature: str | None
    language: str
    pattern: str | None

    @property
    def full_name(self) -> str:
        if not self.container:
            return self.name
        return f"{self.container}.{self.name}"


@dataclass(slots=True)
class TagCache:
    symbols: list[SymbolRecord]
    files: list[str]
    version: int


class AnalysisService:
    def __init__(self, cfg: Any, project: Any, lsp: Any):
        self.cfg = cfg.analysis
        self.project = project
        self.root = project.root
        self.cwd = project.cwd
        self.lsp = lsp
        self._version = 0
        self._tag_cache: dict[bool, TagCache] = {}
        self._file_cache: dict[str, list[str]] = {}
        self._compile_db_cache: dict[str, dict[str, Any]] | None = None

    def invalidate(self, paths: Iterable[str | Path]) -> None:
        self._version += 1
        self._tag_cache.clear()
        self._compile_db_cache = None
        for item in paths:
            try:
                rel = self.project.relpath(item)
            except Exception:
                continue
            self._file_cache.pop(rel, None)

    def symbol_search(
        self,
        *,
        query: str,
        kind: str | None = None,
        path: str | None = None,
        language: str | None = None,
        limit: int = 100,
        fuzzy: bool = True,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        if not query.strip():
            _fail("E_INPUT_INVALID", "query must not be empty")
        query_norm = query.strip().lower()
        want_kind = _normalize_kind_name(kind)
        tags = self._filter_symbols(
            self._load_tags(include_generated=include_generated, timeout_ms=timeout_ms),
            path=path,
            language=language,
        )
        matches = []
        for item in tags:
            if want_kind is not None and item.kind != want_kind:
                continue
            haystacks = [item.name.lower(), item.full_name.lower()]
            matched = any(query_norm in value for value in haystacks) if fuzzy else query_norm in haystacks
            if not matched:
                continue
            matches.append(
                {
                    "name": item.name,
                    "kind": item.kind,
                    "file_path": item.file_path,
                    "line": item.line,
                    "character": item.character,
                    "container": item.container,
                    "signature": item.signature,
                }
            )
        matches.sort(key=lambda item: (item["name"] != query, item["file_path"], item["line"]))
        data = {"matches": matches[:limit]}
        return self._envelope(
            tool="symbol_search",
            start=start,
            data=data,
            scanned_files=len({item.file_path for item in tags}),
            returned_items=len(data["matches"]),
        )

    def symbol_definition(
        self,
        *,
        symbol: str,
        path_hint: str | None = None,
        language: str | None = None,
        prefer_exact: bool = True,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        rec, alternatives = self._resolve_symbol(
            symbol=symbol,
            path_hint=path_hint,
            language=language,
            prefer_exact=prefer_exact,
            include_generated=include_generated,
            timeout_ms=timeout_ms,
        )
        data = {
            "symbol": symbol,
            "definition": self._symbol_payload(rec, include_doc=True),
            "alternatives": [self._symbol_payload(item, include_doc=False) for item in alternatives],
        }
        return self._envelope(
            tool="symbol_definition",
            start=start,
            data=data,
            scanned_files=len({rec.file_path, *(item.file_path for item in alternatives)}),
            returned_items=1,
        )

    def symbol_references(
        self,
        *,
        symbol: str,
        definition_file: str | None = None,
        include_declaration: bool = True,
        path: str | None = None,
        language: str | None = None,
        limit: int = 500,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        rec, _ = self._resolve_symbol(
            symbol=symbol,
            path_hint=definition_file or path,
            language=language,
            prefer_exact=True,
            include_generated=include_generated,
            timeout_ms=timeout_ms,
        )
        warnings: list[str] = []
        if not self._lsp_enabled(rec.abs_path):
            warnings.append("REDUCED_PRECISION_NO_LSP")
        refs = self._find_references(
            rec,
            path=path,
            include_generated=include_generated,
            timeout_ms=timeout_ms,
            limit=limit,
        )
        if not include_declaration:
            refs = [item for item in refs if not (item["file_path"] == rec.file_path and item["line"] == rec.line)]
        data = {
            "symbol": symbol,
            "total": len(refs),
            "references": refs[:limit],
        }
        return self._envelope(
            tool="symbol_references",
            start=start,
            data=data,
            scanned_files=len({item["file_path"] for item in refs}) if refs else 0,
            returned_items=len(data["references"]),
            warnings=warnings,
        )

    def file_outline(
        self,
        *,
        file_path: str,
        max_depth: int = 4,
        include_private: bool = True,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        path = self._resolve(file_path, must_exist=True)
        if path.is_dir():
            _fail("E_INPUT_INVALID", f"file_path must be a file: {file_path}")
        rel = self.project.relpath(path)
        language = _language_for_path(path)
        if language is None:
            _fail("E_UNSUPPORTED_LANGUAGE", f"no outline backend for {rel}")
        tags = [item for item in self._load_tags(include_generated=False, timeout_ms=timeout_ms) if item.file_path == rel]
        symbols = self._outline_tree(tags, max_depth=max_depth, include_private=include_private)
        data = {"file_path": rel, "language": language, "symbols": symbols}
        return self._envelope(
            tool="file_outline",
            start=start,
            data=data,
            scanned_files=1,
            returned_items=len(symbols),
        )

    def syntax_diagnostics(
        self,
        *,
        target: str,
        language: str | None = None,
        max_files: int = 200,
        severity_min: str | None = None,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        target_path = self._resolve(target, must_exist=True)
        warnings: list[str] = []
        if target_path.is_file():
            files = [target_path]
        else:
            files = list(self._iter_source_files(target_path, include_generated=include_generated))[:max_files]
        items: list[dict[str, Any]] = []
        for item in files:
            lang = _normalize_language(language) or _language_for_path(item)
            if lang == "python":
                items.extend(self._python_diagnostics(item))
                continue
            if lang in {"c", "cpp"}:
                diags, extra_warnings = self._compiler_diagnostics(item, lang, timeout_ms=timeout_ms)
                items.extend(diags)
                warnings.extend(extra_warnings)
                continue
            if lang is None:
                continue
        level = _severity_rank(severity_min)
        filtered = [item for item in items if _severity_rank(item["severity"]) >= level]
        summary = {
            "error": sum(1 for item in filtered if item["severity"] == "error"),
            "warning": sum(1 for item in filtered if item["severity"] == "warning"),
            "information": sum(1 for item in filtered if item["severity"] == "information"),
        }
        data = {"target": self.project.relpath(target_path), "diagnostics": filtered, "summary": summary}
        return self._envelope(
            tool="syntax_diagnostics",
            start=start,
            data=data,
            scanned_files=len(files),
            returned_items=len(filtered),
            warnings=sorted(set(warnings)),
        )

    def function_callers(
        self,
        *,
        function: str,
        path: str | None = None,
        language: str | None = None,
        limit: int = 500,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        target, _ = self._resolve_function(
            function,
            path_hint=path,
            language=language,
            include_generated=include_generated,
            timeout_ms=timeout_ms,
        )
        warnings: list[str] = []
        if not self._lsp_enabled(target.abs_path):
            warnings.append("REDUCED_PRECISION_NO_LSP")
        callers = []
        functions = self._function_symbols(path=path, language=language, include_generated=include_generated, timeout_ms=timeout_ms)
        for candidate in functions:
            for call in self._function_calls(candidate):
                if call["name"] != target.name:
                    continue
                callers.append(
                    {
                        "caller_name": candidate.full_name,
                        "file_path": candidate.file_path,
                        "line": call["line"],
                        "call_expression": call["call_expression"],
                    }
                )
        callers.sort(key=lambda item: (item["file_path"], item["line"]))
        data = {
            "function": function,
            "resolved_definition": self._symbol_payload(target, include_doc=False),
            "callers": callers[:limit],
        }
        return self._envelope(
            tool="function_callers",
            start=start,
            data=data,
            scanned_files=len({item.file_path for item in functions}),
            returned_items=len(data["callers"]),
            warnings=warnings,
        )

    def function_callees(
        self,
        *,
        function: str,
        definition_file: str | None = None,
        include_external: bool = False,
        language: str | None = None,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        target, _ = self._resolve_function(
            function,
            path_hint=definition_file,
            language=language,
            include_generated=include_generated,
            timeout_ms=timeout_ms,
        )
        warnings: list[str] = []
        if not self._lsp_enabled(target.abs_path):
            warnings.append("REDUCED_PRECISION_NO_LSP")
        callees = self._resolve_callees(target, include_external=include_external)
        data = {"function": function, "callees": callees}
        return self._envelope(
            tool="function_callees",
            start=start,
            data=data,
            scanned_files=1,
            returned_items=len(callees),
            warnings=warnings,
        )

    def call_chain(
        self,
        *,
        function: str,
        max_depth: int = 3,
        max_branches: int = 20,
        direction: Literal["forward", "backward"] = "forward",
        path: str | None = None,
        language: str | None = None,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        if max_depth < 1:
            _fail("E_INPUT_INVALID", "max_depth must be at least 1")
        root, _ = self._resolve_function(
            function,
            path_hint=path,
            language=language,
            include_generated=include_generated,
            timeout_ms=timeout_ms,
        )
        chains: list[dict[str, Any]] = []
        truncated = False

        def walk(node: SymbolRecord, depth: int, trail: list[SymbolRecord]) -> None:
            nonlocal truncated
            if len(chains) >= max_branches:
                truncated = True
                return
            if depth >= max_depth:
                chains.append({"nodes": [self._chain_node(item) for item in trail]})
                return
            next_nodes = self._next_chain_nodes(node, direction=direction)
            if not next_nodes:
                chains.append({"nodes": [self._chain_node(item) for item in trail]})
                return
            progressed = False
            for nxt in next_nodes:
                if nxt.full_name in {item.full_name for item in trail}:
                    continue
                progressed = True
                walk(nxt, depth + 1, [*trail, nxt])
                if len(chains) >= max_branches:
                    truncated = True
                    return
            if not progressed:
                chains.append({"nodes": [self._chain_node(item) for item in trail]})

        walk(root, 0, [root])
        data = {
            "function": function,
            "direction": direction,
            "max_depth": max_depth,
            "chains": chains,
            "truncated": truncated,
        }
        return self._envelope(
            tool="call_chain",
            start=start,
            data=data,
            scanned_files=len({node["file_path"] for chain in chains for node in chain["nodes"] if node["file_path"]}),
            returned_items=len(chains),
        )

    def patch_symbol_map(
        self,
        *,
        patch: str | None = None,
        commit_id: str | None = None,
        repo_root: str | None = None,
        include_context_symbols: bool = True,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        mapping_root = self.root if repo_root is None else self._resolve(repo_root, must_exist=True)
        if mapping_root.is_file():
            _fail("E_INPUT_INVALID", f"repo_root must be a directory: {repo_root}")
        patch_text = self._resolve_patch_text(patch=patch, commit_id=commit_id, mapping_root=mapping_root)
        try:
            patch_set = PatchSet(patch_text.splitlines(keepends=True))
        except UnidiffParseError as err:
            _fail("E_INPUT_INVALID", f"patch is not a valid unified diff: {err}")
        warnings: list[str] = []
        files: list[dict[str, Any]] = []
        mapped_hunks = 0
        for item in patch_set:
            raw_path = item.path
            rel_path = raw_path[2:] if raw_path.startswith(("a/", "b/")) else raw_path
            abs_path = (mapping_root / rel_path).resolve()
            try:
                ensure_path(self.root, abs_path)
            except ValueError:
                _fail("E_PATH_FORBIDDEN", f"path outside project boundary: {abs_path}")
            language = _language_for_path(abs_path)
            if language is None:
                warnings.append(f"UNSUPPORTED_LANGUAGE:{rel_path}")
                continue
            symbols = []
            if abs_path.exists():
                symbols = self._outline_symbols_for_file(self.project.relpath(abs_path), timeout_ms=timeout_ms)
            hunks = []
            for hunk in item:
                target_start = max(hunk.target_start or hunk.source_start or 1, 1)
                target_end = target_start + max(hunk.target_length or len(list(hunk)) or 1, 1) - 1
                hunk_symbols = [self._range_symbol_payload(sym) for sym in symbols if self._intersects(target_start, target_end, sym["range_start"], sym["range_end"])]
                if not hunk_symbols and include_context_symbols:
                    nearest = self._nearest_symbol(symbols, target_start)
                    if nearest is not None:
                        hunk_symbols = [self._range_symbol_payload(nearest)]
                if hunk_symbols:
                    mapped_hunks += 1
                hunks.append(
                    {
                        "old_start": hunk.source_start,
                        "new_start": hunk.target_start,
                        "symbols": hunk_symbols,
                    }
                )
            files.append({"file_path": rel_path, "hunks": hunks})
        data = {
            "files": files,
            "summary": {"total_files": len(files), "total_hunks": sum(len(item["hunks"]) for item in files), "mapped_hunks": mapped_hunks},
        }
        return self._envelope(
            tool="patch_symbol_map",
            start=start,
            data=data,
            scanned_files=len(files),
            returned_items=sum(len(item["hunks"]) for item in files),
            warnings=sorted(set(warnings)),
        )

    def _resolve_patch_text(
        self,
        *,
        patch: str | None,
        commit_id: str | None,
        mapping_root: Path,
    ) -> str:
        raw_patch = (patch or "").strip()
        raw_commit = (commit_id or "").strip()
        if bool(raw_patch) == bool(raw_commit):
            _fail("E_INPUT_INVALID", "provide exactly one of patch or commit_id")
        if raw_patch:
            return patch or ""
        git_dir = mapping_root / ".git"
        if not git_dir.exists():
            _fail("E_INPUT_INVALID", f"not a git repository: {mapping_root}")
        commit = self._git_show_commit(mapping_root, raw_commit)
        patch_text = self._git_patch(mapping_root, commit)
        if not patch_text.strip():
            _fail("E_INPUT_INVALID", f"commit has empty patch: {commit}")
        return patch_text

    def _git_show_commit(self, repo_root: Path, commit_id: str) -> str:
        if not commit_id:
            _fail("E_INPUT_INVALID", "commit_id must not be empty")
        res = subprocess.run(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{commit_id}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            msg = res.stderr.strip() or res.stdout.strip() or "invalid commit"
            _fail("E_INPUT_INVALID", msg)
        return res.stdout.strip()

    def _git_patch(self, repo_root: Path, commit: str) -> str:
        res = subprocess.run(
            ["git", "show", "--format=", "--patch", "--no-ext-diff", "--no-color", "--find-renames", "--end-of-options", commit],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            msg = res.stderr.strip() or res.stdout.strip() or "failed to read commit patch"
            _fail("E_INPUT_INVALID", msg)
        return res.stdout

    def impact_analysis(
        self,
        *,
        changed_symbols: list[str] | None = None,
        patch: str | None = None,
        max_depth: int = 2,
        include_tests: bool = True,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        if not changed_symbols and not patch:
            _fail("E_INPUT_INVALID", "changed_symbols or patch is required")
        roots = list(dict.fromkeys(changed_symbols or []))
        warnings: list[str] = []
        if patch:
            mapped = self.patch_symbol_map(patch=patch, timeout_ms=timeout_ms)
            for file_item in mapped["data"]["files"]:
                for hunk in file_item["hunks"]:
                    for sym in hunk["symbols"]:
                        roots.append(sym["name"])
            warnings.extend(mapped["warnings"])
        roots = list(dict.fromkeys(item for item in roots if item))
        impacted: dict[str, dict[str, Any]] = {}
        queue: list[tuple[str, int]] = [(item, 0) for item in roots]
        while queue:
            name, distance = queue.pop(0)
            try:
                rec, _ = self._resolve_function(name, include_generated=False, timeout_ms=timeout_ms)
            except Exception:
                try:
                    rec, _ = self._resolve_symbol(name, prefer_exact=False, include_generated=False, timeout_ms=timeout_ms)
                except Exception:
                    continue
            cur = impacted.get(rec.full_name)
            if cur is None or distance < cur["distance"]:
                impacted[rec.full_name] = {"name": rec.full_name, "file_path": rec.file_path, "distance": distance}
            if distance >= max_depth:
                continue
            for nxt in self._next_chain_nodes(rec, direction="forward"):
                queue.append((nxt.full_name, distance + 1))
            for nxt in self._next_chain_nodes(rec, direction="backward"):
                queue.append((nxt.full_name, distance + 1))
        impacted_symbols = sorted(impacted.values(), key=lambda item: (item["distance"], item["file_path"], item["name"]))
        suggested_tests = self._suggested_tests(impacted_symbols) if include_tests else []
        cross_file = len({item["file_path"] for item in impacted_symbols})
        score = min(100, len(impacted_symbols) * 12 + cross_file * 10 + max((item["distance"] for item in impacted_symbols), default=0) * 8)
        if score >= 60:
            level = "high"
        elif score >= 30:
            level = "medium"
        else:
            level = "low"
        data = {
            "roots": roots,
            "impacted_symbols": impacted_symbols,
            "suggested_tests": suggested_tests,
            "risk": {"level": level, "score": score},
        }
        return self._envelope(
            tool="impact_analysis",
            start=start,
            data=data,
            scanned_files=cross_file,
            returned_items=len(impacted_symbols),
            warnings=sorted(set(warnings)),
        )

    def _resolve(self, raw: str, *, must_exist: bool) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = (self.cwd / path).resolve()
        try:
            ensure_path(self.root, path)
        except ValueError:
            _fail("E_PATH_FORBIDDEN", f"path outside project boundary: {path}")
        if must_exist and not path.exists():
            _fail("E_PATH_NOT_FOUND", f"path not found: {path}")
        return path

    def _effective_timeout(self, timeout_ms: int | None) -> float:
        cur = timeout_ms if timeout_ms is not None else self.cfg.default_timeout_ms
        return max(cur, 1) / 1000.0

    def _envelope(
        self,
        *,
        tool: str,
        start: float,
        data: dict[str, Any],
        scanned_files: int,
        returned_items: int,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "tool": tool,
            "version": "v0",
            "data": data,
            "stats": {
                "elapsed_ms": int((time.monotonic() - start) * 1000),
                "scanned_files": scanned_files,
                "returned_items": returned_items,
            },
            "warnings": warnings or [],
        }

    def _load_tags(self, *, include_generated: bool, timeout_ms: int | None) -> list[SymbolRecord]:
        cache = self._tag_cache.get(include_generated)
        if cache is not None and cache.version == self._version:
            return cache.symbols
        files = [item for item in self._iter_source_files(self.root, include_generated=include_generated)]
        files = files[: self.cfg.index_max_files]
        if not files:
            self._tag_cache[include_generated] = TagCache(symbols=[], files=[], version=self._version)
            return []
        if shutil.which(self.cfg.ctags_command[0]) is None:
            _fail("E_UNSUPPORTED_LANGUAGE", "ctags is not available")
        cmd = [
            *self.cfg.ctags_command,
            "--output-format=json",
            "--fields=+nKsS",
            "--extras=-F",
            "--sort=no",
            *[self.project.relpath(item) for item in files],
        ]
        try:
            res = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._effective_timeout(timeout_ms),
            )
        except subprocess.TimeoutExpired:
            _fail("E_ANALYSIS_TIMEOUT", "ctags indexing timed out")
        if res.returncode not in {0, 1}:
            _fail("E_INTERNAL", res.stderr.strip() or "ctags failed")
        symbols = []
        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if raw.get("_type") != "tag":
                continue
            rel = str(raw.get("path", ""))
            abs_path = (self.root / rel).resolve()
            language = _language_for_path(abs_path)
            if language is None:
                continue
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            kind = _normalize_ctags_kind(str(raw.get("kind", "")), raw.get("scopeKind"), raw.get("signature"))
            line_no = int(raw.get("line", 1))
            symbols.append(
                SymbolRecord(
                    name=name,
                    kind=kind,
                    file_path=rel,
                    abs_path=abs_path,
                    line=line_no,
                    character=self._symbol_character(abs_path, line_no, name),
                    container=_clean_scope(raw.get("scope")),
                    scope_kind=_clean_scope(raw.get("scopeKind")),
                    signature=_clean_scope(raw.get("signature")),
                    language=language,
                    pattern=_clean_scope(raw.get("pattern")),
                )
            )
        self._tag_cache[include_generated] = TagCache(
            symbols=symbols,
            files=[self.project.relpath(item) for item in files],
            version=self._version,
        )
        return symbols

    def _filter_symbols(self, symbols: list[SymbolRecord], *, path: str | None, language: str | None) -> list[SymbolRecord]:
        result = symbols
        want_path = None
        if path:
            path_obj = self._resolve(path, must_exist=True)
            want_path = self.project.relpath(path_obj)
            if path_obj.is_dir():
                prefix = want_path.rstrip("/") + "/"
                result = [item for item in result if item.file_path == want_path or item.file_path.startswith(prefix)]
            else:
                result = [item for item in result if item.file_path == want_path]
        want_language = _normalize_language(language)
        if want_language is not None:
            result = [item for item in result if item.language == want_language]
        return result

    def _resolve_symbol(
        self,
        symbol: str,
        *,
        path_hint: str | None = None,
        language: str | None = None,
        prefer_exact: bool = True,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> tuple[SymbolRecord, list[SymbolRecord]]:
        if not symbol.strip():
            _fail("E_INPUT_INVALID", "symbol must not be empty")
        tags = self._filter_symbols(
            self._load_tags(include_generated=include_generated, timeout_ms=timeout_ms),
            path=path_hint,
            language=language,
        )
        exact_full = [item for item in tags if item.full_name == symbol]
        exact_name = [item for item in tags if item.name == symbol]
        fuzzy = [item for item in tags if item.full_name.endswith(f".{symbol}") or item.name == symbol]
        candidates = exact_full or exact_name or fuzzy
        if not candidates:
            _fail("E_SYMBOL_NOT_FOUND", f"symbol not found: {symbol}")
        ranked = sorted(candidates, key=lambda item: (item.full_name != symbol, item.name != symbol, item.file_path, item.line))
        first = ranked[0]
        alternatives = ranked[1:]
        if prefer_exact:
            exact_unique = [item for item in ranked if item.full_name == first.full_name or item.name == first.name]
            if len(exact_unique) > 1 and not path_hint:
                _fail("E_AMBIGUOUS_SYMBOL", f"multiple definitions match: {symbol}")
        return first, alternatives

    def _resolve_function(
        self,
        function: str,
        *,
        path_hint: str | None = None,
        language: str | None = None,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> tuple[SymbolRecord, list[SymbolRecord]]:
        rec, alternatives = self._resolve_symbol(
            symbol=function,
            path_hint=path_hint,
            language=language,
            prefer_exact=True,
            include_generated=include_generated,
            timeout_ms=timeout_ms,
        )
        if rec.kind not in {"function", "method"}:
            _fail("E_SYMBOL_NOT_FOUND", f"function not found: {function}")
        return rec, alternatives

    def _line_text(self, path: Path, line_no: int) -> str:
        lines = self._lines(path)
        if 1 <= line_no <= len(lines):
            return lines[line_no - 1]
        return ""

    def _lines(self, path: Path) -> list[str]:
        rel = self.project.relpath(path)
        lines = self._file_cache.get(rel)
        if lines is not None:
            return lines
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        self._file_cache[rel] = lines
        return lines

    def _symbol_character(self, path: Path, line_no: int, name: str) -> int:
        line = self._line_text(path, line_no)
        if not line:
            return 1
        match = re.search(rf"\b{re.escape(name)}\b", line)
        if match is None:
            return 1
        return match.start() + 1

    def _symbol_payload(self, item: SymbolRecord, *, include_doc: bool) -> dict[str, Any]:
        payload = {
            "name": item.name,
            "kind": item.kind,
            "file_path": item.file_path,
            "line": item.line,
            "character": item.character,
            "signature": item.signature,
            "doc": None,
        }
        if include_doc:
            payload["doc"] = self._line_text(item.abs_path, item.line).strip()
        return payload

    def _outline_tree(self, symbols: list[SymbolRecord], *, max_depth: int, include_private: bool) -> list[dict[str, Any]]:
        ordered = sorted(symbols, key=lambda item: (item.line, item.character, item.name))
        if not include_private:
            ordered = [item for item in ordered if not item.name.startswith("_")]
        ranges = {id(item): self._range_for_symbol(item) for item in ordered}
        nodes: dict[int, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []
        stack: list[tuple[SymbolRecord, dict[str, Any]]] = []
        for item in ordered:
            start_line, end_line = ranges[id(item)]
            node = {
                "name": item.name,
                "kind": item.kind,
                "range": {"start_line": start_line, "end_line": end_line},
                "children": [],
            }
            nodes[id(item)] = node
            while stack and not self._contains(ranges[id(stack[-1][0])], (start_line, end_line)):
                stack.pop()
            parent = None
            if item.container:
                for candidate, candidate_node in reversed(stack):
                    if candidate.name == item.container or candidate.full_name.endswith(f".{item.container}"):
                        parent = candidate_node
                        break
            if parent is None and stack:
                parent = stack[-1][1]
            if parent is None:
                roots.append(node)
            else:
                parent["children"].append(node)
            stack.append((item, node))
        self._trim_outline_depth(roots, depth=1, max_depth=max_depth)
        return roots

    def _range_for_symbol(self, item: SymbolRecord) -> tuple[int, int]:
        lines = self._lines(item.abs_path)
        if item.line > len(lines):
            return item.line, item.line
        if item.language == "python":
            return item.line, _python_block_end(lines, item.line)
        if item.language in {"c", "cpp", "go", "rust", "javascript", "typescript"}:
            return item.line, _brace_block_end(lines, item.line)
        return item.line, item.line

    def _trim_outline_depth(self, nodes: list[dict[str, Any]], *, depth: int, max_depth: int) -> None:
        for node in nodes:
            if depth >= max_depth:
                node["children"] = []
                continue
            self._trim_outline_depth(node["children"], depth=depth + 1, max_depth=max_depth)

    def _find_references(
        self,
        rec: SymbolRecord,
        *,
        path: str | None,
        include_generated: bool,
        timeout_ms: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self._effective_timeout(timeout_ms)
        result = []
        files = self._reference_files(rec, path=path, include_generated=include_generated)
        pat = re.compile(rf"\b{re.escape(rec.name)}\b")
        for file_path in files:
            if time.monotonic() > deadline:
                _fail("E_ANALYSIS_TIMEOUT", "reference search timed out")
            abs_path = (self.root / file_path).resolve()
            for idx, line in enumerate(self._lines(abs_path), start=1):
                for match in pat.finditer(line):
                    result.append(
                        {
                            "file_path": file_path,
                            "line": idx,
                            "character": match.start() + 1,
                            "snippet": line.strip(),
                        }
                    )
                    if len(result) >= limit:
                        return result
        return result

    def _reference_files(self, rec: SymbolRecord, *, path: str | None, include_generated: bool) -> list[str]:
        if path:
            target = self._resolve(path, must_exist=True)
            if target.is_file():
                return [self.project.relpath(target)]
            prefix = self.project.relpath(target).rstrip("/") + "/"
            return [self.project.relpath(item) for item in self._iter_source_files(target, include_generated=include_generated) if self.project.relpath(item).startswith(prefix)]
        return [self.project.relpath(item) for item in self._iter_source_files(self.root, include_generated=include_generated) if _language_for_path(item) == rec.language]

    def _iter_source_files(self, root: Path, *, include_generated: bool) -> Iterable[Path]:
        if root.is_file():
            if root.suffix in _SUPPORTED_EXTENSIONS:
                yield root
            return
        for item in root.rglob("*"):
            if item.is_dir():
                continue
            if item.suffix not in _SUPPORTED_EXTENSIONS:
                continue
            if not include_generated and _is_generated(item):
                continue
            yield item

    def _python_diagnostics(self, path: Path) -> list[dict[str, Any]]:
        rel = self.project.relpath(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(text, filename=rel)
            return []
        except SyntaxError as err:
            return [
                {
                    "file_path": rel,
                    "line": err.lineno or 1,
                    "character": err.offset or 1,
                    "severity": "error",
                    "code": "python-syntax",
                    "message": err.msg,
                }
            ]

    def _compiler_diagnostics(self, path: Path, language: str, *, timeout_ms: int | None) -> tuple[list[dict[str, Any]], list[str]]:
        cmd, cwd, warnings = self._compile_command(path, language)
        try:
            res = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._effective_timeout(timeout_ms),
            )
        except subprocess.TimeoutExpired:
            _fail("E_ANALYSIS_TIMEOUT", f"compiler diagnostics timed out for {path}")
        text = "\n".join(part for part in [res.stdout, res.stderr] if part)
        diags = []
        for line in text.splitlines():
            match = _COMPILER_DIAG.match(line.strip())
            if match is None:
                continue
            file_name, line_no, col_no, severity, message = match.groups()
            file_path = file_name
            try:
                file_path = self.project.relpath(Path(file_name))
            except Exception:
                pass
            sev = "information" if severity == "note" else ("warning" if severity == "warning" else "error")
            diags.append(
                {
                    "file_path": file_path,
                    "line": int(line_no),
                    "character": int(col_no or 1),
                    "severity": sev,
                    "code": None,
                    "message": message.strip(),
                }
            )
        return diags, warnings

    def _compile_command(self, path: Path, language: str) -> tuple[list[str], Path, list[str]]:
        db = self._compile_db()
        entry = db.get(str(path.resolve()))
        warnings: list[str] = []
        if entry is not None:
            raw = entry.get("arguments")
            if raw is None and entry.get("command"):
                raw = shlex.split(str(entry["command"]))
            if isinstance(raw, list):
                cmd = list(raw)
                cmd = _rewrite_compile_command(cmd, path)
                return cmd, Path(entry.get("directory", self.root)).resolve(), warnings
        compiler = _default_compiler(path, language)
        if compiler is None:
            _fail("E_UNSUPPORTED_LANGUAGE", f"no compiler available for {path.name}")
        warnings.append("REDUCED_PRECISION_NO_COMPILE_DB")
        cmd = [compiler, "-fsyntax-only", "-x", _compiler_language_arg(path, language), str(path)]
        return cmd, self.root, warnings

    def _compile_db(self) -> dict[str, dict[str, Any]]:
        if self._compile_db_cache is not None:
            return self._compile_db_cache
        path = self.root / "compile_commands.json"
        if not path.exists():
            self._compile_db_cache = {}
            return self._compile_db_cache
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._compile_db_cache = {}
            return self._compile_db_cache
        self._compile_db_cache = {}
        for item in rows:
            file_name = item.get("file")
            if not file_name:
                continue
            resolved = Path(item.get("directory", self.root)).joinpath(file_name).resolve()
            self._compile_db_cache[str(resolved)] = item
        return self._compile_db_cache

    def _function_symbols(
        self,
        *,
        path: str | None = None,
        language: str | None = None,
        include_generated: bool = False,
        timeout_ms: int | None = None,
    ) -> list[SymbolRecord]:
        items = self._filter_symbols(
            self._load_tags(include_generated=include_generated, timeout_ms=timeout_ms),
            path=path,
            language=language,
        )
        return [item for item in items if item.kind in {"function", "method"}]

    def _function_calls(self, item: SymbolRecord) -> list[dict[str, Any]]:
        start_line, end_line = self._range_for_symbol(item)
        lines = self._lines(item.abs_path)
        out = []
        for idx in range(start_line, min(end_line, len(lines)) + 1):
            line = lines[idx - 1]
            if item.language == "python":
                if line.lstrip().startswith(("def ", "class ", "#")):
                    continue
                out.extend(self._scan_call_line(line, idx, python=True))
            elif item.language in {"c", "cpp", "go", "rust", "javascript", "typescript"}:
                clean = re.sub(r"//.*$", "", line)
                out.extend(self._scan_call_line(clean, idx, python=False))
        return out

    def _scan_call_line(self, line: str, line_no: int, *, python: bool) -> list[dict[str, Any]]:
        result = []
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", line):
            name = match.group(1)
            if python and name in _PYTHON_CALL_SKIP:
                continue
            if not python and name in _CONTROL_KEYWORDS:
                continue
            result.append({"name": name, "line": line_no, "call_expression": line.strip()})
        return result

    def _resolve_callees(self, item: SymbolRecord, *, include_external: bool) -> list[dict[str, Any]]:
        out = []
        seen: set[tuple[str, str | None, int | None]] = set()
        for call in self._function_calls(item):
            resolved = self._best_function_match(call["name"], language=item.language, preferred_file=item.file_path, preferred_container=item.container)
            if resolved is None:
                if not include_external:
                    continue
                key = (call["name"], None, None)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"name": call["name"], "file_path": None, "line": None, "is_external": True})
                continue
            key = (resolved.full_name, resolved.file_path, resolved.line)
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": resolved.full_name, "file_path": resolved.file_path, "line": resolved.line, "is_external": False})
        return out

    def _best_function_match(
        self,
        name: str,
        *,
        language: str,
        preferred_file: str | None = None,
        preferred_container: str | None = None,
    ) -> SymbolRecord | None:
        candidates = [item for item in self._load_tags(include_generated=False, timeout_ms=None) if item.name == name and item.kind in {"function", "method"} and item.language == language]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                preferred_container is not None and item.container != preferred_container,
                preferred_file is not None and item.file_path != preferred_file,
                item.file_path,
                item.line,
            )
        )
        return candidates[0]

    def _next_chain_nodes(self, item: SymbolRecord, *, direction: Literal["forward", "backward"]) -> list[SymbolRecord]:
        if direction == "forward":
            return [self._best_function_match(call["name"], language=item.language, preferred_file=item.file_path, preferred_container=item.container) for call in self._function_calls(item) if self._best_function_match(call["name"], language=item.language, preferred_file=item.file_path, preferred_container=item.container) is not None]
        callers = []
        for candidate in self._function_symbols(include_generated=False):
            for call in self._function_calls(candidate):
                if call["name"] == item.name:
                    callers.append(candidate)
                    break
        seen: set[str] = set()
        out = []
        for candidate in callers:
            if candidate.full_name in seen:
                continue
            seen.add(candidate.full_name)
            out.append(candidate)
        return out

    def _chain_node(self, item: SymbolRecord) -> dict[str, Any]:
        return {"name": item.full_name, "file_path": item.file_path, "line": item.line}

    def _outline_symbols_for_file(self, rel_path: str, *, timeout_ms: int | None) -> list[dict[str, Any]]:
        return [
            {
                "name": item.full_name,
                "kind": item.kind,
                "range_start": self._range_for_symbol(item)[0],
                "range_end": self._range_for_symbol(item)[1],
                "file_path": item.file_path,
                "line": item.line,
            }
            for item in self._load_tags(include_generated=False, timeout_ms=timeout_ms)
            if item.file_path == rel_path
        ]

    def _nearest_symbol(self, symbols: list[dict[str, Any]], line_no: int) -> dict[str, Any] | None:
        before = [item for item in symbols if item["range_start"] <= line_no]
        if before:
            return max(before, key=lambda item: item["range_start"])
        if symbols:
            return min(symbols, key=lambda item: item["range_start"])
        return None

    def _range_symbol_payload(self, item: dict[str, Any]) -> dict[str, Any]:
        return {"name": item["name"], "kind": item["kind"], "range_start": item["range_start"], "range_end": item["range_end"]}

    def _suggested_tests(self, impacted_symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
        test_files = [item for item in self._iter_source_files(self.root, include_generated=False) if _is_test_file(item)]
        suggestions: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for test_file in test_files:
            rel = self.project.relpath(test_file)
            lines = self._lines(test_file)
            joined = "\n".join(lines)
            for impacted in impacted_symbols:
                base = impacted["name"].split(".")[-1]
                if not re.search(rf"\b{re.escape(base)}\b", joined):
                    continue
                key = (rel, base)
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append({"name": test_file.stem, "file_path": rel, "reason": f"references {base}"})
        return suggestions

    def _lsp_enabled(self, path: Path) -> bool:
        return bool(self.cfg.prefer_lsp and self.lsp is not None and self.lsp.available(path))

    @staticmethod
    def _contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
        return outer[0] <= inner[0] and outer[1] >= inner[1]

    @staticmethod
    def _intersects(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
        return start_a <= end_b and start_b <= end_a


def _normalize_ctags_kind(kind: str, scope_kind: object, signature: object) -> str:
    raw = kind.strip().lower()
    scope = str(scope_kind).lower() if scope_kind is not None else ""
    if raw in {"function", "member"} and signature and scope == "class":
        return "method"
    if raw in {"function", "member"} and signature and scope in {"struct", "namespace"}:
        return "function"
    if raw in {"function", "member"} and signature:
        return "function"
    if raw == "class":
        return "class"
    if raw in {"namespace"}:
        return raw
    if raw in {"struct", "union", "enum"}:
        return raw
    if raw in {"prototype"}:
        return "function"
    if raw in {"macro"}:
        return "macro"
    if raw in {"typedef"}:
        return "type"
    if raw in {"variable", "local", "enumerator", "label"}:
        return "variable"
    return raw or "symbol"


def _normalize_kind_name(kind: str | None) -> str | None:
    if kind is None:
        return None
    norm = kind.strip().lower()
    if norm in {"method", "function", "class", "variable", "module", "namespace", "macro", "type", "struct", "union", "enum"}:
        return norm
    return norm


def _normalize_language(language: str | None) -> str | None:
    if language is None or not language.strip():
        return None
    norm = language.strip().lower()
    if norm in {"cxx", "c++"}:
        return "cpp"
    if norm in {"py"}:
        return "python"
    if norm in {"js"}:
        return "javascript"
    if norm in {"ts"}:
        return "typescript"
    return norm


def _language_for_path(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _PYTHON_EXTENSIONS:
        return "python"
    if ext in _C_EXTENSIONS:
        return "c"
    if ext in _CPP_EXTENSIONS:
        return "cpp"
    if ext in _HEADER_EXTENSIONS:
        return _header_language(path)
    if ext in _GO_EXTENSIONS:
        return "go"
    if ext in _RUST_EXTENSIONS:
        return "rust"
    if ext in _JS_EXTENSIONS:
        return "javascript"
    if ext in _TS_EXTENSIONS:
        return "typescript"
    return None


def _header_language(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"\b(namespace|class|template|std::)\b", text):
        return "cpp"
    return "c"


def _clean_scope(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _python_block_end(lines: list[str], start_line: int) -> int:
    base = lines[start_line - 1]
    indent = len(base) - len(base.lstrip())
    if not base.rstrip().endswith(":"):
        return start_line
    end = start_line
    for idx in range(start_line, len(lines)):
        cur = lines[idx]
        if not cur.strip():
            continue
        cur_indent = len(cur) - len(cur.lstrip())
        if cur_indent <= indent:
            break
        end = idx + 1
    return max(end, start_line)


def _brace_block_end(lines: list[str], start_line: int) -> int:
    balance = 0
    seen = False
    end = start_line
    for idx in range(start_line - 1, len(lines)):
        line = re.sub(r"//.*$", "", lines[idx])
        balance += line.count("{")
        if "{" in line:
            seen = True
        balance -= line.count("}")
        if seen:
            end = idx + 1
            if balance <= 0:
                return end
        elif ";" in line:
            return start_line
    return end


def _rewrite_compile_command(cmd: list[str], path: Path) -> list[str]:
    out: list[str] = []
    skip_next = False
    for idx, item in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if item in {"-c", "-o"}:
            skip_next = item == "-o"
            continue
        if item == str(path):
            out.append(item)
            continue
        out.append(item)
    if "-fsyntax-only" not in out:
        out.append("-fsyntax-only")
    if str(path) not in out:
        out.append(str(path))
    return out


def _default_compiler(path: Path, language: str) -> str | None:
    if language == "c":
        return shutil.which("clang") or shutil.which("gcc")
    if language == "cpp":
        return shutil.which("clang++") or shutil.which("g++")
    _ = path
    return None


def _compiler_language_arg(path: Path, language: str) -> str:
    if language == "c":
        return "c-header" if path.suffix.lower() in _HEADER_EXTENSIONS else "c"
    return "c++-header" if path.suffix.lower() in _HEADER_EXTENSIONS else "c++"


def _severity_rank(value: str | None) -> int:
    order = {"information": 1, "warning": 2, "error": 3}
    return order.get((value or "information").lower(), 1)


def _is_generated(path: Path) -> bool:
    if any(part in _GENERATED_PARTS for part in path.parts):
        return True
    name = path.name.lower()
    return ".generated." in name or name.endswith(".min.js")


def _is_test_file(path: Path) -> bool:
    name = path.name.lower()
    return "test" in name or any(part.lower() == "tests" for part in path.parts)


def _fail(code: str, message: str) -> None:
    exc_type = ValueError if code in _NON_RETRYABLE_CODES else RuntimeError
    raise exc_type(f"{code}: {message}")
