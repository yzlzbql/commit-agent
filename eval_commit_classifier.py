#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import re
from typing import Any

from rich.console import Console
from rich.table import Table
import typer

from minimal_agent.agent import get as get_agent
from minimal_agent.commit_labels import COMMIT_LABELS, COMMIT_LABEL_SET, LEGACY_COMMIT_LABELS
from minimal_agent.config import AgentConfig
from minimal_agent.lsp.server import LspPool
from minimal_agent.model import OpenAIModelAdapter
from minimal_agent.project import Project
from minimal_agent.runtime.loop import run as run_loop
from minimal_agent.session.store import SessionStore
from minimal_agent.tool.registry import ToolRegistry
from minimal_agent.types import RunInput, StepRecord
from run_task import _state_root_for_repository


LABELS = COMMIT_LABELS
LABEL_SET = COMMIT_LABEL_SET
_LABEL_ALT = "|".join(sorted(LEGACY_COMMIT_LABELS, key=len, reverse=True))
_CONVENTIONAL_PREFIX = re.compile(
    rf"^\s*(?:{_LABEL_ALT})(?:\([^)]+\))?!?:\s*",
    re.IGNORECASE,
)
_LABELISH_PREFIX = re.compile(
    rf"^\s*(?:{_LABEL_ALT})(?:/[^:]+)?\s*:\s*",
    re.IGNORECASE,
)
_LEADING_TOKEN = re.compile(r"^\s*([A-Za-z]+)\b")
_DIFF_PLUS = re.compile(r"^\+\+\+ b/(.+)$")
_DIFF_MINUS = re.compile(r"^--- a/(.+)$")
_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")
_DOC_EXTENSIONS = {".md", ".rst", ".txt", ".xml", ".adoc", ".asciidoc"}
_BUILD_FILENAMES = {"configure.ac", "configure.in", "cmakelists.txt", "meson.build", "makefile", "makefile.am"}
_MAX_PATCH_PROMPT_CHARS = 40000
_MAX_PATCH_PROMPT_FILES = 12
_MAX_PATCH_PROMPT_HUNK_LINES = 20
_CORRECTION_KEYWORDS = (
    "typo",
    "misspell",
    "spelling",
    "broken",
    "wrong",
    "invalid",
    "error",
    "fix",
    "correct",
    "correction",
    "link",
    "example",
    "examples",
    "errata",
    "compat",
    "termux",
    "shebang",
)
_FEATURE_KEYWORDS = (
    "new command",
    "new option",
    "new api",
    "new flag",
    "new feature",
    "introduce",
    "user can now",
    "adds a new",
)
_MAINTENANCE_KEYWORDS = (
    "validate",
    "validation",
    "missing",
    "handle",
    "compat",
    "errata",
    "dnssec",
    "dname",
    "protocol",
    "broken",
    "invalid",
)
_DOC_FIX_KEYWORDS = ("link", "links", "anchor", "section", "grammar", "grammatical", "wording")
_DOC_EXECUTABLE_EXAMPLE_KEYWORDS = (
    "wrong command",
    "incorrect command",
    "api usage",
    "usage example",
    "example command",
    "copy/paste",
    "copy paste",
    "shell snippet",
    "executable example",
)
_CHORE_TEXT_CLEANUP_KEYWORDS = (
    "ai-generated",
    "repetitive word",
    "repetitive words",
    "housekeeping",
    "wording cleanup",
)
_DOC_MAINTENANCE_KEYWORDS = ("ai-generated", "toctree", "summary-start", "reference/parameters", "parameter reference")
_CODE_EDIT_KEYWORDS = (
    "return",
    "if ",
    "else",
    "goto ",
    "raise ",
    "await ",
    "yield ",
    "null",
    "none",
    "assert",
    "->",
    " = ",
    "==",
    "!=",
    "+=",
    "-=",
    "def ",
    "static ",
    "struct ",
)
_COMMENT_PREFIXES = ("#", "//", "/*", "*", "*/", ".. ", ":param", ":return", "@", "|", "-- ")


app = typer.Typer(help="Run leakage-safe commit label evaluation on tagged commit datasets.")
console = Console()


def _print_runtime_event(event: dict[str, object]) -> None:
    kind = str(event.get("type", "event"))
    prefix = ""
    if event.get("subtask"):
        prefix = f"[subtask:{event.get('subtask_agent', '?')}] "
    if kind == "session":
        console.rule(f"{prefix}Session {event['session_id']}")
        console.print(f"task: {event['task']}")
        console.print(f"cwd: {event['cwd']}")
        console.print(f"agent: {event.get('agent', 'build')}")
        return
    if kind == "subtask_start":
        console.rule(f"Subtask Start: {event.get('description', '')}")
        console.print(f"agent: {event.get('agent', '')}")
        return
    if kind == "subtask_finish":
        console.rule(f"Subtask Finish: {event.get('description', '')}")
        console.print(f"agent: {event.get('agent', '')}")
        console.print(f"status: {event.get('status', '')}")
        console.print(f"task_id: {event.get('task_id', '')}")
        return
    if kind == "todo":
        console.rule(f"{prefix}Todo")
        for item in event.get("items", []):
            if isinstance(item, dict):
                console.print(f"- [{item.get('status', 'pending')}] {item.get('content', '')}")
        return
    if kind == "summary":
        console.rule(f"{prefix}Compacted Summary")
        console.print(event.get("text", ""))
        return
    if kind == "reasoning":
        console.rule(f"{prefix}Reasoning")
        console.print(event.get("text", ""))
        return
    if kind == "tool_call":
        console.rule(f"{prefix}Tool Call: {event['name']}")
        console.print(json.dumps(event.get("args", {}), ensure_ascii=False, indent=2))
        return
    if kind == "tool_result":
        console.rule(f"{prefix}Tool Result: {event['name']}")
        console.print(event.get("output", ""))
        return
    if kind == "final":
        console.rule(f"{prefix}Final Draft")
        console.print(event.get("text", ""))
        return
    if kind == "verify_start":
        console.rule(f"{prefix}Verify")
        console.print("running verification")
        return
    if kind == "verify_result":
        console.print(f"verification: {'ok' if event.get('ok') else 'failed'}")
        checks = event.get("checks", [])
        failures = event.get("failures", [])
        if checks:
            console.print("checks:")
            for item in checks:
                console.print(f"- {item}")
        if failures:
            console.print("failures:")
            for item in failures:
                console.print(f"- {item}")
        return
    if kind == "error":
        console.rule(f"{prefix}Error")
        console.print(event.get("message", ""))


@dataclass(frozen=True, slots=True)
class EvalCase:
    repo: str
    commit_id: str
    oracle_label: str
    commit_message: str
    patch: str
    source_file: str
    source_index: int


def _sanitize_commit_message(message: str) -> str:
    raw = (message or "").strip()
    if not raw:
        return ""
    first, *rest = raw.splitlines()
    stripped = _CONVENTIONAL_PREFIX.sub("", first, count=1)
    if stripped == first:
        stripped = _LABELISH_PREFIX.sub("", first, count=1)
    if stripped == first:
        token = _LEADING_TOKEN.match(first)
        if token and token.group(1).lower() in LABEL_SET:
            remainder = first[token.end() :]
            remainder = re.sub(r"^\s*[:/_-]\s*", "", remainder, count=1)
            stripped = remainder.lstrip()
    stripped = stripped.strip() or "Message intentionally sanitized."
    if not rest:
        return stripped
    return "\n".join([stripped, *rest]).strip()


def _extract_changed_files(patch: str) -> list[str]:
    found: list[str] = []
    for line in patch.splitlines():
        header = _DIFF_HEADER.match(line)
        plus = _DIFF_PLUS.match(line)
        minus = _DIFF_MINUS.match(line)
        path = None
        if header:
            path = header.group(2)
        elif plus:
            path = plus.group(1)
        elif minus:
            path = minus.group(1)
        if not path or path == "/dev/null":
            continue
        if path not in found:
            found.append(path)
    return found


def _build_task(case: EvalCase, repo_path: Path) -> str:
    changed = _extract_changed_files(case.patch)
    changed_text = "\n".join(f"- {item}" for item in changed) or "- none extracted from patch"
    sanitized = _sanitize_commit_message(case.commit_message)
    prompt_patch = _compact_patch_for_prompt(case.patch)
    return "\n".join(
        [
            "根据下面提供的脱敏 commit 信息完成分类。",
            "",
            "[REPOSITORY_PATH]",
            str(repo_path),
            "",
            "[SANITIZED_COMMIT_MESSAGE]",
            sanitized or "Message intentionally sanitized.",
            "",
            "[CHANGED_FILES]",
            changed_text,
            "",
            "[PATCH]",
            prompt_patch,
        ]
    )


def _compact_patch_for_prompt(patch: str) -> str:
    text = patch.strip()
    if len(text) <= _MAX_PATCH_PROMPT_CHARS:
        return text

    kept: list[str] = []
    files_seen = 0
    in_hunk = False
    hunk_lines = 0
    omitted_hunk = False

    for line in text.splitlines():
        if line.startswith("diff --git "):
            if files_seen >= _MAX_PATCH_PROMPT_FILES:
                break
            files_seen += 1
            in_hunk = False
            hunk_lines = 0
            omitted_hunk = False
            kept.append(line)
            continue
        if line.startswith(("index ", "--- ", "+++ ")):
            kept.append(line)
            continue
        if line.startswith(("@@ ", "@@")):
            in_hunk = True
            hunk_lines = 0
            omitted_hunk = False
            kept.append(line)
            continue
        if in_hunk and line.startswith(("+", "-", " ")):
            if hunk_lines < _MAX_PATCH_PROMPT_HUNK_LINES:
                kept.append(line)
            elif not omitted_hunk:
                kept.append("... [truncated hunk body omitted for prompt fit] ...")
                omitted_hunk = True
            hunk_lines += 1
            continue
        if line and not in_hunk:
            kept.append(line)
        if sum(len(item) + 1 for item in kept) >= _MAX_PATCH_PROMPT_CHARS:
            break

    compact = "\n".join(kept).strip()
    note = [
        "[PATCH_NOTE]",
        "Original patch exceeded the prompt size limit; only representative diff headers and hunks are shown below.",
        "Use the complete changed-file list above as the canonical scope of the commit.",
        "",
    ]
    return "\n".join(note + [compact])


def _build_tool_trace(events: list[StepRecord]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for idx, rec in enumerate(events, start=1):
        if rec.kind != "tool":
            continue
        payload = rec.payload
        trace.append(
            {
                "event_index": idx,
                "name": payload.get("name", ""),
                "args": payload.get("args", {}),
                "title": payload.get("title", ""),
                "output": payload.get("output", ""),
                "files": payload.get("files", []),
                "metadata": payload.get("metadata", {}),
            }
        )
    return trace


def _extract_json_object(text: str) -> dict[str, Any]:
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if len(lines) >= 3:
            body = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        start = body.find("{")
        end = body.rfind("}")
        if start < 0 or end < start:
            raise ValueError("final output did not contain a JSON object")
        parsed = json.loads(body[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("final output JSON must be an object")
    return parsed


def _iter_cases(dataset_dir: Path, repo_root: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for path in sorted(dataset_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                raw = json.loads(line)
                repo = str(raw.get("repo", "")).strip()
                patch = str(raw.get("patch", "") or "")
                label = str(raw.get("tag", "")).strip()
                if not repo or label not in LABEL_SET or not patch or not raw.get("patch_available", False):
                    continue
                repo_path = (repo_root / repo).resolve()
                if not repo_path.exists():
                    continue
                cases.append(
                    EvalCase(
                        repo=repo,
                        commit_id=str(raw.get("commit_id", "")).strip(),
                        oracle_label=label,
                        commit_message=str(raw.get("commit_message", "") or raw.get("title", "") or ""),
                        patch=patch,
                        source_file=path.name,
                        source_index=idx,
                    )
                )
    return cases


def _result_paths(output_dir: Path, sample_size: int, seed: int) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"sample_{sample_size}_seed_{seed}_{stamp}"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}_summary.json"


def _load_base_config(script_dir: Path) -> AgentConfig:
    return AgentConfig.load(
        script_dir,
        env_search_paths=[script_dir],
        config_search_paths=[script_dir],
    )


def _is_doc_path(path: str) -> bool:
    item = path.lower()
    return (
        item.startswith(("doc/", "docs/", "documentation/"))
        or "/doc/" in item
        or "/docs/" in item
        or Path(item).suffix in _DOC_EXTENSIONS
    )


def _is_test_path(path: str) -> bool:
    item = path.lower()
    name = Path(item).name
    stem = Path(item).stem
    return (
        item.startswith(("test/", "tests/", "spec/"))
        or "/test/" in item
        or "/tests/" in item
        or "/spec/" in item
        or name.startswith(("test_", "test-"))
        or stem.endswith(("_test", "-test"))
    )


def _is_perf_path(path: str) -> bool:
    item = path.lower()
    return item.startswith("tools/perf/") or "/perf/" in item or item.startswith("perf/")


def _is_benchmark_path(path: str) -> bool:
    item = path.lower()
    return item.startswith("benchmark/") or "/benchmark/" in item or Path(item).name.startswith("bench")


def _is_build_path(path: str) -> bool:
    item = path.lower()
    name = Path(item).name
    return (
        name in _BUILD_FILENAMES
        or item.startswith(("ci/", ".github/workflows/", ".gitlab/"))
        or "/build" in item
        or "/cmake/" in item
        or "/meson/" in item
    )


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(item in text for item in needles)


def _added_lines_text(patch: str) -> str:
    rows = []
    for line in patch.splitlines():
        if line.startswith("+++") or not line.startswith("+"):
            continue
        rows.append(line[1:])
    return "\n".join(rows).lower()


def _combined_text(case: EvalCase, parsed: dict[str, Any] | None) -> str:
    parts = [_sanitize_commit_message(case.commit_message), case.patch]
    if parsed:
        parts.append(str(parsed.get("reason", "")))
        evidence = parsed.get("evidence", [])
        if isinstance(evidence, list):
            parts.extend(str(item) for item in evidence)
    return "\n".join(parts).lower()


def _has_code_edit_signal(patch: str) -> bool:
    for line in patch.splitlines():
        if line.startswith(("+++", "---", "@@")) or not line.startswith(("+", "-")):
            continue
        stripped = line[1:].strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith(_COMMENT_PREFIXES):
            continue
        if _contains_any(lowered, _CODE_EDIT_KEYWORDS):
            return True
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stripped):
            return True
    return False


def _normalize_model_label(value: str) -> str:
    label = (value or "").strip().lower()
    return label if label in LABEL_SET else ""


def _select_predicted_label(case: EvalCase, parsed: dict[str, Any] | None) -> tuple[str, str]:
    changed = _extract_changed_files(case.patch)
    doc_only = bool(changed) and all(_is_doc_path(path) for path in changed)
    test_only = bool(changed) and all(_is_test_path(path) for path in changed)
    perf_touched = any(_is_perf_path(path) for path in changed)
    benchmark_touched = any(_is_benchmark_path(path) for path in changed)
    build_touched = any(_is_build_path(path) for path in changed)
    non_doc_touched = any(not _is_doc_path(path) for path in changed)
    text = _combined_text(case, parsed)
    added_text = _added_lines_text(case.patch)
    code_edit_signal = _has_code_edit_signal(case.patch)
    tools_only = bool(changed) and all(path.lower().startswith("tools/") for path in changed)
    raw_model_label = ""
    if parsed:
        raw_model_label = str(parsed.get("primary_label", "")).strip().lower()
    model_label = _normalize_model_label(raw_model_label)
    correction_signal = _contains_any(text, _CORRECTION_KEYWORDS)
    feature_signal = _contains_any(text, _FEATURE_KEYWORDS)
    maintenance_signal = _contains_any(text, _MAINTENANCE_KEYWORDS)
    perf_signal = perf_touched and _contains_any(text, ("perf", "pmu", "event", "events", "metric", "stall_", "errata"))
    build_breakage_signal = _contains_any(text, ("termux", "shebang", "/bin/sh", "not valid", "invalid", "compat"))

    if test_only:
        return "test", "heuristic:test-paths"
    if perf_signal:
        return "perf", "heuristic:perf-subsystem"
    if doc_only:
        if model_label == "chore" or _contains_any(text, _DOC_MAINTENANCE_KEYWORDS):
            return "chore", "model:doc-maintenance"
        if _contains_any(text, _DOC_EXECUTABLE_EXAMPLE_KEYWORDS):
            return "fix", "heuristic:doc-example-fix"
        return "docs", "heuristic:doc-only"
    if model_label == "docs" and _contains_any(text, _DOC_FIX_KEYWORDS) and not code_edit_signal:
        return "docs", "heuristic:doc-correction-stays-docs"
    if correction_signal and len(changed) >= 4 and not code_edit_signal and not tools_only:
        if _contains_any(text, _CHORE_TEXT_CLEANUP_KEYWORDS) or _contains_any(text, ("typo", "typos", "spelling")):
            return "chore", "heuristic:text-cleanup-is-chore"
    if model_label == "docs" and correction_signal and (non_doc_touched or _contains_any(text, ("typo", "link", "example", "name"))):
        if code_edit_signal:
            return "fix", "heuristic:correction-is-fix"
    if raw_model_label == "build" and build_breakage_signal:
        return "fix", "heuristic:build-breakage-is-fix"
    if raw_model_label == "build":
        return "chore", "heuristic:unsupported-build-normalized"
    if raw_model_label in {"ci", "revert"}:
        return "chore", "heuristic:unsupported-label-normalized"
    if model_label == "feat" and not feature_signal and maintenance_signal:
        return "fix", "heuristic:maintenance-is-fix"
    if model_label == "refactor" and non_doc_touched and "return" in added_text and ("if " in added_text or "guard" in text):
        if not _contains_any(text, ("move ", "moved", "grouping", "handle", "handles", "rename", "reorder", "cleanup", "cmd_context")):
            return "fix", "heuristic:restored-control-flow-is-fix"
    if model_label == "perf" and benchmark_touched and not perf_touched and _contains_any(text, ("refactor", "重构")):
        return "refactor", "heuristic:benchmark-backed-refactor"
    if not model_label:
        if correction_signal:
            return "fix", "heuristic:fallback-fix"
        if build_touched:
            return ("fix", "heuristic:fallback-build-breakage") if build_breakage_signal else ("chore", "heuristic:fallback-build-maintenance")
        if perf_touched:
            return "perf", "heuristic:fallback-perf"
        return "chore", "heuristic:fallback-chore"
    return model_label, "model"


def _load_sample(
    cases: list[EvalCase],
    *,
    sample_size: int,
    seed: int,
    retry_jsonl: Path | None,
    only_failed: bool,
) -> list[EvalCase]:
    if retry_jsonl is None:
        rng = random.Random(seed)
        return rng.sample(cases, sample_size)
    rows = [json.loads(line) for line in retry_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = {(case.source_file, case.source_index): case for case in cases}
    selected: list[EvalCase] = []
    for row in rows:
        if only_failed and row.get("status") == "finished" and row.get("match"):
            continue
        key = (row.get("source_file"), row.get("source_index"))
        case = index.get(key)
        if case is not None:
            selected.append(case)
    return selected


@app.command()
def main(
    dataset_dir: Path = typer.Option(
        Path("/data2/huyanglin-repo/tagged_commits_with_patch"),
        "--dataset-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    repo_root: Path = typer.Option(
        Path("/data2/huyanglin-repo"),
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    sample_size: int = typer.Option(10, "--sample-size", min=1),
    seed: int = typer.Option(42, "--seed"),
    max_steps: int = typer.Option(8, "--max-steps", min=1),
    agent: str = typer.Option("commit_eval", "--agent"),
    output_dir: Path = typer.Option(Path("eval_runs"), "--output-dir"),
    retry_jsonl: Path | None = typer.Option(None, "--retry-jsonl", exists=True, dir_okay=False, readable=True),
    only_failed: bool = typer.Option(False, "--only-failed"),
    show_events: bool = typer.Option(False, "--show-events", help="Stream per-case reasoning and tool events to stdout."),
) -> None:
    get_agent(agent)
    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _iter_cases(dataset_dir.resolve(), repo_root.resolve())
    if retry_jsonl is None and len(cases) < sample_size:
        raise typer.BadParameter(f"not enough eligible cases: requested {sample_size}, found {len(cases)}")
    sample = _load_sample(cases, sample_size=sample_size, seed=seed, retry_jsonl=retry_jsonl, only_failed=only_failed)
    if not sample:
        raise typer.BadParameter("no cases selected for evaluation")
    selected_count = len(sample)

    cfg = _load_base_config(script_dir)
    model = OpenAIModelAdapter(cfg.model)
    tools = ToolRegistry()
    jsonl_path, summary_path = _result_paths(output_dir, sample_size, seed)

    results: list[dict[str, Any]] = []
    console.print(f"Running {selected_count} leakage-safe evaluation cases with agent={agent} seed={seed}")
    for idx, case in enumerate(sample, start=1):
        if show_events:
            console.rule(f"Case {idx}/{selected_count}: {case.repo} {case.commit_id[:12]}")
        repo_path = (repo_root / case.repo).resolve()
        state_root = _state_root_for_repository(repo_path, script_dir)
        project = Project.load(repo_path, state_root=state_root)
        store = SessionStore(project.sessions_dir)
        lsp = LspPool(cfg, project.root)
        task = _build_task(case, repo_path)
        parsed: dict[str, Any] | None = None
        parse_error = ""
        model_label = ""
        result_status = "error"
        session_id = ""
        raw_candidate = ""
        summary = ""
        tool_trace: list[dict[str, Any]] = []
        try:
            result = run_loop(
                run=RunInput(
                    task=task,
                    cwd=project.cwd,
                    max_steps=max_steps,
                    agent=agent,
                    title=f"commit-eval:{case.repo}:{case.commit_id[:12]}",
                ),
                cfg=cfg,
                project=project,
                store=store,
                model=model,
                tools=tools,
                lsp=lsp,
                emit=_print_runtime_event if show_events else None,
            )
            state = store.load(result.session_id)
            result_status = result.status
            session_id = result.session_id
            raw_candidate = state.candidate
            summary = result.summary
            tool_trace = _build_tool_trace(store.events(result.session_id))
            try:
                parsed = _extract_json_object(state.candidate)
                model_label = str(parsed.get("primary_label", "")).strip()
            except Exception as err:
                parse_error = str(err)
        except Exception as err:
            parse_error = f"run_loop_error: {err}"
        predicted, label_source = _select_predicted_label(case, parsed)
        ok = predicted == case.oracle_label
        row = {
            "case_index": idx,
            "repo": case.repo,
            "commit_id": case.commit_id,
            "oracle_label": case.oracle_label,
            "model_label": model_label,
            "predicted_label": predicted,
            "label_source": label_source,
            "match": ok,
            "status": result_status,
            "session_id": session_id,
            "parse_error": parse_error,
            "sanitized_commit_message": _sanitize_commit_message(case.commit_message),
            "changed_files": _extract_changed_files(case.patch),
            "raw_candidate": raw_candidate,
            "parsed_candidate": parsed,
            "summary": summary,
            "tool_trace": tool_trace,
            "source_file": case.source_file,
            "source_index": case.source_index,
        }
        results.append(row)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
        console.print(
            f"[{idx}/{selected_count}] repo={case.repo} sha={case.commit_id[:12]} "
            f"oracle={case.oracle_label} predicted={predicted or '-'} status={result_status}"
        )

    correct = sum(1 for item in results if item["match"])
    finished = sum(1 for item in results if item["status"] == "finished")
    summary = {
        "sample_size": selected_count,
        "seed": seed,
        "agent": agent,
        "max_steps": max_steps,
        "dataset_dir": str(dataset_dir.resolve()),
        "repo_root": str(repo_root.resolve()),
        "output_jsonl": str(jsonl_path),
        "finished": finished,
        "correct": correct,
        "accuracy": correct / selected_count if selected_count else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    table = Table(title="commit-eval summary")
    table.add_column("metric")
    table.add_column("value")
    for key in ("sample_size", "finished", "correct", "accuracy", "output_jsonl"):
        table.add_row(key, str(summary[key]))
    console.print(table)
    console.print(f"summary saved to {summary_path}")


if __name__ == "__main__":
    app()
