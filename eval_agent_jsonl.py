#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import random
import subprocess
import threading
from typing import Any

from rich.console import Console
from rich.table import Table
import typer

from minimal_agent.commit_labels import COMMIT_LABELS, COMMIT_LABEL_SET

LABELS = COMMIT_LABELS
LABEL_SET = COMMIT_LABEL_SET
REPO_ALIASES = {
    "cpython": "python3",
    "go": "Golang",
    "glib": "glib2",
    "kernel": "Kernel",
    "linux": "Kernel",
    "networkmanager": "Networkmanager",
    "NetworkManager": "Networkmanager",
}
AUTO_SAMPLE_30_NAME = "auto_sample_30_each"
AUTO_SAMPLE_30_SPECS: tuple[tuple[str, dict[str, int]], ...] = (
    ("data/kernel_tagged_commits.jsonl", {"feat": 30, "fix": 30}),
    ("data/oracle_commits.jsonl", {"refactor": 30, "docs": 30, "test": 30}),
)

app = typer.Typer(help="Evaluate commit-label predictions against a JSONL dataset using the local agent runtime.")
console = Console()
_THREAD_STATE = threading.local()


def _load_base_config(script_dir: Path) -> AgentConfig:
    from minimal_agent.config import AgentConfig

    return AgentConfig.load(
        script_dir,
        env_search_paths=[script_dir],
        config_search_paths=[script_dir],
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    body = (text or "").strip()
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


def _repair_json_candidate(model: OpenAIModelAdapter, raw_candidate: str) -> dict[str, Any] | None:
    body = (raw_candidate or "").strip()
    if not body:
        return None
    msgs = [
        {
            "role": "system",
            "content": (
                "You repair malformed model output into a single valid JSON object. "
                "Return exactly one JSON object and no markdown, no prose, and no code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite the following text into one valid JSON object. "
                "Preserve the original meaning and fields when possible.\n\n"
                f"{body}"
            ),
        },
    ]
    try:
        out = list(model.stream(msgs, []))
    except Exception:
        return None
    for item in out:
        if item.get("type") == "final":
            try:
                return _extract_json_object(str(item.get("text", "")))
            except Exception:
                return None
    return None


def _replace_marker_value(task_template: str, marker: str, value: str) -> str:
    lines = task_template.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != marker:
            continue
        if idx + 1 >= len(lines):
            raise ValueError(f"marker {marker} exists but no following line to replace")
        lines[idx + 1] = value
        return "\n".join(lines)
    raise ValueError(f"marker not found in task template: {marker}")


def _build_task(task_template: str, repo_path: Path, commit_id: str) -> str:
    task = _replace_marker_value(task_template, "[REPOSITORY_PATH]", str(repo_path))
    return _replace_marker_value(task, "[COMMIT]", commit_id)


def _normalize_label(value: object) -> str:
    label = str(value or "").strip().lower()
    return label if label in LABEL_SET else ""


def _resolve_repo_path(repo: str, repo_root: Path) -> tuple[Path | None, str]:
    raw = (repo or "").strip()
    if not raw:
        return None, ""
    direct = (repo_root / raw).resolve()
    if direct.is_dir():
        return direct, raw
    alias = REPO_ALIASES.get(raw)
    if alias:
        aliased = (repo_root / alias).resolve()
        if aliased.is_dir():
            return aliased, alias
    lower_map = {item.name.lower(): item for item in repo_root.iterdir() if item.is_dir()}
    for key in {raw.lower(), str(alias or "").lower()}:
        if key and key in lower_map:
            return lower_map[key].resolve(), lower_map[key].name
    return None, alias or raw


def _commit_exists(repo_path: Path, commit_id: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--verify", f"{commit_id}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _result_paths(output_dir: Path, input_jsonl: Path | str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_ref = input_jsonl if isinstance(input_jsonl, Path) else Path(input_jsonl)
    stem = f"{input_ref.stem}_{stamp}"
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}_summary.json"


def _dataset_row(raw: dict[str, Any], *, source_index: int, source_jsonl: Path) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "source_jsonl": str(source_jsonl),
        "repo": str(raw.get("repo", "")).strip(),
        "commit_id": str(raw.get("commit_id", "")).strip(),
        "oracle_label": _normalize_label(raw.get("tag")),
    }


def _load_input_rows(
    input_jsonl: Path,
    *,
    only_repo: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_jsonl = input_jsonl.resolve()
    with source_jsonl.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            raw = json.loads(line)
            repo = str(raw.get("repo", "")).strip()
            if only_repo and repo != only_repo:
                continue
            rows.append(_dataset_row(raw, source_index=idx, source_jsonl=source_jsonl))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _load_auto_sample_rows(
    script_dir: Path,
    *,
    only_repo: str | None,
    limit: int | None,
    sample_seed: int | None,
) -> list[dict[str, Any]]:
    rng = random.Random(sample_seed)
    rows: list[dict[str, Any]] = []
    for relative_path, label_counts in AUTO_SAMPLE_30_SPECS:
        source_jsonl = (script_dir / relative_path).resolve()
        by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in label_counts}
        with source_jsonl.open(encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                raw = json.loads(line)
                repo = str(raw.get("repo", "")).strip()
                if only_repo and repo != only_repo:
                    continue
                row = _dataset_row(raw, source_index=idx, source_jsonl=source_jsonl)
                label = row["oracle_label"]
                if label in by_label:
                    by_label[label].append(row)
        for label, count in label_counts.items():
            candidates = by_label[label]
            if len(candidates) < count:
                raise typer.BadParameter(
                    f"{relative_path} only has {len(candidates)} rows for label={label}, need {count}"
                )
            rows.extend(rng.sample(candidates, count))
    rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _project_bundle(
    repo_path: Path,
    script_dir: Path,
    cfg: AgentConfig,
    cache: dict[Path, tuple[Project, SessionStore, LspPool]],
) -> tuple[Project, SessionStore, LspPool]:
    from minimal_agent.lsp.server import LspPool
    from minimal_agent.project import Project
    from minimal_agent.session.store import SessionStore
    from run_task import _state_root_for_repository

    resolved = repo_path.resolve()
    cached = cache.get(resolved)
    if cached is not None:
        return cached
    state_root = _state_root_for_repository(resolved, script_dir)
    project = Project.load(resolved, state_root=state_root)
    store = SessionStore(project.sessions_dir)
    lsp = LspPool(cfg, project.root)
    cache[resolved] = (project, store, lsp)
    return project, store, lsp


def _thread_runtime(
    script_dir: Path,
    cfg: AgentConfig,
) -> tuple[OpenAIModelAdapter, ToolRegistry, dict[Path, tuple[Project, SessionStore, LspPool]]]:
    from minimal_agent.model import OpenAIModelAdapter
    from minimal_agent.tool.registry import ToolRegistry

    model = getattr(_THREAD_STATE, "model", None)
    tools = getattr(_THREAD_STATE, "tools", None)
    project_cache = getattr(_THREAD_STATE, "project_cache", None)
    if model is None:
        model = OpenAIModelAdapter(cfg.model)
        _THREAD_STATE.model = model
    if tools is None:
        tools = ToolRegistry()
        _THREAD_STATE.tools = tools
    if project_cache is None:
        project_cache = {}
        _THREAD_STATE.project_cache = project_cache
    return model, tools, project_cache


def _print_runtime_event(event: dict[str, object]) -> None:
    kind = str(event.get("type", "event"))
    if kind == "session":
        console.rule(f"Session {event['session_id']}")
        console.print(f"cwd: {event['cwd']}")
        console.print(f"agent: {event.get('agent', 'build')}")
        return
    if kind == "tool_call":
        console.rule(f"Tool Call: {event['name']}")
        console.print(json.dumps(event.get("args", {}), ensure_ascii=False, indent=2))
        return
    if kind == "tool_result":
        console.rule(f"Tool Result: {event['name']}")
        console.print(event.get("output", ""))
        return
    if kind == "final":
        console.rule("Final Draft")
        console.print(event.get("text", ""))
        return
    if kind == "error":
        console.rule("Error")
        console.print(event.get("message", ""))


def _evaluate_row(
    *,
    index: int,
    row: dict[str, Any],
    script_dir: Path,
    repo_root: Path,
    task_template: str,
    cfg: AgentConfig,
    agent: str,
    max_steps: int,
    show_events: bool,
) -> dict[str, Any]:
    from minimal_agent.runtime.loop import run as run_loop
    from minimal_agent.types import RunInput

    repo = row["repo"]
    commit_id = row["commit_id"]
    oracle_label = row["oracle_label"]
    repo_path, resolved_name = _resolve_repo_path(repo, repo_root)

    result_row: dict[str, Any] = {
        "case_index": index,
        "source_index": row["source_index"],
        "source_jsonl": row.get("source_jsonl", ""),
        "repo": repo,
        "resolved_repo_dir": resolved_name,
        "commit_id": commit_id,
        "oracle_label": oracle_label,
        "predicted_label": "",
        "match": False,
        "run_status": "",
        "eval_status": "",
        "session_id": "",
        "parse_error": "",
        "run_error": "",
        "parsed_candidate": None,
        "repair_session_id": "",
    }

    if repo_path is None:
        result_row["run_status"] = "not_run"
        result_row["eval_status"] = "repo_not_found"
        return result_row
    if not _commit_exists(repo_path, commit_id):
        result_row["run_status"] = "not_run"
        result_row["eval_status"] = "commit_not_found"
        return result_row

    task = _build_task(task_template, repo_path, commit_id)
    try:
        model, tools, project_cache = _thread_runtime(script_dir, cfg)
        project, store, lsp = _project_bundle(repo_path, script_dir, cfg, project_cache)
        run_result = run_loop(
            run=RunInput(
                task=task,
                cwd=project.cwd,
                max_steps=max_steps,
                agent=agent,
                title=f"jsonl-eval:{repo}:{commit_id[:12]}",
            ),
            cfg=cfg,
            project=project,
            store=store,
            model=model,
            tools=tools,
            lsp=lsp,
            emit=_print_runtime_event if show_events else None,
        )
        result_row["run_status"] = run_result.status
        result_row["session_id"] = run_result.session_id
        state = store.load(run_result.session_id)
        try:
            parsed = _extract_json_object(state.candidate)
            predicted_label = _normalize_label(parsed.get("primary_label"))
            result_row["parsed_candidate"] = parsed
            result_row["predicted_label"] = predicted_label
            result_row["match"] = predicted_label == oracle_label
            result_row["eval_status"] = "ok" if predicted_label else "invalid_label"
        except Exception as err:
            parsed = _repair_json_candidate(model, state.candidate)
            if parsed is None:
                repair_task = (
                    task
                    + "\n\n[JSON_RETRY_INSTRUCTION]\n"
                    + "Previous attempt failed because the final answer was not valid JSON. "
                    + "Re-run the analysis and output exactly one JSON object only. "
                    + "Keep every field concise. Do not use markdown fences or any extra text."
                )
                repair_result = run_loop(
                    run=RunInput(
                        task=repair_task,
                        cwd=project.cwd,
                        max_steps=min(max_steps, 4),
                        agent=agent,
                        title=f"jsonl-eval-repair:{repo}:{commit_id[:12]}",
                    ),
                    cfg=cfg,
                    project=project,
                    store=store,
                    model=model,
                    tools=tools,
                    lsp=lsp,
                    emit=_print_runtime_event if show_events else None,
                )
                result_row["repair_session_id"] = repair_result.session_id
                repair_state = store.load(repair_result.session_id)
                try:
                    parsed = _extract_json_object(repair_state.candidate)
                    result_row["run_status"] = repair_result.status
                    result_row["session_id"] = repair_result.session_id
                except Exception:
                    parsed = _repair_json_candidate(model, repair_state.candidate)
            if parsed is not None:
                predicted_label = _normalize_label(parsed.get("primary_label"))
                result_row["parsed_candidate"] = parsed
                result_row["predicted_label"] = predicted_label
                result_row["match"] = predicted_label == oracle_label
                result_row["eval_status"] = "ok" if predicted_label else "invalid_label"
            else:
                result_row["parse_error"] = str(err)
                result_row["eval_status"] = "parse_error"
    except Exception as err:
        result_row["run_error"] = str(err)
        result_row["run_status"] = "error"
        result_row["eval_status"] = "run_error"

    return result_row


@app.command()
def main(
    input_jsonl: Path | None = typer.Argument(None),
    task_file: Path = typer.Option(Path("my-task.txt"), "--task-file", exists=True, dir_okay=False, readable=True),
    repo_root: Path = typer.Option(
        Path("/data2/opencode/examples/minimal-agent-cli/src-openeuler"),
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(Path("eval_runs/agent_jsonl"), "--output-dir"),
    agent: str = typer.Option("commit_eval", "--agent"),
    max_steps: int = typer.Option(8, "--max-steps", min=1),
    max_workers: int = typer.Option(1, "--max-workers", min=1),
    limit: int | None = typer.Option(None, "--limit", min=1),
    only_repo: str | None = typer.Option(None, "--only-repo"),
    auto_sample_30: bool = typer.Option(
        False,
        "--auto-sample-30",
        help=(
            "Use the built-in random sample from relative paths: "
            "data/kernel_tagged_commits.jsonl (feat/fix 30 each) and "
            "data/oracle_commits.jsonl (refactor/docs/test 30 each)."
        ),
    ),
    sample_seed: int | None = typer.Option(None, "--sample-seed"),
    show_events: bool = typer.Option(False, "--show-events"),
) -> None:
    from minimal_agent.agent import get as get_agent

    script_dir = Path(__file__).resolve().parent
    get_agent(agent)
    task_template = task_file.read_text(encoding="utf-8").strip()
    if not task_template:
        raise typer.BadParameter(f"task file is empty: {task_file}")
    if auto_sample_30 and input_jsonl is not None:
        raise typer.BadParameter("do not pass input_jsonl when using --auto-sample-30")
    if not auto_sample_30 and input_jsonl is None:
        raise typer.BadParameter("missing input_jsonl, or use --auto-sample-30")
    if input_jsonl is not None:
        input_jsonl = input_jsonl.resolve()
        if not input_jsonl.is_file():
            raise typer.BadParameter(f"input jsonl not found: {input_jsonl}")

    cfg = _load_base_config(script_dir)
    repo_root = repo_root.resolve()
    output_dir = (script_dir / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if auto_sample_30:
        selection_mode = AUTO_SAMPLE_30_NAME
        rows = _load_auto_sample_rows(
            script_dir,
            only_repo=only_repo,
            limit=limit,
            sample_seed=sample_seed,
        )
        input_ref: Path | str = Path(f"{AUTO_SAMPLE_30_NAME}.jsonl")
    else:
        selection_mode = "input_jsonl"
        rows = _load_input_rows(
            input_jsonl,
            only_repo=only_repo,
            limit=limit,
        )
        input_ref = input_jsonl
    jsonl_path, summary_path = _result_paths(output_dir, input_ref)

    if not rows:
        raise typer.BadParameter("no input rows selected")
    if show_events and max_workers > 1:
        raise typer.BadParameter("--show-events does not support --max-workers > 1")

    if auto_sample_30:
        console.print(
            f"Using {AUTO_SAMPLE_30_NAME} from relative paths with seed="
            f"{sample_seed if sample_seed is not None else 'system-random'}"
        )
    console.print(f"Running {len(rows)} cases with agent={agent} max_workers={max_workers}")
    results: list[dict[str, Any]] = []
    if max_workers == 1:
        for index, row in enumerate(rows, start=1):
            result_row = _evaluate_row(
                index=index,
                row=row,
                script_dir=script_dir,
                repo_root=repo_root,
                task_template=task_template,
                cfg=cfg,
                agent=agent,
                max_steps=max_steps,
                show_events=show_events,
            )
            results.append(result_row)
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result_row, ensure_ascii=False))
                fh.write("\n")
            console.print(
                f"[{index}/{len(rows)}] repo={row['repo']} sha={row['commit_id'][:12]} "
                f"oracle={row['oracle_label'] or '-'} predicted={result_row['predicted_label'] or '-'} "
                f"status={result_row['eval_status']}"
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_row,
                    index=index,
                    row=row,
                    script_dir=script_dir,
                    repo_root=repo_root,
                    task_template=task_template,
                    cfg=cfg,
                    agent=agent,
                    max_steps=max_steps,
                    show_events=False,
                ): (index, row)
                for index, row in enumerate(rows, start=1)
            }
            completed = 0
            for future in as_completed(futures):
                index, row = futures[future]
                result_row = future.result()
                results.append(result_row)
                with jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(result_row, ensure_ascii=False))
                    fh.write("\n")
                completed += 1
                console.print(
                    f"[{completed}/{len(rows)}] case={index} repo={row['repo']} sha={row['commit_id'][:12]} "
                    f"oracle={row['oracle_label'] or '-'} predicted={result_row['predicted_label'] or '-'} "
                    f"status={result_row['eval_status']}"
                )

    total = len(results)
    correct = sum(1 for item in results if item["match"])
    runnable = sum(1 for item in results if item["eval_status"] not in {"repo_not_found", "commit_not_found"})
    parsed_ok = sum(1 for item in results if item["eval_status"] == "ok")
    eval_status_counts = Counter(item["eval_status"] for item in results)
    run_status_counts = Counter(item["run_status"] for item in results)

    by_label: dict[str, dict[str, float | int | None]] = {}
    for label in LABELS:
        label_rows = [item for item in results if item["oracle_label"] == label]
        label_total = len(label_rows)
        label_correct = sum(1 for item in label_rows if item["match"])
        by_label[label] = {
            "total": label_total,
            "correct": label_correct,
            "accuracy": (label_correct / label_total) if label_total else None,
        }

    summary = {
        "input_jsonl": str(input_jsonl) if input_jsonl is not None else None,
        "selection_mode": selection_mode,
        "sample_seed": sample_seed if auto_sample_30 else None,
        "sample_sources": dict(AUTO_SAMPLE_30_SPECS) if auto_sample_30 else None,
        "task_file": str(task_file.resolve()),
        "repo_root": str(repo_root),
        "output_jsonl": str(jsonl_path),
        "agent": agent,
        "max_steps": max_steps,
        "max_workers": max_workers,
        "total": total,
        "runnable": runnable,
        "parsed_ok": parsed_ok,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "eval_status_counts": dict(eval_status_counts),
        "run_status_counts": dict(run_status_counts),
        "accuracy_by_label": by_label,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    overview = Table(title="agent jsonl eval")
    overview.add_column("metric")
    overview.add_column("value")
    for key in ("total", "runnable", "parsed_ok", "correct", "accuracy"):
        overview.add_row(key, str(summary[key]))
    overview.add_row("output_jsonl", str(jsonl_path))
    console.print(overview)

    detail = Table(title="accuracy by label")
    detail.add_column("label")
    detail.add_column("total", justify="right")
    detail.add_column("correct", justify="right")
    detail.add_column("accuracy", justify="right")
    for label in LABELS:
        item = by_label[label]
        accuracy = item["accuracy"]
        detail.add_row(
            label,
            str(item["total"]),
            str(item["correct"]),
            "-" if accuracy is None else f"{accuracy:.4f}",
        )
    console.print(detail)
    console.print(f"summary saved to {summary_path}")


if __name__ == "__main__":
    app()
