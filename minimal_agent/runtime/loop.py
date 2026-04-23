from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any, Callable

from ..agent import get as get_agent, list_agents
from ..analysis import AnalysisService
from ..config import AgentConfig
from ..model import ModelAdapter
from ..policy import ensure_repeat, ensure_retry, ensure_steps
from ..prompt import build_messages
from ..project import Project
from ..session.store import SessionStore
from ..session.summary import build as build_summary
from ..tool.registry import ToolRegistry
from ..types import RunInput, RunResult, SessionState, StepRecord, TaskRequest, ToolCall, ToolCtx, ToolResult, VerifyResult
from .doom import check as check_doom_loop
from .retry import classify
from .state import bump_step, record_candidate, record_error, record_verification, set_phase
from .todo import on_final, on_tool, on_verify, seed
from .verify import run as run_verify


def run(
    run: RunInput,
    cfg: AgentConfig,
    project: Project,
    store: SessionStore,
    model: ModelAdapter,
    tools: ToolRegistry,
    lsp: object,
    analysis: AnalysisService | None = None,
    st: SessionState | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    analysis = analysis or AnalysisService(cfg, project, lsp)
    st = st or store.create(run)
    st.agent = st.agent or run.agent
    st.title = st.title or run.title or run.task[:80]
    st.parent_id = st.parent_id or run.parent_id
    agent = get_agent(st.agent)
    allowed = None if agent.tools is None else set(agent.tools)
    allowed_tools = {item["name"] for item in tools.schemas(allowed)}
    subagents = [
        {"name": item.name, "description": item.description}
        for item in list_agents(mode="subagent")
    ]
    st.todo = store.load_todo(st.id) or st.todo
    st.summary = store.load_summary(st.id)
    _emit(
        emit,
        {
            "type": "session",
            "session_id": st.id,
            "task": run.task,
            "cwd": str(project.cwd),
            "agent": st.agent,
            "parent_id": st.parent_id,
            "title": st.title,
        },
    )

    if st.phase == "analyze":
        if not st.todo:
            st.todo = seed(run.task)
            store.save_todo(st.id, st.todo)
            _emit(
                emit,
                {
                    "type": "todo",
                    "items": [{"content": item.content, "status": item.status, "priority": item.priority} for item in st.todo],
                },
            )
        store.append_event(
            st.id,
            StepRecord(kind="analyze", ts=time.time(), payload={"task": run.task, "agent": st.agent}),
        )
        set_phase(st, "act")
        store.save_state(st)

    calls = _calls(store.events(st.id))

    while st.phase in {"act", "verify"}:
        if st.phase == "act":
            try:
                ensure_steps(run, st)
            except RuntimeError as err:
                next_action = str(err)
                if str(err) == "step budget exceeded":
                    next_action = _step_budget_next_action(run, cfg, st)
                record_error(st, next_action)
                store.append_event(
                    st.id,
                    StepRecord(kind="error", ts=time.time(), payload={"kind": "policy", "message": str(err)}),
                )
                set_phase(st, "stop")
                store.save_state(st)
                _emit(emit, {"type": "error", "kind": "policy", "message": str(err), "next_action": next_action})
                break
            recs = store.events(st.id)
            force_plain_text = (
                _consecutive_guard_count(recs) >= 2
                or _todos_completed(st.todo)
                or _read_only_session_should_finish(store, st)
            )
            tool_schemas = [] if force_plain_text else tools.schemas(allowed_tools)
            msgs = build_messages(
                run,
                st,
                recs,
                tool_schemas,
                bool(cfg.lsp.servers),
                agent=agent,
                available_agents=subagents,
                force_plain_text=force_plain_text,
            )
            try:
                out = list(model.stream(msgs, tool_schemas))
            except Exception as err:
                info = classify("model", err)
                msg = str(err)
                retryable_model_error = (
                    "tool arguments were not valid JSON" in msg
                    or "maximum context length" in msg
                )
                if retryable_model_error:
                    st.retry_count += 1
                    record_error(st, msg)
                    store.append_event(
                        st.id,
                        StepRecord(
                            kind="error",
                            ts=time.time(),
                            payload={"kind": info.kind, "message": msg, "next_action": info.next_action},
                        ),
                    )
                    try:
                        ensure_retry(cfg.limits.max_retries, st)
                    except RuntimeError:
                        set_phase(st, "stop")
                    store.save_state(st)
                    _emit(emit, {"type": "error", "kind": info.kind, "message": msg, "next_action": info.next_action})
                    continue
                raise
            if not out:
                err = RuntimeError("model returned no events")
                info = classify("model", err)
                record_error(st, str(err))
                set_phase(st, "stop")
                _emit(emit, {"type": "error", "kind": info.kind, "message": str(err), "next_action": info.next_action})
                return _result(run.task, cfg, store, st, info.next_action)
            moved = False
            for item in out:
                kind = item.get("type")
                if kind == "reasoning":
                    _emit(emit, {"type": "reasoning", "text": item.get("text", "")})
                    store.append_event(
                        st.id,
                        StepRecord(kind="reasoning", ts=time.time(), payload={"text": item.get("text", "")}),
                    )
                    continue
                if kind == "tool_call":
                    call = ToolCall(name=item["name"], args=item.get("args", {}))
                    _emit(emit, {"type": "tool_call", "name": call.name, "args": call.args})
                    if call.name not in allowed_tools:
                        err = RuntimeError(f"tool not available for agent {st.agent}: {call.name}")
                        record_error(st, str(err))
                        store.append_event(
                            st.id,
                            StepRecord(kind="error", ts=time.time(), payload={"kind": "policy", "message": str(err)}),
                        )
                        set_phase(st, "stop")
                        store.save_state(st)
                        _emit(emit, {"type": "error", "kind": "policy", "message": str(err)})
                        moved = True
                        break
                    try:
                        ensure_repeat(cfg.limits.max_repeat_tool_calls, calls, call)
                    except RuntimeError as err:
                        record_error(st, str(err))
                        store.append_event(
                            st.id,
                            StepRecord(kind="error", ts=time.time(), payload={"kind": "policy", "message": str(err)}),
                        )
                        set_phase(st, "stop")
                        store.save_state(st)
                        _emit(emit, {"type": "error", "kind": "policy", "message": str(err)})
                        moved = True
                        break
                    bump_step(st)
                    res: ToolResult
                    guard = check_doom_loop(recs, call, cfg.limits.doom_loop_threshold)
                    if guard:
                        res = ToolResult(
                            title=f"{call.name} blocked by loop guard",
                            output=guard,
                            metadata={"guard": True},
                        )
                    else:
                        ctx = ToolCtx(
                            run=run,
                            st=st,
                            project=project,
                            cfg=cfg,
                            store=store,
                            lsp=lsp,
                            analysis=analysis,
                            emit=emit,
                            run_subtask=lambda req: _run_subtask(
                                req,
                                parent_state=st,
                                cfg=cfg,
                                project=project,
                                store=store,
                                model=model,
                                tools=tools,
                                lsp=lsp,
                                analysis=analysis,
                                emit=emit,
                            ),
                            available_agents=subagents,
                        )
                        try:
                            res = tools.execute(ctx, call)
                        except Exception as err:
                            info = classify("tool", err)
                            st.retry_count += 1
                            record_error(st, str(err))
                            store.append_event(
                                st.id,
                                StepRecord(
                                    kind="error",
                                    ts=time.time(),
                                    payload={"kind": info.kind, "message": str(err), "next_action": info.next_action},
                                ),
                            )
                            try:
                                ensure_retry(cfg.limits.max_retries, st)
                            except RuntimeError:
                                set_phase(st, "stop")
                            store.save_state(st)
                            _emit(
                                emit,
                                {"type": "error", "kind": info.kind, "message": str(err), "next_action": info.next_action},
                            )
                            moved = True
                            break
                    calls.append(call)
                    st.todo = on_tool(st.todo, res)
                    store.save_todo(st.id, st.todo)
                    _emit(
                        emit,
                        {
                            "type": "tool_result",
                            "name": call.name,
                            "title": res.title,
                            "output": res.output,
                            "metadata": res.metadata,
                        },
                    )
                    store.append_event(
                        st.id,
                        StepRecord(
                            kind="tool",
                            ts=time.time(),
                            payload={
                                "name": call.name,
                                "args": call.args,
                                "title": res.title,
                                "output": res.output,
                                "files": res.metadata.get("files", []),
                                "metadata": res.metadata,
                            },
                        ),
                    )
                    store.save_state(st)
                    moved = True
                    break
                if kind == "final":
                    bump_step(st)
                    record_candidate(st, item.get("text", ""))
                    _emit(emit, {"type": "final", "text": st.candidate})
                    st.todo = on_final(st.todo)
                    store.save_todo(st.id, st.todo)
                    store.append_event(
                        st.id,
                        StepRecord(kind="final", ts=time.time(), payload={"text": st.candidate}),
                    )
                    if cfg.verify.commands and st.agent == "build":
                        set_phase(st, "verify")
                    else:
                        set_phase(st, "finish")
                    store.save_state(st)
                    moved = True
                    break
            if moved:
                continue
            err = RuntimeError("model did not produce tool_call or final")
            record_error(st, str(err))
            set_phase(st, "stop")
            store.save_state(st)
            _emit(emit, {"type": "error", "kind": "model", "message": str(err)})
            break

        if st.phase == "verify":
            bump_step(st)
            _emit(emit, {"type": "verify_start"})
            vr = run_verify(project, cfg, store, st)
            record_verification(st, vr)
            _emit(
                emit,
                {
                    "type": "verify_result",
                    "ok": vr.ok,
                    "checks": vr.checks,
                    "failures": vr.failures,
                    "next_action": vr.next_action,
                },
            )
            st.todo = on_verify(st.todo, vr.ok)
            store.save_todo(st.id, st.todo)
            store.append_event(
                st.id,
                StepRecord(kind="verify", ts=time.time(), payload=vr.model_dump()),
            )
            if vr.ok:
                set_phase(st, "finish")
            else:
                st.retry_count += 1
                info = classify("verify", RuntimeError("; ".join(vr.failures)))
                cap = _step_cap(run, cfg)
                if st.retry_count <= cfg.limits.max_retries and (cap is None or st.step_count < cap):
                    set_phase(st, "act")
                    record_error(st, info.next_action)
                else:
                    set_phase(st, "stop")
            store.save_state(st)

    next_action = st.last_error or (st.verification.next_action if st.verification else "")
    result = _result(run.task, cfg, store, st, next_action)
    _emit(
        emit,
        {
            "type": "result",
            "status": result.status,
            "session_id": result.session_id,
            "files": result.files,
            "next_action": result.next_action,
            "verification_ok": result.verification.ok,
            "summary": result.summary,
            "agent": result.agent,
        },
    )
    return result


def _run_subtask(
    req: TaskRequest,
    *,
    parent_state: SessionState,
    cfg: AgentConfig,
    project: Project,
    store: SessionStore,
    model: ModelAdapter,
    tools: ToolRegistry,
    lsp: object,
    analysis: AnalysisService | None,
    emit: Callable[[dict[str, Any]], None] | None,
) -> RunResult:
    agent = get_agent(req.subagent_type)
    child_state = store.load(req.task_id) if req.task_id else None
    child_input = RunInput(
        task=req.prompt,
        cwd=project.cwd,
        resume=req.task_id,
        max_steps=req.max_steps if req.max_steps is not None else (agent.steps if agent.steps is not None else cfg.limits.max_steps),
        agent=req.subagent_type,
        parent_id=parent_state.id,
        title=req.description,
    )
    _emit(
        emit,
        {
            "type": "subtask_start",
            "description": req.description,
            "agent": req.subagent_type,
            "task_id": req.task_id or "",
        },
    )
    result = run(
        run=child_input,
        cfg=cfg,
        project=project,
        store=store,
        model=model,
        tools=tools,
        lsp=lsp,
        analysis=analysis,
        st=child_state,
        emit=_subtask_emit(emit, req.description, req.subagent_type),
    )
    _emit(
        emit,
        {
            "type": "subtask_finish",
            "description": req.description,
            "agent": req.subagent_type,
            "task_id": result.session_id,
            "status": result.status,
        },
    )
    return result


def _subtask_emit(
    emit: Callable[[dict[str, Any]], None] | None,
    description: str,
    agent: str,
) -> Callable[[dict[str, Any]], None] | None:
    if emit is None:
        return None

    def inner(event: dict[str, Any]) -> None:
        tagged = dict(event)
        tagged["subtask"] = True
        tagged["subtask_description"] = description
        tagged["subtask_agent"] = agent
        emit(tagged)

    return inner


def _calls(recs: list[StepRecord]) -> list[ToolCall]:
    result = []
    for rec in recs:
        if rec.kind != "tool":
            continue
        result.append(ToolCall(name=rec.payload.get("name", ""), args=rec.payload.get("args", {})))
    return result


def _consecutive_guard_count(recs: list[StepRecord]) -> int:
    count = 0
    for rec in reversed(recs):
        if rec.kind != "tool":
            break
        if not rec.payload.get("metadata", {}).get("guard"):
            break
        count += 1
    return count


def _todos_completed(items: list[Any]) -> bool:
    return bool(items) and all(getattr(item, "status", None) == "completed" for item in items)


def _read_only_session_should_finish(store: SessionStore, st: SessionState) -> bool:
    if st.agent == "commit_eval":
        return st.step_count >= 5
    return st.step_count >= 8 and not store.files(st.id)


def _step_cap(run: RunInput, cfg: AgentConfig) -> int | None:
    return run.max_steps if run.max_steps is not None else cfg.limits.max_steps


def _step_budget_next_action(run: RunInput, cfg: AgentConfig, st: SessionState) -> str:
    cap = _step_cap(run, cfg) or 1
    suggested = max(cap * 2, cap + 8)
    return f"Resume with --resume {st.id} --max-steps {suggested}, or return final from the current context."


def _result(task: str, cfg: AgentConfig, store: SessionStore, st: SessionState, next_action: str) -> RunResult:
    files = store.files(st.id)
    summary = build_summary(task, store.events(st.id), files, next_action)
    store.save_summary(st.id, summary)
    st.summary = summary
    store.save_state(st)
    vr = st.verification or _verification_result(cfg, store, st, next_action)
    return RunResult(
        status="finished" if st.phase == "finish" else "stopped",
        summary=summary,
        files=files,
        verification=vr,
        session_id=st.id,
        next_action=next_action or vr.next_action,
        agent=st.agent,
    )


def _emit(emit: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if emit is not None:
        emit(event)


def _verification_result(cfg: AgentConfig, store: SessionStore, st: SessionState, next_action: str) -> VerifyResult:
    if cfg.verify.commands and st.agent == "build":
        return run_verify(Project.load(st.cwd), cfg, store, st)
    if st.phase == "finish":
        msg = "verification skipped (no verify.commands configured)"
        if st.agent != "build":
            msg = f"verification skipped for agent {st.agent}"
        return VerifyResult(
            ok=True,
            checks=[msg],
            next_action="Complete",
        )
    return VerifyResult(
        ok=False,
        failures=["session stopped before completion"],
        next_action=next_action or "Continue acting",
    )
