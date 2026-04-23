from __future__ import annotations

from ..types import SessionState, VerifyResult


def set_phase(st: SessionState, phase: str) -> SessionState:
    st.phase = phase  # type: ignore[assignment]
    return st


def bump_step(st: SessionState) -> SessionState:
    st.step_count += 1
    return st


def record_error(st: SessionState, msg: str) -> SessionState:
    st.last_error = msg
    return st


def record_candidate(st: SessionState, text: str) -> SessionState:
    st.candidate = text
    st.last_error = ""
    return st


def record_verification(st: SessionState, vr: VerifyResult) -> SessionState:
    st.verification = vr
    return st
