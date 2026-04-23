from __future__ import annotations

from pydantic import BaseModel


class RetryInfo(BaseModel):
    kind: str
    retryable: bool
    next_action: str


def classify(kind: str, err: Exception) -> RetryInfo:
    msg = str(err)
    if kind == "verify":
        return RetryInfo(kind="verify_failed", retryable=True, next_action="Fix the verification failure")
    if kind == "tool":
        if msg.startswith("E_"):
            code = msg.split(":", 1)[0]
            retryable = code in {"E_ANALYSIS_TIMEOUT", "E_INTERNAL"}
            next_action = "Adjust tool usage" if not retryable else "Retry the analysis with corrected scope or a larger timeout"
            return RetryInfo(kind=code.lower(), retryable=retryable, next_action=next_action)
        retryable = not isinstance(err, ValueError)
        next = "Adjust tool usage" if not retryable else "Retry with a corrected tool call"
        return RetryInfo(kind="tool_input_error" if not retryable else "tool_runtime_error", retryable=retryable, next_action=next)
    return RetryInfo(kind="model_error", retryable=False, next_action=msg or "Inspect the model failure")
