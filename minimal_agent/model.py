from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from typing import Any, Iterable

from openai import NOT_GIVEN, OpenAI

from .config import ModelCfg


class ModelAdapter(ABC):
    @abstractmethod
    def stream(self, msgs: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        raise NotImplementedError


class OpenAIModelAdapter(ModelAdapter):
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        self.model_name = _resolve_optional(cfg.name_env) or cfg.name
        api_key = _resolve_required(cfg.api_key_env)
        self.client = OpenAI(
            api_key=api_key,
            organization=_resolve_optional(cfg.organization_env),
            project=_resolve_optional(cfg.project_env),
            base_url=_resolve_optional(cfg.base_url_env),
            timeout=cfg.timeout_seconds,
        )

    def stream(self, msgs: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=msgs,
            tools=[_tool_schema(item) for item in tools],
            tool_choice="auto",
            parallel_tool_calls=False,
            reasoning_effort=self.cfg.reasoning_effort or NOT_GIVEN,
            verbosity=self.cfg.verbosity or NOT_GIVEN,
            temperature=self.cfg.temperature if self.cfg.temperature is not None else NOT_GIVEN,
            max_completion_tokens=(
                self.cfg.max_completion_tokens if self.cfg.max_completion_tokens is not None else NOT_GIVEN
            ),
        )
        if not res.choices:
            raise RuntimeError("model returned no choices")
        msg = res.choices[0].message
        text = (msg.content or "").strip()
        if msg.tool_calls:
            if text:
                yield {"type": "reasoning", "text": text}
            call = msg.tool_calls[0]
            raw = call.function.arguments or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError as err:
                raise RuntimeError(f"tool arguments were not valid JSON: {err}") from err
            if not isinstance(args, dict):
                raise RuntimeError("tool arguments must decode to a JSON object")
            yield {"type": "tool_call", "name": call.function.name, "args": args}
            return
        if text:
            yield {"type": "final", "text": text}
            return
        raise RuntimeError("model returned neither tool calls nor final text")


def _resolve_optional(name: str | None) -> str | None:
    if not name:
        return None
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_required(name: str) -> str:
    value = _resolve_optional(name)
    if value:
        return value
    raise RuntimeError(f"missing required environment variable: {name}")


def _tool_schema(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": item["name"],
            "description": item["description"],
            "parameters": item["schema"],
        },
    }
