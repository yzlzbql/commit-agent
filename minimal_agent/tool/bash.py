from __future__ import annotations

from pathlib import Path
import subprocess

from pydantic import BaseModel

from ..policy import ensure_shell
from ..types import ToolCtx, ToolResult, ToolSpec


class BashArgs(BaseModel):
    command: str
    timeout: int = 120
    workdir: str | None = None


def spec() -> ToolSpec:
    return ToolSpec(name="bash", description="Run a shell command in the project", input_model=BashArgs, execute=run)


def run(ctx: ToolCtx, args: BashArgs) -> ToolResult:
    cwd = (ctx.project.cwd if args.workdir is None else (ctx.project.cwd / args.workdir)).resolve()
    ensure_shell(ctx.project.root, cwd, args.command)
    proc = subprocess.Popen(
        args.command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(timeout=args.timeout)
    body = (out + "\n" + err).strip()
    return ToolResult(
        title=args.command,
        output=body,
        metadata={"exit_code": proc.returncode, "cwd": str(cwd)},
    )
