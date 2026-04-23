from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from ..config import AgentConfig, LspServerCfg
from ..types import LspRequest, LspResult
from .client import LspClient, PathUri


class LspPool:
    def __init__(self, cfg: AgentConfig, root: Path):
        self.cfg = cfg
        self.root = root
        self.clients: dict[str, LspClient] = {}
        self.touched: list[str] = []

    def available(self, path: str | Path) -> bool:
        return self._cfg(path) is not None

    def refresh(self, path: str | Path) -> None:
        item = Path(path).resolve()
        rel = item.relative_to(self.root).as_posix() if self.root in item.parents or item == self.root else str(item)
        if rel not in self.touched:
            self.touched.append(rel)
        cfg = self._cfg(item)
        if cfg is None:
            return
        cli = self._client(cfg)
        if cli is None or not item.exists():
            return
        cli.open(PathUri(str(item)), item.read_text())

    def call(self, req: LspRequest) -> LspResult:
        path = (self.root / req.file_path).resolve()
        cfg = self._cfg(path)
        if cfg is None and req.operation == "workspace_symbol" and self.cfg.lsp.servers:
            cfg = self.cfg.lsp.servers[0]
        if cfg is None:
            return LspResult(ok=False, error="unavailable")
        cli = self._client(cfg)
        if cli is None:
            return LspResult(ok=False, error="unavailable")
        uri = PathUri(str(path))
        line = max(req.line - 1, 0)
        ch = max(req.character - 1, 0)
        if req.operation == "workspace_symbol":
            data = cli.workspace_symbol(req.query or "")
        elif req.operation == "go_to_definition":
            data = cli.definition(uri, line, ch)
        elif req.operation == "find_references":
            data = cli.references(uri, line, ch)
        elif req.operation == "hover":
            data = cli.hover(uri, line, ch)
        elif req.operation == "document_symbol":
            data = cli.document_symbol(uri)
        else:
            data = cli.implementation(uri, line, ch)
        return LspResult(ok=True, items=[data] if data else [])

    def diagnostics(self, path: str | Path) -> list[dict]:
        item = Path(path).resolve()
        cfg = self._cfg(item)
        if cfg is None:
            return []
        cli = self._client(cfg)
        if cli is None:
            return []
        return cli.diagnostics(PathUri(str(item)))

    def _cfg(self, path: str | Path) -> LspServerCfg | None:
        item = Path(path)
        ext = item.suffix
        for cfg in self.cfg.lsp.servers:
            if ext in cfg.extensions:
                return cfg
        return None

    def _client(self, cfg: LspServerCfg) -> LspClient | None:
        cur = self.clients.get(cfg.name)
        if cur is not None:
            return cur
        if not cfg.command or shutil.which(cfg.command[0]) is None:
            return None
        proc = subprocess.Popen(
            cfg.command,
            cwd=self.root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        cli = LspClient(proc)
        try:
            cli.initialize(str(self.root))
        except Exception:
            return None
        self.clients[cfg.name] = cli
        return cli
