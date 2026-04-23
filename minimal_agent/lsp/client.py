from __future__ import annotations

from itertools import count
import json
import selectors
import subprocess

from lsprotocol import types


class LspClient:
    def __init__(self, proc: subprocess.Popen[bytes]):
        self.proc = proc
        self.ids = count(1)
        self.initialized = False
        self.diagnostics_by_uri: dict[str, list[dict]] = {}

    def initialize(self, root: str) -> dict:
        params = types.InitializeParams(process_id=None, root_uri=PathUri(root), capabilities={})
        result = self.request("initialize", params.model_dump(mode="json"))
        self.notify("initialized", {})
        self.initialized = True
        return result

    def open(self, uri: str, text: str, lang: str = "") -> None:
        params = types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(uri=uri, language_id=lang, version=1, text=text)
        )
        self.notify("textDocument/didOpen", params.model_dump(mode="json"))

    def change(self, uri: str, text: str, version: int = 2) -> None:
        params = types.DidChangeTextDocumentParams(
            text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=version),
            content_changes=[types.TextDocumentContentChangeEvent(text=text)],
        )
        self.notify("textDocument/didChange", params.model_dump(mode="json"))

    def definition(self, uri: str, line: int, character: int) -> dict:
        return self.request("textDocument/definition", _pos(uri, line, character))

    def references(self, uri: str, line: int, character: int) -> dict:
        params = _pos(uri, line, character)
        params["context"] = {"includeDeclaration": True}
        return self.request("textDocument/references", params)

    def hover(self, uri: str, line: int, character: int) -> dict:
        return self.request("textDocument/hover", _pos(uri, line, character))

    def document_symbol(self, uri: str) -> dict:
        params = types.DocumentSymbolParams(text_document=types.TextDocumentIdentifier(uri=uri))
        return self.request("textDocument/documentSymbol", params.model_dump(mode="json"))

    def workspace_symbol(self, query: str) -> dict:
        params = types.WorkspaceSymbolParams(query=query)
        return self.request("workspace/symbol", params.model_dump(mode="json"))

    def implementation(self, uri: str, line: int, character: int) -> dict:
        return self.request("textDocument/implementation", _pos(uri, line, character))

    def diagnostics(self, uri: str) -> list[dict]:
        return list(self.diagnostics_by_uri.get(uri, []))

    def request(self, method: str, params: dict) -> dict:
        req_id = next(self.ids)
        data = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        self._send(data)
        while True:
            msg = self._read_message()
            if msg is None:
                return {}
            if msg.get("id") == req_id:
                if "error" in msg:
                    return {"error": msg["error"]}
                result = msg.get("result")
                if isinstance(result, dict):
                    return result
                if isinstance(result, list):
                    return {"items": result}
                return {}
            self._handle_message(msg)

    def notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, data: dict) -> None:
        if self.proc.stdin is None:
            return
        raw = json.dumps(data).encode("utf-8")
        msg = f"Content-Length: {len(raw)}\r\n\r\n".encode("utf-8") + raw
        self.proc.stdin.write(msg)
        self.proc.stdin.flush()

    def _read_message(self, timeout: float = 5.0) -> dict | None:
        if self.proc.stdout is None:
            return None
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            ready = sel.select(timeout)
        finally:
            sel.close()
        if not ready:
            return None
        headers: dict[str, str] = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            if line in {b"\r\n", b"\n"}:
                break
            head = line.decode("utf-8", errors="replace").strip()
            if ":" not in head:
                continue
            key, value = head.split(":", 1)
            headers[key.lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length <= 0:
            return None
        payload = self.proc.stdout.read(length)
        if not payload:
            return None
        return json.loads(payload.decode("utf-8", errors="replace"))

    def _handle_message(self, msg: dict) -> None:
        if msg.get("method") != "textDocument/publishDiagnostics":
            return
        params = msg.get("params", {})
        uri = params.get("uri")
        diagnostics = params.get("diagnostics", [])
        if isinstance(uri, str) and isinstance(diagnostics, list):
            self.diagnostics_by_uri[uri] = diagnostics


def _pos(uri: str, line: int, character: int) -> dict:
    params = types.TextDocumentPositionParams(
        text_document=types.TextDocumentIdentifier(uri=uri),
        position=types.Position(line=line, character=character),
    )
    return params.model_dump(mode="json")


def PathUri(path: str) -> str:
    return f"file://{path}"
