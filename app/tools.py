from __future__ import annotations

import asyncio
import contextlib
import html
import json
import re
import sys
import tempfile
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx


class Tools:
    async def web(
        self,
        query: str,
        *,
        maxres: int = 5,
        allow: list[str] | None = None,
    ) -> list[dict[str, str]]:
        q = (query or "").strip()
        if not q:
            return []

        n = max(1, min(int(maxres or 5), 10))
        allowset = {d.lower().lstrip(".") for d in (allow or []) if isinstance(d, str) and d.strip()}

        url = "https://html.duckduckgo.com/html/"
        timeout = httpx.Timeout(20.0, connect=10.0)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = await client.post(url, data={"q": q})
            resp.raise_for_status()
            page = resp.text

        linkpat = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        snippat = re.compile(
            r'class="result__snippet"[^>]*>(?P<snip>.*?)</',
            re.IGNORECASE | re.DOTALL,
        )
        tagpat = re.compile(r"<[^>]+>")

        out: list[dict[str, str]] = []
        for m in linkpat.finditer(page):
            href = html.unescape(m.group("href"))
            title = tagpat.sub("", html.unescape(m.group("title"))).strip()
            if not title:
                continue

            fixed = _ddgurl(href)
            host = urlparse(fixed).hostname or ""
            host = host.lower().lstrip(".")
            if allowset and host and host not in allowset and not any(host.endswith("." + d) for d in allowset):
                continue

            snippet = ""
            tail = page[m.end() : m.end() + 2500]
            ms = snippat.search(tail)
            if ms:
                snippet = tagpat.sub("", html.unescape(ms.group("snip"))).strip()

            out.append({"title": title, "url": fixed, "snippet": snippet})
            if len(out) >= n:
                break

        return out

    async def py(self, code: str, *, timeout: int = 10) -> dict[str, Any]:
        c = code if isinstance(code, str) else ""
        t = max(1, min(int(timeout or 10), 60))
        cap = 100_000

        tmp = tempfile.TemporaryDirectory(prefix="codex-proxy-ci-")
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-c",
                c,
                cwd=tmp.name,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            )

            try:
                outb, errb = await asyncio.wait_for(proc.communicate(), timeout=t)
            except asyncio.TimeoutError:
                proc.kill()
                outb, errb = await proc.communicate()
                return {
                    "ok": False,
                    "timeout": True,
                    "exit_code": None,
                    "stdout": outb.decode("utf-8", errors="replace")[:cap],
                    "stderr": errb.decode("utf-8", errors="replace")[:cap],
                }

            return {
                "ok": proc.returncode == 0,
                "timeout": False,
                "exit_code": proc.returncode,
                "stdout": outb.decode("utf-8", errors="replace")[:cap],
                "stderr": errb.decode("utf-8", errors="replace")[:cap],
            }
        finally:
            tmp.cleanup()

    def _mcptransport(self, url: str, transport: str | None) -> str:
        t = (transport or "").strip().lower()
        if t in {"sse", "streamable_http", "streamable-http", "http"}:
            return "sse" if t == "sse" else "streamable_http"

        p = urlparse(url)
        path = (p.path or "").lower()
        if path.endswith("/sse") or "/sse/" in path:
            return "sse"
        return "streamable_http"

    @contextlib.asynccontextmanager
    async def _mcp(self, url: str, *, headers: dict[str, str] | None, transport: str | None):
        try:
            from mcp.client.session import ClientSession
            from mcp.client.sse import sse_client
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
        except Exception as exc:
            raise RuntimeError(f"mcp not available: {exc}") from exc

        t = self._mcptransport(url, transport)
        if t == "sse":
            async with sse_client(url, headers=headers) as (read, write):
                async with ClientSession(read, write) as sess:
                    await sess.initialize()
                    yield sess
            return

        async with create_mcp_http_client(headers=headers) as client:
            async with streamable_http_client(url, http_client=client) as (read, write, _getid):
                async with ClientSession(read, write) as sess:
                    await sess.initialize()
                    yield sess

    async def mcptools(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        transport: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        lim = max(1, min(int(limit or 200), 1000))
        out: list[dict[str, Any]] = []

        async with self._mcp(url, headers=headers, transport=transport) as sess:
            cur: str | None = None
            for _ in range(50):
                res = await sess.list_tools(cursor=cur)
                for tool in res.tools:
                    out.append(
                        {
                            "name": tool.name,
                            "description": tool.description or tool.title or "",
                            "input_schema": tool.inputSchema or {},
                        }
                    )
                    if len(out) >= lim:
                        return out
                cur = res.nextCursor
                if not cur:
                    break

        return out

    async def mcpcall(
        self,
        url: str,
        name: str,
        *,
        arguments: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        transport: str | None = None,
    ) -> dict[str, Any]:
        nm = (name or "").strip()
        if not nm:
            raise ValueError("mcp tool name is required")

        async with self._mcp(url, headers=headers, transport=transport) as sess:
            res = await sess.call_tool(nm, arguments=arguments or {})

        if res.structuredContent is not None:
            raw = json.dumps(res.structuredContent, ensure_ascii=True, separators=(",", ":"))
            return {"is_error": bool(res.isError), "output": raw}

        texts: list[str] = []
        other = False
        dumps: list[dict[str, Any]] = []
        for item in res.content:
            try:
                d = item.model_dump()
            except Exception:
                d = {"type": getattr(item, "type", "unknown"), "value": str(item)}
            dumps.append(d)
            if d.get("type") == "text" and isinstance(d.get("text"), str):
                texts.append(d["text"])
            else:
                other = True

        if not other and texts:
            return {"is_error": bool(res.isError), "output": "\n".join(texts).strip()}

        cap = 2000
        for d in dumps:
            if isinstance(d.get("data"), str) and len(d["data"]) > cap:
                d["data"] = d["data"][:cap] + "...(truncated)"

        raw = json.dumps(dumps, ensure_ascii=True, separators=(",", ":"))
        return {"is_error": bool(res.isError), "output": raw}


def _ddgurl(href: str) -> str:
    p = urlparse(href)
    if p.scheme in {"http", "https"}:
        return href
    if p.path.startswith("/l/"):
        qs = parse_qs(p.query or "")
        uddg = qs.get("uddg", [None])[0]
        if isinstance(uddg, str) and uddg:
            return unquote(uddg)
    return href

