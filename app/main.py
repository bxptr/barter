from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import httpx
import jsonschema
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, field_validator

from app.codex import Call, Codex, CodexError, Result, Update, Usage


MODELS = [
    "gpt-5.3-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5-codex",
    "codex-mini-latest",
]

EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]

SYSTEM = "You are a general-purpose, helpful AI assistant"


@dataclass(frozen=True)
class Cfg:
    bin: str
    cwd: Path
    timeout: int

    @classmethod
    def env(cls) -> "Cfg":
        bin = os.getenv("CODEX_BINARY", "codex").strip()
        cwd = Path(os.getenv("CODEX_WORKDIR", os.getcwd())).expanduser().resolve()
        timeout = int(os.getenv("CODEX_TIMEOUT_SECONDS", "900"))
        return cls(bin=bin, cwd=cwd, timeout=timeout)


class ImgErr(ValueError):
    pass


DATAURL = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE)
MAXIMG = 20 * 1024 * 1024


class Tmpimgs:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="codex-proxy-img-")
        self._cache: dict[str, Path] = {}

    @property
    def dir(self) -> Path:
        return Path(self._tmp.name)

    def close(self) -> None:
        self._tmp.cleanup()

    async def get(self, url: str) -> Path:
        cached = self._cache.get(url)
        if cached is not None:
            return cached

        p = urlparse(url)

        if url.startswith("data:"):
            path = self._write(url)
            self._cache[url] = path
            return path

        if p.scheme in {"http", "https"}:
            path = await self._dl(url)
            self._cache[url] = path
            return path

        if p.scheme == "file":
            path = Path(unquote(p.path)).expanduser().resolve()
            if not path.exists():
                raise ImgErr(f"missing local file: {path}")
            self._cache[url] = path
            return path

        path = Path(url).expanduser().resolve()
        if path.exists():
            self._cache[url] = path
            return path

        raise ImgErr("unsupported image url")

    def _write(self, url: str) -> Path:
        m = DATAURL.match(url)
        if not m:
            raise ImgErr("invalid data url")

        mime = m.group("mime")
        suf = mimetypes.guess_extension(mime) or ".img"

        try:
            raw = base64.b64decode(m.group("data"), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImgErr("invalid base64") from exc

        if len(raw) > MAXIMG:
            raise ImgErr("image too large")

        path = self.dir / f"img-{uuid4().hex}{suf}"
        path.write_bytes(raw)
        return path

    async def _dl(self, url: str) -> Path:
        timeout = httpx.Timeout(20.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        raw = resp.content
        if len(raw) > MAXIMG:
            raise ImgErr("image too large")

        ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        suf = mimetypes.guess_extension(ct) or Path(urlparse(url).path).suffix or ".img"

        path = self.dir / f"img-{uuid4().hex}{suf}"
        path.write_bytes(raw)
        return path


class ImageURL(BaseModel):
    url: str
    detail: str | None = None

    model_config = ConfigDict(extra="allow")


class Part(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None

    model_config = ConfigDict(extra="allow")


class Msg(BaseModel):
    role: str
    content: str | list[Part] | None = None
    name: str | None = None

    model_config = ConfigDict(extra="allow")


class Streamopts(BaseModel):
    include_usage: bool = False

    model_config = ConfigDict(extra="allow")


class Chatreq(BaseModel):
    model: str
    messages: list[Msg]
    stream: bool = False
    reasoning_effort: str | None = None
    stream_options: Streamopts | None = None
    response_format: dict[str, Any] | None = None

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("messages")
    @classmethod
    def nonempty(cls, v: list[Msg]) -> list[Msg]:
        if not v:
            raise ValueError("messages must be non-empty")
        return v


class Reason(BaseModel):
    effort: str | None = None

    model_config = ConfigDict(extra="allow")


class Respreq(BaseModel):
    model: str
    input: str | list[dict[str, Any]] | None = None
    instructions: str | None = None
    stream: bool = False
    store: bool = True
    reasoning: Reason | None = None
    previous_response_id: str | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, Any] | None = None
    background: bool | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    text: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None

    temperature: float | None = None
    top_p: float | None = None

    model_config = ConfigDict(extra="allow")


def errpayload(
    msg: str,
    *,
    type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {"error": {"message": msg, "type": type, "param": param, "code": code}}


def err(
    status: int,
    msg: str,
    *,
    type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(status_code=status, content=errpayload(msg, type=type, param=param, code=code))


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


def _canon(typ: str) -> str | None:
    t = (typ or "").strip()
    if not t:
        return None
    if t == "code_interpreter":
        return "code_interpreter"
    if t in {"web_search", "web_search_preview"} or t.startswith("web_search_"):
        return "web_search"
    return None


def toolset(raw: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not raw:
        return [], {}

    if not isinstance(raw, list):
        raise ValueError("tools must be a list")

    defs: list[dict[str, Any]] = []
    enabled: dict[str, dict[str, Any]] = {}

    for t in raw:
        if not isinstance(t, dict):
            raise ValueError("each tool must be an object")

        typ = t.get("type")
        if not isinstance(typ, str) or not typ.strip():
            raise ValueError("each tool needs a string 'type'")

        name = _canon(typ)
        if name is None:
            raise ValueError(f"unsupported tool type '{typ}'")

        defs.append(t)

        if name == "web_search":
            allow = None
            filt = t.get("filters")
            if isinstance(filt, dict) and isinstance(filt.get("allowed_domains"), list):
                allow = [d for d in filt.get("allowed_domains") if isinstance(d, str)]
            enabled[name] = {"allow": allow}
        else:
            enabled[name] = {}

    return defs, enabled


def toolchoice(raw: Any | None, enabled: dict[str, dict[str, Any]]) -> Any:
    if raw is None:
        return "auto" if enabled else "none"

    if isinstance(raw, str):
        s = raw.strip()
        if s in {"none", "auto", "required"}:
            return s
        name = _canon(s)
        if name and name in enabled:
            return {"type": name}
        raise ValueError("tool_choice must be 'none', 'auto', 'required', or an object with type")

    if isinstance(raw, dict):
        typ = raw.get("type")
        if isinstance(typ, str):
            name = _canon(typ)
            if name and name in enabled:
                return {"type": name}
        raise ValueError("tool_choice object must specify an enabled tool type")

    raise ValueError("tool_choice must be a string or object")


def actjson(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    if not t:
        return None

    m = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        t = m.group(1).strip()

    try:
        val = json.loads(t)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass

    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        frag = t[i : j + 1]
        try:
            val = json.loads(frag)
            return val if isinstance(val, dict) else None
        except json.JSONDecodeError:
            return None

    return None


def actlead(enabled: dict[str, dict[str, Any]], choice: Any) -> str:
    names = ", ".join(sorted(enabled))
    forced = ""
    if isinstance(choice, dict) and isinstance(choice.get("type"), str):
        forced = choice["type"]
    if choice == "required":
        forced = names or ""

    must = f" You must call {forced}." if forced else ""
    return (
        "TOOL MODE. Decide the next step for the assistant.\n"
        f"Allowed tools: {names or 'none'}.{must}\n"
        "Output exactly one JSON object and nothing else.\n"
        "To call a tool:\n"
        '  {"type":"tool","name":"web_search","arguments":{"query":"...","max_results":5}}\n'
        '  {"type":"tool","name":"code_interpreter","arguments":{"code":"..."}}\n'
        "To finish:\n"
        '  {"type":"final","text":"..."}'
    )


def actparse(text: str) -> tuple[str, str | None, dict[str, Any] | None, str | None]:
    act = actjson(text)
    if act is None:
        return "final", None, None, text

    typ = str(act.get("type") or "").strip().lower()
    if typ == "tool":
        name = act.get("name") or act.get("tool")
        name = _canon(str(name or "").strip()) or str(name or "").strip()
        args: dict[str, Any] | None = None
        if isinstance(act.get("arguments"), dict):
            args = act["arguments"]
        elif isinstance(act.get("args"), dict):
            args = act["args"]
        return "tool", name or None, args or {}, None

    if typ == "final":
        t = act.get("text")
        if not isinstance(t, str):
            t = str(t or "")
        return "final", None, None, t

    return "final", None, None, text


def useadd(a: Usage | None, b: Usage | None) -> Usage | None:
    if a is None:
        return b
    if b is None:
        return a

    def add(x: int | None, y: int | None) -> int | None:
        if x is None and y is None:
            return None
        return (x or 0) + (y or 0)

    return Usage(prompt=add(a.prompt, b.prompt), completion=add(a.completion, b.completion), total=add(a.total, b.total))


def toolmsg(name: str, payload: dict[str, Any]) -> Msg:
    raw = json.dumps(payload, ensure_ascii=True)
    if len(raw) > 20_000:
        raw = raw[:20_000] + "...(truncated)"
    return Msg(role="tool", content=f"[{name}] {raw}")


async def dotool(
    t: Tools,
    name: str,
    args: dict[str, Any],
    opts: dict[str, dict[str, Any]],
    tid: str | None = None,
) -> tuple[dict[str, Any], Msg]:
    if name == "web_search":
        tid = tid or f"ws_{uuid4().hex}"
        q = args.get("query") or args.get("q") or ""
        q = q if isinstance(q, str) else str(q)
        maxres = args.get("max_results") or args.get("maxres") or 5
        try:
            results = await t.web(q, maxres=int(maxres), allow=opts.get("web_search", {}).get("allow"))
            item = {
                "id": tid,
                "type": "web_search_call",
                "status": "completed",
                "query": q,
                "results": results,
            }
            msg = toolmsg("web_search", {"query": q, "results": results})
            return item, msg
        except Exception as exc:
            item = {
                "id": tid,
                "type": "web_search_call",
                "status": "failed",
                "query": q,
                "error": {"message": str(exc)},
                "results": [],
            }
            msg = toolmsg("web_search", {"query": q, "error": str(exc), "results": []})
            return item, msg

    if name == "code_interpreter":
        tid = tid or f"ci_{uuid4().hex}"
        code = args.get("code") or args.get("python") or ""
        code = code if isinstance(code, str) else str(code)
        timeout = args.get("timeout_seconds") or args.get("timeout") or 10
        run = await t.py(code, timeout=int(timeout))
        outputs: list[dict[str, str]] = []
        if run.get("stdout"):
            outputs.append({"type": "logs", "logs": str(run.get("stdout") or "")})
        if run.get("stderr"):
            outputs.append({"type": "logs", "logs": str(run.get("stderr") or "")})
        item = {
            "id": tid,
            "type": "code_interpreter_call",
            "status": "completed" if run.get("ok") else "failed",
            "code": code,
            "container_id": "local",
            "outputs": outputs,
        }
        msg = toolmsg("code_interpreter", {"code": code, **run})
        return item, msg

    raise ValueError(f"unsupported tool '{name}'")


BASELEAD = "Use the conversation below and respond to the latest user request."


def fmtcanon(raw: Any | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("format must be an object")

    typ = raw.get("type")
    if typ is None:
        return None
    if not isinstance(typ, str):
        raise ValueError("format.type must be a string")

    t = typ.strip().lower()
    if not t or t == "text":
        return None
    if t == "json_object":
        return {"type": "json_object"}
    if t == "json_schema":
        src = raw.get("json_schema") if isinstance(raw.get("json_schema"), dict) else raw
        schema = src.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("json_schema.schema must be an object")

        name = src.get("name")
        name = name.strip() if isinstance(name, str) and name.strip() else "response"
        strict = bool(src.get("strict")) if src.get("strict") is not None else False
        desc = src.get("description")
        desc = desc.strip() if isinstance(desc, str) and desc.strip() else None
        return {"type": "json_schema", "name": name, "schema": schema, "strict": strict, "description": desc}

    raise ValueError(f"unsupported format type '{typ}'")


def fmtlead(fmt: dict[str, Any] | None, lead: str = BASELEAD) -> str:
    if not fmt:
        return lead

    if fmt.get("type") == "json_object":
        return (
            f"{lead}\n\n"
            "Return a valid JSON object and nothing else. "
            "Do not use Markdown, code fences, or trailing commentary."
        )

    if fmt.get("type") == "json_schema":
        sch = json.dumps(fmt["schema"], ensure_ascii=True, separators=(",", ":"))
        strict = "true" if fmt.get("strict") else "false"
        return (
            f"{lead}\n\n"
            "Return JSON that matches the JSON Schema exactly. Output only JSON, no other text.\n"
            f"Schema name: {fmt.get('name')}\n"
            f"Strict: {strict}\n"
            f"Schema: {sch}"
        )

    return lead


def jval(text: str) -> Any | None:
    s = (text or "").strip()
    if not s:
        return None

    m = re.search(r"```(?:json)?\\s*(.*?)\\s*```", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            val, _end = dec.raw_decode(s[i:])
            return val
        except json.JSONDecodeError:
            continue

    return None


def fmtout(fmt: dict[str, Any], text: str) -> tuple[str | None, str | None]:
    val = jval(text)
    if val is None:
        return None, "not valid JSON"

    if fmt.get("type") == "json_object":
        if not isinstance(val, dict):
            return None, "expected a JSON object"
        out = json.dumps(val, ensure_ascii=True, separators=(",", ":"))
        return out, None

    if fmt.get("type") == "json_schema":
        try:
            jsonschema.validate(instance=val, schema=fmt["schema"])
        except jsonschema.ValidationError as exc:
            return None, f"schema mismatch: {exc.message}"
        except Exception as exc:
            return None, f"schema error: {exc}"
        out = json.dumps(val, ensure_ascii=True, separators=(",", ":"))
        return out, None

    return None, "unsupported format"


async def gen(
    cdx: Any,
    *,
    model: str,
    msgs: list[Msg],
    imgs: Tmpimgs,
    sysm: str,
    effort: str | None,
    fmt: dict[str, Any] | None,
    tries: int = 2,
) -> tuple[str, Usage | None]:
    use: Usage | None = None
    msgs2 = list(msgs)
    last: str = ""

    for attempt in range(max(0, int(tries)) + 1):
        lead = fmtlead(fmt, BASELEAD)
        prompt, paths = await mkprompt(msgs2, system=sysm, imgs=imgs, lead=lead)
        res: Result = await cdx.run(Call(model=model, prompt=prompt, imgs=paths, effort=effort))
        use = useadd(use, res.usage)
        last = res.text

        if not fmt:
            return last, use

        out, em = fmtout(fmt, last)
        if out is not None:
            return out, use

        if attempt >= int(tries):
            break

        msgs2.append(
            Msg(
                role="user",
                content=(
                    f"Your output was invalid structured JSON ({em}). "
                    "Try again and output only JSON that matches the required format."
                ),
            )
        )

    raise ValueError("model returned invalid structured output")


def split(s: str, *, size: int = 800) -> list[str]:
    n = max(1, int(size))
    return [s[i : i + n] for i in range(0, len(s), n)] or [""]


def mkapp(cfg: Cfg | None = None, codex: Any | None = None, tools: Any | None = None) -> FastAPI:
    cfg = cfg or Cfg.env()
    codex = codex or Codex(bin=cfg.bin, cwd=cfg.cwd, timeout=cfg.timeout)
    tools = tools or Tools()

    app = FastAPI()

    app.state.cfg = cfg
    app.state.codex = codex
    app.state.tools = tools
    app.state.store: dict[str, dict[str, Any]] = {}
    app.state.items: dict[str, list[dict[str, Any]]] = {}

    @app.get("/healthz")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        now = int(time.time())
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": now, "owned_by": "codex"}
                    for m in MODELS
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat(body: Chatreq, request: Request):
        cdx = request.app.state.codex

        bad = chkmodel(body.model)
        if bad:
            return bad

        bad = chkeff(body.reasoning_effort, param="reasoning_effort")
        if bad:
            return bad

        fmt: dict[str, Any] | None = None
        if body.response_format is not None:
            try:
                fmt = fmtcanon(body.response_format)
            except ValueError as exc:
                return err(
                    400,
                    f"invalid response_format: {exc}",
                    param="response_format",
                    code="invalid_response_format",
                )

        cid = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())

        if body.stream:
            opts = body.stream_options

            if fmt:
                imgs = Tmpimgs()
                try:
                    text, use = await gen(
                        cdx,
                        model=body.model,
                        msgs=body.messages,
                        imgs=imgs,
                        sysm=_system(None),
                        effort=body.reasoning_effort,
                        fmt=fmt,
                    )
                except ImgErr as exc:
                    imgs.close()
                    return err(400, f"image error: {exc}", param="messages", code="invalid_image")
                except CodexError as exc:
                    imgs.close()
                    return err(502, f"codex failed: {exc}", type="server_error", code="codex_error")
                except ValueError as exc:
                    imgs.close()
                    return err(502, f"structured output failed: {exc}", type="server_error", code="invalid_model_output")
                finally:
                    imgs.close()

                def stream():
                    sentrole = False
                    for d in split(text):
                        delta: dict[str, str] = {"content": d}
                        if not sentrole:
                            delta = {"role": "assistant", "content": d}
                            sentrole = True

                        chunk = {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": body.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        yield ssedata(chunk)

                    finish = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield ssedata(finish)

                    if opts and opts.include_usage and use is not None:
                        yield ssedata(
                            {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": body.model,
                                "choices": [],
                                "usage": chatusage(use),
                            }
                        )

                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            imgs = Tmpimgs()
            try:
                prompt, paths = await mkprompt(body.messages, system=_system(None), imgs=imgs)
            except ImgErr as exc:
                imgs.close()
                return err(400, f"image error: {exc}", param="messages", code="invalid_image")

            call = Call(model=body.model, prompt=prompt, imgs=paths, effort=body.reasoning_effort)

            async def stream():
                sentrole = False
                try:
                    async for up in cdx.stream(call):
                        if up.delta:
                            delta: dict[str, str] = {"content": up.delta}
                            if not sentrole:
                                delta = {"role": "assistant", "content": up.delta}
                                sentrole = True

                            chunk = {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": body.model,
                                "choices": [
                                    {"index": 0, "delta": delta, "finish_reason": None}
                                ],
                            }
                            yield ssedata(chunk)

                        if up.done:
                            finish = {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": body.model,
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ],
                            }
                            yield ssedata(finish)

                            if opts and opts.include_usage and up.usage is not None:
                                yield ssedata(
                                    {
                                        "id": cid,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": body.model,
                                        "choices": [],
                                        "usage": chatusage(up.usage),
                                    }
                                )

                            yield "data: [DONE]\n\n"

                except CodexError as exc:
                    yield ssedata(errpayload(str(exc), type="server_error", code="codex_error"))
                    yield "data: [DONE]\n\n"
                finally:
                    imgs.close()

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        imgs = Tmpimgs()
        try:
            if fmt:
                text, use = await gen(
                    cdx,
                    model=body.model,
                    msgs=body.messages,
                    imgs=imgs,
                    sysm=_system(None),
                    effort=body.reasoning_effort,
                    fmt=fmt,
                )
                res = Result(text=text, usage=use)
            else:
                prompt, paths = await mkprompt(body.messages, system=_system(None), imgs=imgs)
                res = await cdx.run(Call(model=body.model, prompt=prompt, imgs=paths, effort=body.reasoning_effort))
        except ImgErr as exc:
            return err(400, f"image error: {exc}", param="messages", code="invalid_image")
        except CodexError as exc:
            return err(502, f"codex failed: {exc}", type="server_error", code="codex_error")
        except ValueError as exc:
            return err(502, f"structured output failed: {exc}", type="server_error", code="invalid_model_output")
        finally:
            imgs.close()

        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": res.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": chatusage(res.usage),
            }
        )

    @app.post("/v1/responses")
    async def responses(body: Respreq, request: Request):
        cdx = request.app.state.codex

        bad = chkmodel(body.model)
        if bad:
            return bad

        if body.background:
            return err(
                400,
                "background is not supported",
                param="background",
                code="not_supported",
            )

        effort = body.reasoning.effort if body.reasoning else None
        bad = chkeff(effort, param="reasoning.effort")
        if bad:
            return bad

        prev = None
        if body.previous_response_id:
            prev = request.app.state.store.get(body.previous_response_id)
            if prev is None:
                return err(
                    404,
                    f"unknown previous_response_id '{body.previous_response_id}'",
                    param="previous_response_id",
                    code="not_found",
                )

        try:
            msgs, items = norminput(body.input)
        except ValueError as exc:
            return err(400, f"invalid input: {exc}", param="input", code="invalid_input")

        if prev and prev.get("output_text"):
            msgs.insert(0, Msg(role="assistant", content=prev["output_text"]))

        try:
            tdefs, topts = toolset(body.tools)
        except ValueError as exc:
            return err(400, f"invalid tools: {exc}", param="tools", code="invalid_tools")

        try:
            tchoice = toolchoice(body.tool_choice, topts)
        except ValueError as exc:
            return err(400, f"invalid tool_choice: {exc}", param="tool_choice", code="invalid_tool_choice")

        toolon = bool(topts) and tchoice != "none"
        sysm = _system(body.instructions)

        fmt: dict[str, Any] | None = None
        fraw: Any | None = None
        fparam = None
        if isinstance(body.text, dict) and "format" in body.text:
            fraw = body.text.get("format")
            fparam = "text.format"
        elif body.response_format is not None:
            fraw = body.response_format
            fparam = "response_format"

        if fraw is not None:
            try:
                fmt = fmtcanon(fraw)
            except ValueError as exc:
                return err(400, f"invalid response format: {exc}", param=fparam, code="invalid_response_format")

        imgs = Tmpimgs()
        try:
            prompt0, paths0 = await mkprompt(msgs, system=sysm, imgs=imgs, lead=fmtlead(fmt, BASELEAD))
        except ImgErr as exc:
            imgs.close()
            return err(400, f"image error: {exc}", param="input", code="invalid_image")

        rid = f"resp_{uuid4().hex}"
        mid = f"msg_{uuid4().hex}"
        created = int(time.time())

        if body.stream and toolon:
            t = request.app.state.tools

            async def stream():
                parts: list[str] = []
                use: Usage | None = None
                msgs2 = list(msgs)
                outs: list[dict[str, Any]] = []
                called = 0
                forced = tchoice.get("type") if isinstance(tchoice, dict) else None

                try:
                    base = {
                        "id": rid,
                        "object": "response",
                        "created_at": created,
                        "status": "in_progress",
                        "error": None,
                        "incomplete_details": None,
                        "instructions": body.instructions,
                        "max_output_tokens": body.max_output_tokens,
                        "model": body.model,
                        "output": [],
                        "output_text": "",
                        "parallel_tool_calls": False,
                        "temperature": body.temperature,
                        "top_p": body.top_p,
                        "tool_choice": tchoice,
                        "tools": tdefs,
                        "usage": None,
                        "metadata": body.metadata or {},
                    }
                    yield sseevt("response.created", {"response": base})
                    yield sseevt("response.in_progress", {"response": base})

                    for _ in range(8):
                        lead = actlead(topts, tchoice)
                        prompt, paths = await mkprompt(msgs2, system=sysm, imgs=imgs, lead=lead)
                        res = await cdx.run(Call(model=body.model, prompt=prompt, imgs=paths, effort=effort))
                        use = useadd(use, res.usage)

                        typ, name, args, finaltxt = actparse(res.text)
                        if typ == "final":
                            if (tchoice == "required" or forced) and called == 0:
                                msgs2.append(
                                    Msg(
                                        role="user",
                                        content="Tool use is required. Call a tool before finishing.",
                                    )
                                )
                                continue
                            break

                        if not name:
                            msgs2.append(Msg(role="user", content="Tool name missing. Output JSON per schema."))
                            continue

                        if name not in topts:
                            msgs2.append(Msg(role="user", content=f"Tool '{name}' is not enabled."))
                            continue

                        if forced and called == 0 and name != forced:
                            msgs2.append(Msg(role="user", content=f"You must call {forced} tool first."))
                            continue

                        outidx = len(outs)
                        tid = ("ws_" if name == "web_search" else "ci_") + uuid4().hex

                        start: dict[str, Any] = {
                            "id": tid,
                            "type": "web_search_call" if name == "web_search" else "code_interpreter_call",
                            "status": "in_progress",
                        }
                        if name == "web_search":
                            q = (args or {}).get("query") or (args or {}).get("q") or ""
                            start["query"] = q if isinstance(q, str) else str(q)
                        if name == "code_interpreter":
                            c = (args or {}).get("code") or (args or {}).get("python") or ""
                            start["code"] = c if isinstance(c, str) else str(c)
                            start["container_id"] = "local"
                            start["outputs"] = []

                        yield sseevt(
                            "response.output_item.added",
                            {"response_id": rid, "output_index": outidx, "item": start},
                        )

                        if name == "web_search":
                            yield sseevt(
                                "response.web_search_call.in_progress",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )
                            yield sseevt(
                                "response.web_search_call.searching",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )
                        else:
                            yield sseevt(
                                "response.code_interpreter_call.in_progress",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )
                            yield sseevt(
                                "response.code_interpreter_call.interpreting",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )

                        item, msg = await dotool(t, name, args or {}, topts, tid=tid)
                        outs.append(item)
                        msgs2.append(msg)
                        called += 1

                        if name == "web_search":
                            yield sseevt(
                                "response.web_search_call.completed",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )
                        else:
                            yield sseevt(
                                "response.code_interpreter_call.completed",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )

                        yield sseevt(
                            "response.output_item.done",
                            {"response_id": rid, "output_index": outidx, "item": item},
                        )

                    msgidx = len(outs)
                    msgitem = {
                        "id": mid,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "", "annotations": []}],
                    }
                    yield sseevt(
                        "response.output_item.added",
                        {"response_id": rid, "output_index": msgidx, "item": msgitem},
                    )

                    if fmt:
                        text, use2 = await gen(
                            cdx,
                            model=body.model,
                            msgs=msgs2,
                            imgs=imgs,
                            sysm=sysm,
                            effort=effort,
                            fmt=fmt,
                        )
                        use = useadd(use, use2)
                        for d in split(text):
                            parts.append(d)
                            yield sseevt(
                                "response.output_text.delta",
                                {
                                    "response_id": rid,
                                    "output_index": msgidx,
                                    "item_id": mid,
                                    "content_index": 0,
                                    "delta": d,
                                },
                            )
                    else:
                        prompt, paths = await mkprompt(msgs2, system=sysm, imgs=imgs, lead=BASELEAD)
                        call = Call(model=body.model, prompt=prompt, imgs=paths, effort=effort)

                        async for up in cdx.stream(call):
                            if up.delta:
                                parts.append(up.delta)
                                yield sseevt(
                                    "response.output_text.delta",
                                    {
                                        "response_id": rid,
                                        "output_index": msgidx,
                                        "item_id": mid,
                                        "content_index": 0,
                                        "delta": up.delta,
                                    },
                                )

                            if up.usage is not None:
                                use = useadd(use, up.usage)

                            if up.done:
                                break

                    text = "".join(parts)
                    final = mkresp(
                        rid=rid,
                        mid=mid,
                        model=body.model,
                        created=created,
                        text=text,
                        usage=use,
                        instr=body.instructions,
                        meta=body.metadata,
                        maxout=body.max_output_tokens,
                        status="completed",
                        outs=outs,
                        tools=tdefs,
                        tool_choice=tchoice,
                    )

                    yield sseevt(
                        "response.output_text.done",
                        {
                            "response_id": rid,
                            "output_index": msgidx,
                            "item_id": mid,
                            "content_index": 0,
                            "text": text,
                        },
                    )
                    yield sseevt(
                        "response.output_item.done",
                        {"response_id": rid, "output_index": msgidx, "item": final["output"][msgidx]},
                    )

                    if body.store:
                        request.app.state.store[rid] = final
                        request.app.state.items[rid] = items

                    yield sseevt("response.completed", {"response": final})

                except CodexError as exc:
                    fail = mkresp(
                        rid=rid,
                        mid=mid,
                        model=body.model,
                        created=created,
                        text="",
                        usage=None,
                        instr=body.instructions,
                        meta=body.metadata,
                        maxout=body.max_output_tokens,
                        status="failed",
                        outs=outs,
                        tools=tdefs,
                        tool_choice=tchoice,
                        error={"message": str(exc), "type": "server_error", "code": "codex_error"},
                    )
                    if body.store:
                        request.app.state.store[rid] = fail
                        request.app.state.items[rid] = items
                    yield sseevt("response.failed", {"response": fail})
                except ImgErr as exc:
                    fail = mkresp(
                        rid=rid,
                        mid=mid,
                        model=body.model,
                        created=created,
                        text="",
                        usage=None,
                        instr=body.instructions,
                        meta=body.metadata,
                        maxout=body.max_output_tokens,
                        status="failed",
                        outs=outs,
                        tools=tdefs,
                        tool_choice=tchoice,
                        error={"message": str(exc), "type": "invalid_request_error", "code": "invalid_image"},
                    )
                    if body.store:
                        request.app.state.store[rid] = fail
                        request.app.state.items[rid] = items
                    yield sseevt("response.failed", {"response": fail})
                except ValueError as exc:
                    fail = mkresp(
                        rid=rid,
                        mid=mid,
                        model=body.model,
                        created=created,
                        text="",
                        usage=None,
                        instr=body.instructions,
                        meta=body.metadata,
                        maxout=body.max_output_tokens,
                        status="failed",
                        outs=outs,
                        tools=tdefs,
                        tool_choice=tchoice,
                        error={"message": str(exc), "type": "server_error", "code": "invalid_model_output"},
                    )
                    if body.store:
                        request.app.state.store[rid] = fail
                        request.app.state.items[rid] = items
                    yield sseevt("response.failed", {"response": fail})
                finally:
                    imgs.close()

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        if body.stream:
            call = Call(model=body.model, prompt=prompt0, imgs=paths0, effort=effort)

            async def stream():
                parts: list[str] = []
                use: Usage | None = None
                try:
                    base = mkresp(
                        rid=rid,
                        mid=mid,
                        model=body.model,
                        created=created,
                        text="",
                        usage=None,
                        instr=body.instructions,
                        meta=body.metadata,
                        maxout=body.max_output_tokens,
                        status="in_progress",
                        tools=tdefs,
                        tool_choice=tchoice,
                    )
                    yield sseevt("response.created", {"response": base})
                    yield sseevt("response.in_progress", {"response": base})
                    yield sseevt(
                        "response.output_item.added",
                        {"response_id": rid, "output_index": 0, "item": base["output"][0]},
                    )

                    if fmt:
                        text, use = await gen(
                            cdx,
                            model=body.model,
                            msgs=msgs,
                            imgs=imgs,
                            sysm=sysm,
                            effort=effort,
                            fmt=fmt,
                        )
                        for d in split(text):
                            if not d:
                                continue
                            yield sseevt(
                                "response.output_text.delta",
                                {
                                    "response_id": rid,
                                    "output_index": 0,
                                    "item_id": mid,
                                    "content_index": 0,
                                    "delta": d,
                                },
                            )

                        final = mkresp(
                            rid=rid,
                            mid=mid,
                            model=body.model,
                            created=created,
                            text=text,
                            usage=use,
                            instr=body.instructions,
                            meta=body.metadata,
                            maxout=body.max_output_tokens,
                            status="completed",
                            tools=tdefs,
                            tool_choice=tchoice,
                        )

                        yield sseevt(
                            "response.output_text.done",
                            {
                                "response_id": rid,
                                "output_index": 0,
                                "item_id": mid,
                                "content_index": 0,
                                "text": text,
                            },
                        )
                        yield sseevt(
                            "response.output_item.done",
                            {"response_id": rid, "output_index": 0, "item": final["output"][0]},
                        )

                        if body.store:
                            request.app.state.store[rid] = final
                            request.app.state.items[rid] = items

                        yield sseevt("response.completed", {"response": final})
                        return

                    async for up in cdx.stream(call):
                        if up.delta:
                            parts.append(up.delta)
                            yield sseevt(
                                "response.output_text.delta",
                                {
                                    "response_id": rid,
                                    "output_index": 0,
                                    "item_id": mid,
                                    "content_index": 0,
                                    "delta": up.delta,
                                },
                            )

                        if up.usage is not None:
                            use = up.usage

                        if up.done:
                            text = "".join(parts)
                            final = mkresp(
                                rid=rid,
                                mid=mid,
                                model=body.model,
                                created=created,
                                text=text,
                                usage=use,
                                instr=body.instructions,
                                meta=body.metadata,
                                maxout=body.max_output_tokens,
                                status="completed",
                                tools=tdefs,
                                tool_choice=tchoice,
                            )

                            yield sseevt(
                                "response.output_text.done",
                                {
                                    "response_id": rid,
                                    "output_index": 0,
                                    "item_id": mid,
                                    "content_index": 0,
                                    "text": text,
                                },
                            )
                            yield sseevt(
                                "response.output_item.done",
                                {"response_id": rid, "output_index": 0, "item": final["output"][0]},
                            )

                            if body.store:
                                request.app.state.store[rid] = final
                                request.app.state.items[rid] = items

                            yield sseevt("response.completed", {"response": final})

                except CodexError as exc:
                    fail = mkresp(
                        rid=rid,
                        mid=mid,
                        model=body.model,
                        created=created,
                        text="",
                        usage=None,
                        instr=body.instructions,
                        meta=body.metadata,
                        maxout=body.max_output_tokens,
                        status="failed",
                        tools=tdefs,
                        tool_choice=tchoice,
                        error={"message": str(exc), "type": "server_error", "code": "codex_error"},
                    )
                    if body.store:
                        request.app.state.store[rid] = fail
                        request.app.state.items[rid] = items
                    yield sseevt("response.failed", {"response": fail})
                except ImgErr as exc:
                    fail = mkresp(
                        rid=rid,
                        mid=mid,
                        model=body.model,
                        created=created,
                        text="",
                        usage=None,
                        instr=body.instructions,
                        meta=body.metadata,
                        maxout=body.max_output_tokens,
                        status="failed",
                        tools=tdefs,
                        tool_choice=tchoice,
                        error={"message": str(exc), "type": "invalid_request_error", "code": "invalid_image"},
                    )
                    if body.store:
                        request.app.state.store[rid] = fail
                        request.app.state.items[rid] = items
                    yield sseevt("response.failed", {"response": fail})
                except ValueError as exc:
                    fail = mkresp(
                        rid=rid,
                        mid=mid,
                        model=body.model,
                        created=created,
                        text="",
                        usage=None,
                        instr=body.instructions,
                        meta=body.metadata,
                        maxout=body.max_output_tokens,
                        status="failed",
                        tools=tdefs,
                        tool_choice=tchoice,
                        error={"message": str(exc), "type": "server_error", "code": "invalid_model_output"},
                    )
                    if body.store:
                        request.app.state.store[rid] = fail
                        request.app.state.items[rid] = items
                    yield sseevt("response.failed", {"response": fail})
                finally:
                    imgs.close()

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        if toolon:
            t = request.app.state.tools
            msgs2 = list(msgs)
            outs: list[dict[str, Any]] = []
            called = 0
            forced = tchoice.get("type") if isinstance(tchoice, dict) else None
            use: Usage | None = None

            try:
                for _ in range(8):
                    lead = actlead(topts, tchoice)
                    prompt, paths = await mkprompt(msgs2, system=sysm, imgs=imgs, lead=lead)
                    res: Result = await cdx.run(Call(model=body.model, prompt=prompt, imgs=paths, effort=effort))
                    use = useadd(use, res.usage)

                    typ, name, args, finaltxt = actparse(res.text)
                    if typ == "final":
                        if (tchoice == "required" or forced) and called == 0:
                            msgs2.append(
                                Msg(
                                    role="user",
                                    content="Tool use is required. Call a tool before finishing.",
                                )
                            )
                            continue
                        break

                    if not name or name not in topts:
                        msgs2.append(Msg(role="user", content=f"Tool '{name or ''}' is not enabled."))
                        continue

                    if forced and called == 0 and name != forced:
                        msgs2.append(Msg(role="user", content=f"You must call {forced} tool first."))
                        continue

                    item, msg = await dotool(t, name, args or {}, topts)
                    outs.append(item)
                    msgs2.append(msg)
                    called += 1

                text, use2 = await gen(
                    cdx,
                    model=body.model,
                    msgs=msgs2,
                    imgs=imgs,
                    sysm=sysm,
                    effort=effort,
                    fmt=fmt,
                )
                use = useadd(use, use2)
                resp = mkresp(
                    rid=rid,
                    mid=mid,
                    model=body.model,
                    created=created,
                    text=text,
                    usage=use,
                    instr=body.instructions,
                    meta=body.metadata,
                    maxout=body.max_output_tokens,
                    status="completed",
                    outs=outs,
                    tools=tdefs,
                    tool_choice=tchoice,
                )
                if body.store:
                    request.app.state.store[rid] = resp
                    request.app.state.items[rid] = items
                return JSONResponse(resp)

            except ImgErr as exc:
                imgs.close()
                return err(400, f"image error: {exc}", param="input", code="invalid_image")
            except CodexError as exc:
                imgs.close()
                return err(502, f"codex failed: {exc}", type="server_error", code="codex_error")
            except ValueError as exc:
                imgs.close()
                return err(502, f"structured output failed: {exc}", type="server_error", code="invalid_model_output")
            finally:
                imgs.close()

        call = Call(model=body.model, prompt=prompt0, imgs=paths0, effort=effort)

        try:
            if fmt:
                text, use = await gen(
                    cdx,
                    model=body.model,
                    msgs=msgs,
                    imgs=imgs,
                    sysm=sysm,
                    effort=effort,
                    fmt=fmt,
                )
                res = Result(text=text, usage=use)
            else:
                res = await cdx.run(call)
        except ImgErr as exc:
            imgs.close()
            return err(400, f"image error: {exc}", param="input", code="invalid_image")
        except CodexError as exc:
            imgs.close()
            return err(502, f"codex failed: {exc}", type="server_error", code="codex_error")
        except ValueError as exc:
            imgs.close()
            return err(502, f"structured output failed: {exc}", type="server_error", code="invalid_model_output")
        finally:
            imgs.close()

        resp = mkresp(
            rid=rid,
            mid=mid,
            model=body.model,
            created=created,
            text=res.text,
            usage=res.usage,
            instr=body.instructions,
            meta=body.metadata,
            maxout=body.max_output_tokens,
            status="completed",
            tools=tdefs,
            tool_choice=tchoice,
        )

        if body.store:
            request.app.state.store[rid] = resp
            request.app.state.items[rid] = items

        return JSONResponse(resp)

    @app.get("/v1/responses/{rid}")
    async def getresp(rid: str, request: Request):
        got = request.app.state.store.get(rid)
        if got is None:
            return err(404, f"response '{rid}' not found", param="response_id", code="not_found")
        return JSONResponse(got)

    @app.get("/v1/responses/{rid}/input_items")
    async def getitems(rid: str, request: Request):
        if rid not in request.app.state.store:
            return err(404, f"response '{rid}' not found", param="response_id", code="not_found")

        data = request.app.state.items.get(rid, [])
        return JSONResponse(
            {
                "object": "list",
                "data": data,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None,
                "has_more": False,
            }
        )

    @app.post("/v1/responses/{rid}/cancel")
    async def cancel(rid: str, request: Request):
        if rid not in request.app.state.store:
            return err(404, f"response '{rid}' not found", param="response_id", code="not_found")
        return err(409, "cancel is not supported", code="not_supported")

    @app.delete("/v1/responses/{rid}")
    async def delresp(rid: str, request: Request):
        if rid not in request.app.state.store:
            return err(404, f"response '{rid}' not found", param="response_id", code="not_found")

        request.app.state.store.pop(rid, None)
        request.app.state.items.pop(rid, None)
        return JSONResponse({"id": rid, "object": "response", "deleted": True})

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def nov1(path: str) -> JSONResponse:
        return err(404, f"/v1/{path} not implemented", code="not_implemented")

    return app


def chkmodel(model: str) -> JSONResponse | None:
    if model in MODELS:
        return None
    return err(
        400,
        f"unsupported model '{model}'. allowed: {', '.join(MODELS)}",
        param="model",
        code="model_not_supported",
    )


def chkeff(eff: str | None, *, param: str) -> JSONResponse | None:
    if not eff:
        return None
    if eff in EFFORTS:
        return None
    return err(
        400,
        f"unsupported reasoning effort '{eff}'. allowed: {', '.join(EFFORTS)}",
        param=param,
        code="reasoning_not_supported",
    )


def _system(extra: str | None) -> str:
    if not extra:
        return SYSTEM
    return f"{SYSTEM}\n\nAdditional instructions:\n{extra.strip()}"


async def mkprompt(
    msgs: list[Msg], *, system: str, imgs: Tmpimgs, lead: str | None = None
) -> tuple[str, list[Path]]:
    lead = lead or "Use the conversation below and respond to the latest user request."
    lines: list[str] = [system.strip(), "", lead, "",]

    paths: list[Path] = []
    idx = 0

    for m in msgs:
        lines.append(f"[{m.role}]")
        c = m.content

        if isinstance(c, str):
            lines.append(c)
        elif isinstance(c, list):
            for p in c:
                t = (p.type or "").strip().lower()
                if t == "text":
                    lines.append(p.text or "")
                elif t == "image_url" and p.image_url:
                    idx += 1
                    path = await imgs.get(p.image_url.url)
                    detail = p.image_url.detail or "auto"
                    lines.append(f"[image {idx}: {path.name}, detail={detail}]")
                    paths.append(path)
                else:
                    lines.append(f"[unsupported content: {p.type}]")
        elif c is None:
            lines.append("")
        else:
            lines.append(str(c))

        lines.append("")

    return "\n".join(lines).strip() + "\n", paths


def norminput(raw: str | list[dict[str, Any]] | None) -> tuple[list[Msg], list[dict[str, Any]]]:
    if raw is None:
        raise ValueError("input is required")

    if isinstance(raw, str):
        return [Msg(role="user", content=raw)], [
            {
                "id": f"initem_{uuid4().hex}",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": raw}],
            }
        ]

    if not isinstance(raw, list) or not raw:
        raise ValueError("input must be a non-empty string or list")

    msgs: list[Msg] = []
    items: list[dict[str, Any]] = []

    for it in raw:
        if not isinstance(it, dict):
            raise ValueError("each input item must be an object")

        role = str(it.get("role") or "user")
        content = it.get("content")
        if isinstance(content, dict):
            content = [content]

        out: list[dict[str, Any]] = []

        if isinstance(content, str):
            msgs.append(Msg(role=role, content=content))
            out.append({"type": "input_text", "text": content})

        elif isinstance(content, list):
            parts: list[Part] = []
            for blk in content:
                if not isinstance(blk, dict):
                    continue

                typ = str(blk.get("type") or "").strip().lower()

                if typ in {"input_text", "text", "output_text"}:
                    text = str(blk.get("text") or blk.get("content") or "")
                    parts.append(Part(type="text", text=text))
                    out.append({"type": "input_text", "text": text})
                    continue

                if typ in {"input_image", "image_url"}:
                    url, detail = imgblk(blk)
                    if not url:
                        raise ValueError("input_image needs image_url or file_data")
                    parts.append(Part(type="image_url", image_url=ImageURL(url=url, detail=detail)))
                    x = {"type": "input_image", "image_url": url}
                    if detail:
                        x["detail"] = detail
                    out.append(x)
                    continue

                if typ in {"input_file", "file"}:
                    url, detail = imgblk(blk)
                    if url:
                        parts.append(Part(type="image_url", image_url=ImageURL(url=url, detail=detail)))
                        x = {"type": "input_image", "image_url": url}
                        if detail:
                            x["detail"] = detail
                        out.append(x)
                    else:
                        text = f"[input_file: {fileinfo(blk)}]"
                        parts.append(Part(type="text", text=text))
                        out.append({"type": "input_text", "text": text})
                    continue

                text = f"[unsupported content: {blk.get('type')}]"
                parts.append(Part(type="text", text=text))
                out.append({"type": "input_text", "text": text})

            msgs.append(Msg(role=role, content=parts))

        elif content is None:
            msgs.append(Msg(role=role, content=""))
            out.append({"type": "input_text", "text": ""})

        else:
            raise ValueError("content must be string or list")

        items.append({"id": f"initem_{uuid4().hex}", "type": "message", "role": role, "content": out})

    if not msgs:
        raise ValueError("no valid messages")

    return msgs, items


def imgblk(blk: dict[str, Any]) -> tuple[str | None, str | None]:
    detail = blk.get("detail") if isinstance(blk.get("detail"), str) and blk.get("detail").strip() else None

    if isinstance(blk.get("image_url"), str):
        return blk["image_url"], detail

    img = blk.get("image_url")
    if isinstance(img, dict) and isinstance(img.get("url"), str):
        return img["url"], (img.get("detail") if isinstance(img.get("detail"), str) else detail)

    if isinstance(blk.get("file_url"), str):
        return blk["file_url"], detail

    if isinstance(blk.get("file_data"), str):
        data = blk["file_data"]
        if data.startswith("data:"):
            return data, detail
        mime = blk.get("mime_type") if isinstance(blk.get("mime_type"), str) else "application/octet-stream"
        return f"data:{mime};base64,{data}", detail

    if isinstance(blk.get("file_id"), str):
        raise ValueError("file_id is not supported; provide image_url or file_data")

    return None, detail


def fileinfo(blk: dict[str, Any]) -> str:
    name = blk.get("filename") if isinstance(blk.get("filename"), str) else None
    mime = blk.get("mime_type") if isinstance(blk.get("mime_type"), str) else None

    if name and mime:
        return f"filename={name}, mime={mime}"
    if name:
        return f"filename={name}"
    if mime:
        return f"mime={mime}"
    return "binary"


def mkresp(
    *,
    rid: str,
    mid: str,
    model: str,
    created: int,
    text: str,
    usage: Usage | None,
    instr: str | None,
    meta: dict[str, Any] | None,
    maxout: int | None,
    status: str,
    outs: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    msg = {
        "id": mid,
        "type": "message",
        "status": "completed" if status == "completed" else status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }

    out = (outs or []) + [msg]
    return {
        "id": rid,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": error,
        "incomplete_details": None,
        "instructions": instr,
        "max_output_tokens": maxout,
        "model": model,
        "output": out,
        "output_text": text,
        "parallel_tool_calls": False,
        "temperature": None,
        "top_p": None,
        "tool_choice": tool_choice if tool_choice is not None else "none",
        "tools": tools or [],
        "usage": respusage(usage),
        "metadata": meta or {},
    }


def chatusage(u: Usage | None) -> dict[str, int | None] | None:
    if u is None:
        return None
    return {"prompt_tokens": u.prompt, "completion_tokens": u.completion, "total_tokens": u.total}


def respusage(u: Usage | None) -> dict[str, Any] | None:
    if u is None:
        return None
    return {
        "input_tokens": u.prompt,
        "input_tokens_details": {"cached_tokens": None},
        "output_tokens": u.completion,
        "output_tokens_details": {"reasoning_tokens": None},
        "total_tokens": u.total,
    }


def ssedata(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\\n\\n"


def sseevt(typ: str, payload: dict[str, Any]) -> str:
    data = dict(payload)
    data["type"] = typ
    return f"event: {typ}\\ndata: {json.dumps(data, ensure_ascii=True)}\\n\\n"


app = mkapp()
