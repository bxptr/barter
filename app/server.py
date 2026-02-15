from __future__ import annotations

import json
import base64
import os
from pathlib import Path
import secrets
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.actions import actlead, actparse, dotool, toolchoice, toolset, useadd
from app.auth import bearer
from app.codex import Call, Codex, CodexError, Result, Usage
from app.config import Cfg, EFFORTS, MODELS, SYSTEM, defeff
from app.errors import autherr, err, errpayload
from app.format import BASELEAD, fmtcanon, fmtlead, gen, split
from app.images import ImgErr, Tmpimgs
from app.keydb import Keydb
from app.prompt import mkprompt, norminput
from app.response import chatusage, mkresp
from app.schema import Chatreq, Msg, Respreq
from app.stream import ssedata, sseevt
from app.tools import Tools


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


def mkapp(cfg: Cfg | None = None, codex: Any | None = None, tools: Any | None = None) -> FastAPI:
    cfg = cfg or Cfg.env()
    codex = codex or Codex(bin=cfg.bin, cwd=cfg.cwd, timeout=cfg.timeout)
    tools = tools or Tools()

    dbp = Path(os.getenv("BARTER_DB", str(cfg.cwd / "barter.db"))).expanduser().resolve()
    admin = os.getenv("BARTER_ADMIN_KEY", "").strip()
    if admin and not admin.startswith("brtr-"):
        raise RuntimeError("BARTER_ADMIN_KEY must start with 'brtr-'")
    keydb = Keydb(dbp)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    app.state.cfg = cfg
    app.state.codex = codex
    app.state.tools = tools
    app.state.keydb = keydb
    app.state.admin = admin
    app.state.store: dict[str, dict[str, Any]] = {}
    app.state.items: dict[str, list[dict[str, Any]]] = {}

    @app.on_event("shutdown")
    def _shutdown() -> None:
        try:
            request_keydb = getattr(app.state, "keydb", None)
            if request_keydb:
                request_keydb.close()
        except Exception:
            pass

    @app.middleware("http")
    async def _auth(req: Request, call_next):
        path = req.url.path or "/"
        if path == "/healthz":
            return await call_next(req)
        if req.method == "OPTIONS":
            return await call_next(req)

        key = bearer(req)
        if not key:
            return autherr()

        if admin and secrets.compare_digest(key, admin):
            req.state.keyid = "admin"
            req.state.admin = True
            return await call_next(req)

        rec = req.app.state.keydb.check(key)
        if not rec:
            return autherr()

        req.state.keyid = rec["id"]
        req.state.admin = bool(rec.get("admin"))
        return await call_next(req)

    @app.get("/healthz")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/v1/keys")
    async def mkkey(request: Request):
        if not bool(getattr(request.state, "admin", False)):
            return err(
                403,
                "admin key required",
                type="permission_error",
                code="insufficient_permissions",
            )

        try:
            body = await request.json()
        except Exception:
            body = {}

        name = body.get("name") if isinstance(body, dict) else None
        name = name.strip() if isinstance(name, str) and name.strip() else None
        admin = bool(body.get("admin")) if isinstance(body, dict) and body.get("admin") is not None else False

        try:
            out = request.app.state.keydb.add(name=name, admin=admin)
        except Exception as exc:
            return err(502, f"key db failed: {exc}", type="server_error", code="db_error")

        return JSONResponse(out)

    @app.get("/v1/keys")
    async def lskeys(request: Request):
        if not bool(getattr(request.state, "admin", False)):
            return err(
                403,
                "admin key required",
                type="permission_error",
                code="insufficient_permissions",
            )

        return JSONResponse({"object": "list", "data": request.app.state.keydb.list()})

    @app.delete("/v1/keys/{kid}")
    async def delkey(kid: str, request: Request):
        if not bool(getattr(request.state, "admin", False)):
            return err(
                403,
                "admin key required",
                type="permission_error",
                code="insufficient_permissions",
            )

        ok = request.app.state.keydb.revoke(kid)
        if not ok:
            return err(404, f"key '{kid}' not found", param="kid", code="not_found")
        return JSONResponse({"id": kid, "object": "api_key", "deleted": True})

    @app.get("/v1/models")
    @app.get("/models")
    async def models() -> JSONResponse:
        now = int(time.time())
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": m, "object": "model", "created": now, "owned_by": "codex"} for m in MODELS],
            }
        )

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat(body: Chatreq, request: Request):
        cdx = request.app.state.codex

        bad = chkmodel(body.model)
        if bad:
            return bad

        effort = defeff(body.reasoning_effort)
        bad = chkeff(effort, param="reasoning_effort")
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
                        effort=effort,
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
                    return err(
                        502,
                        f"structured output failed: {exc}",
                        type="server_error",
                        code="invalid_model_output",
                    )
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

            call = Call(model=body.model, prompt=prompt, imgs=paths, effort=effort)

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
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                            }
                            yield ssedata(chunk)

                        if up.done:
                            finish = {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": body.model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
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
                    effort=effort,
                    fmt=fmt,
                )
                res = Result(text=text, usage=use)
            else:
                prompt, paths = await mkprompt(body.messages, system=_system(None), imgs=imgs)
                res = await cdx.run(Call(model=body.model, prompt=prompt, imgs=paths, effort=effort))
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
    @app.post("/responses")
    async def responses(body: Respreq, request: Request):
        cdx = request.app.state.codex

        bad = chkmodel(body.model)
        if bad:
            return bad

        if body.background:
            return err(400, "background is not supported", param="background", code="not_supported")

        effort = defeff(body.reasoning.effort if body.reasoning else None)
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
                forcedsrv = tchoice.get("server_label") if isinstance(tchoice, dict) else None
                forcedmcp = tchoice.get("name") if isinstance(tchoice, dict) else None
                seq = 0

                def ev(typ: str, payload: dict[str, Any]) -> str:
                    nonlocal seq
                    data = dict(payload)
                    data["sequence_number"] = seq
                    seq += 1
                    return sseevt(typ, data)

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
                    yield ev("response.created", {"response": base})
                    yield ev("response.in_progress", {"response": base})

                    mcp = topts.get("mcp") if isinstance(topts.get("mcp"), dict) else None
                    if mcp and isinstance(mcp.get("servers"), dict):
                        for label, srv in sorted(mcp["servers"].items()):
                            if not isinstance(srv, dict):
                                continue
                            url = srv.get("url")
                            if not isinstance(url, str) or not url.strip():
                                continue

                            headers: dict[str, str] = {}
                            if isinstance(srv.get("headers"), dict):
                                headers.update({str(k): str(v) for k, v in srv["headers"].items()})
                            auth = srv.get("authorization")
                            if isinstance(auth, str) and auth.strip() and "authorization" not in {k.lower() for k in headers}:
                                val = auth.strip()
                                headers["Authorization"] = val if " " in val else f"Bearer {val}"

                            outidx = len(outs)
                            tid = f"mcp_tools_{uuid4().hex}"
                            try:
                                tools = await t.mcptools(
                                    url,
                                    headers=headers or None,
                                    transport=srv.get("transport") if isinstance(srv.get("transport"), str) else None,
                                )
                                allow = srv.get("allowed_tools")
                                if isinstance(allow, list):
                                    allowset = {str(x) for x in allow}
                                    tools = [x for x in tools if isinstance(x, dict) and x.get("name") in allowset]
                                if isinstance(mcp.get("tools"), dict):
                                    mcp["tools"][label] = tools
                                item = {"id": tid, "type": "mcp_list_tools", "server_label": label, "tools": tools}
                            except Exception as exc:
                                if isinstance(mcp.get("tools"), dict):
                                    mcp["tools"][label] = []
                                item = {
                                    "id": tid,
                                    "type": "mcp_list_tools",
                                    "server_label": label,
                                    "tools": [],
                                    "error": str(exc),
                                }

                            outs.append(item)
                            yield ev(
                                "response.output_item.added",
                                {"response_id": rid, "output_index": outidx, "item": item},
                            )
                            yield ev(
                                "response.output_item.done",
                                {"response_id": rid, "output_index": outidx, "item": item},
                            )

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
                            msgs2.append(Msg(role="user", content="Tool name missing."))
                            continue
                        if name not in topts:
                            msgs2.append(Msg(role="user", content=f"Tool '{name}' is not enabled."))
                            continue

                        if forced and called == 0 and name != forced:
                            msgs2.append(Msg(role="user", content=f"You must call {forced} tool first."))
                            continue

                        if forced == "mcp" and called == 0:
                            asl = (args or {}).get("server_label")
                            anm = (args or {}).get("name") or (args or {}).get("tool") or (args or {}).get("tool_name")
                            asl = asl.strip() if isinstance(asl, str) else None
                            anm = anm.strip() if isinstance(anm, str) else None
                            if forcedsrv and asl and asl != forcedsrv:
                                msgs2.append(Msg(role="user", content=f"You must call mcp server_label={forcedsrv} first."))
                                continue
                        if forcedmcp and anm and anm != forcedmcp:
                            msgs2.append(Msg(role="user", content=f"You must call mcp name={forcedmcp} first."))
                            continue

                        outidx = len(outs)
                        tid = f"{('ws' if name == 'web_search' else 'ci' if name == 'code_interpreter' else 'mcp')}_{uuid4().hex}"
                        start: dict[str, Any] = {
                            "id": tid,
                            "type": (
                                "web_search_call"
                                if name == "web_search"
                                else "code_interpreter_call"
                                if name == "code_interpreter"
                                else "mcp_call"
                            ),
                            "status": "in_progress",
                        }
                        if name == "web_search":
                            q = (args or {}).get("query") or (args or {}).get("q") or ""
                            start["query"] = q if isinstance(q, str) else str(q)
                            start["action"] = {
                                "type": "search",
                                "query": start["query"],
                                "queries": [start["query"]],
                            }
                        if name == "code_interpreter":
                            c = (args or {}).get("code") or (args or {}).get("python") or ""
                            start["code"] = c if isinstance(c, str) else str(c)
                            start["container_id"] = "local"
                            start["outputs"] = []
                        if name == "mcp":
                            sl = (args or {}).get("server_label") or ""
                            nm = (args or {}).get("name") or (args or {}).get("tool") or ""
                            start["server_label"] = sl if isinstance(sl, str) else str(sl)
                            start["name"] = nm if isinstance(nm, str) else str(nm)
                            targs = (args or {}).get("arguments") or (args or {}).get("args") or {}
                            if isinstance(targs, str):
                                try:
                                    targs = json.loads(targs)
                                except json.JSONDecodeError:
                                    targs = {}
                            if targs is None:
                                targs = {}
                            if not isinstance(targs, dict):
                                targs = {}
                            start["arguments"] = json.dumps(targs, ensure_ascii=True, separators=(",", ":"))

                        yield ev("response.output_item.added", {"response_id": rid, "output_index": outidx, "item": start})

                        if name == "web_search":
                            yield ev(
                                "response.web_search_call.in_progress",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )
                            yield ev(
                                "response.web_search_call.searching",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )
                        elif name == "code_interpreter":
                            yield ev(
                                "response.code_interpreter_call.in_progress",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )
                            yield ev(
                                "response.code_interpreter_call.interpreting",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )

                        item, msg = await dotool(t, name, args or {}, topts, tid=tid)
                        outs.append(item)
                        msgs2.append(msg)
                        called += 1

                        if name == "web_search":
                            yield ev(
                                "response.web_search_call.completed",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )
                        elif name == "code_interpreter":
                            yield ev(
                                "response.code_interpreter_call.completed",
                                {"response_id": rid, "output_index": outidx, "item_id": tid},
                            )

                        yield ev("response.output_item.done", {"response_id": rid, "output_index": outidx, "item": item})

                    msgidx = len(outs)
                    msgitem = {
                        "id": mid,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "", "annotations": []}],
                    }
                    yield ev("response.output_item.added", {"response_id": rid, "output_index": msgidx, "item": msgitem})

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
                            yield ev(
                                "response.output_text.delta",
                                {
                                    "response_id": rid,
                                    "output_index": msgidx,
                                    "item_id": mid,
                                    "content_index": 0,
                                    "delta": d,
                                    "logprobs": [],
                                },
                            )
                    else:
                        prompt, paths = await mkprompt(msgs2, system=sysm, imgs=imgs, lead=BASELEAD)
                        call = Call(model=body.model, prompt=prompt, imgs=paths, effort=effort)

                        async for up in cdx.stream(call):
                            if up.delta:
                                parts.append(up.delta)
                                yield ev(
                                    "response.output_text.delta",
                                    {
                                        "response_id": rid,
                                        "output_index": msgidx,
                                        "item_id": mid,
                                        "content_index": 0,
                                        "delta": up.delta,
                                        "logprobs": [],
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

                    yield ev(
                        "response.output_text.done",
                        {
                            "response_id": rid,
                            "output_index": msgidx,
                            "item_id": mid,
                            "content_index": 0,
                            "text": text,
                            "logprobs": [],
                        },
                    )
                    yield ev(
                        "response.output_item.done",
                        {"response_id": rid, "output_index": msgidx, "item": final["output"][msgidx]},
                    )

                    if body.store:
                        request.app.state.store[rid] = final
                        request.app.state.items[rid] = items

                    yield ev("response.completed", {"response": final})

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
                    yield ev("response.failed", {"response": fail})
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
                    yield ev("response.failed", {"response": fail})
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
                    yield ev("response.failed", {"response": fail})
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
                seq = 0

                def ev(typ: str, payload: dict[str, Any]) -> str:
                    nonlocal seq
                    data = dict(payload)
                    data["sequence_number"] = seq
                    seq += 1
                    return sseevt(typ, data)

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
                    yield ev("response.created", {"response": base})
                    yield ev("response.in_progress", {"response": base})
                    yield ev(
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
                            yield ev(
                                "response.output_text.delta",
                                {
                                    "response_id": rid,
                                    "output_index": 0,
                                    "item_id": mid,
                                    "content_index": 0,
                                    "delta": d,
                                    "logprobs": [],
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

                        yield ev(
                            "response.output_text.done",
                            {
                                "response_id": rid,
                                "output_index": 0,
                                "item_id": mid,
                                "content_index": 0,
                                "text": text,
                                "logprobs": [],
                            },
                        )
                        yield ev(
                            "response.output_item.done",
                            {"response_id": rid, "output_index": 0, "item": final["output"][0]},
                        )

                        if body.store:
                            request.app.state.store[rid] = final
                            request.app.state.items[rid] = items

                        yield ev("response.completed", {"response": final})
                        return

                    async for up in cdx.stream(call):
                        if up.delta:
                            parts.append(up.delta)
                            yield ev(
                                "response.output_text.delta",
                                {
                                    "response_id": rid,
                                    "output_index": 0,
                                    "item_id": mid,
                                    "content_index": 0,
                                    "delta": up.delta,
                                    "logprobs": [],
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

                            yield ev(
                                "response.output_text.done",
                                {
                                    "response_id": rid,
                                    "output_index": 0,
                                    "item_id": mid,
                                    "content_index": 0,
                                    "text": text,
                                    "logprobs": [],
                                },
                            )
                            yield ev(
                                "response.output_item.done",
                                {"response_id": rid, "output_index": 0, "item": final["output"][0]},
                            )

                            if body.store:
                                request.app.state.store[rid] = final
                                request.app.state.items[rid] = items

                            yield ev("response.completed", {"response": final})

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
                    yield ev("response.failed", {"response": fail})
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
                    yield ev("response.failed", {"response": fail})
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
                    yield ev("response.failed", {"response": fail})
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
            forcedsrv = tchoice.get("server_label") if isinstance(tchoice, dict) else None
            forcedmcp = tchoice.get("name") if isinstance(tchoice, dict) else None
            use: Usage | None = None

            try:
                mcp = topts.get("mcp") if isinstance(topts.get("mcp"), dict) else None
                if mcp and isinstance(mcp.get("servers"), dict):
                    for label, srv in sorted(mcp["servers"].items()):
                        if not isinstance(srv, dict):
                            continue
                        url = srv.get("url")
                        if not isinstance(url, str) or not url.strip():
                            continue

                        headers: dict[str, str] = {}
                        if isinstance(srv.get("headers"), dict):
                            headers.update({str(k): str(v) for k, v in srv["headers"].items()})
                        auth = srv.get("authorization")
                        if isinstance(auth, str) and auth.strip() and "authorization" not in {k.lower() for k in headers}:
                            val = auth.strip()
                            headers["Authorization"] = val if " " in val else f"Bearer {val}"

                        tid = f"mcp_tools_{uuid4().hex}"
                        try:
                            tools = await t.mcptools(
                                url,
                                headers=headers or None,
                                transport=srv.get("transport") if isinstance(srv.get("transport"), str) else None,
                            )
                            allow = srv.get("allowed_tools")
                            if isinstance(allow, list):
                                allowset = {str(x) for x in allow}
                                tools = [x for x in tools if isinstance(x, dict) and x.get("name") in allowset]
                            if isinstance(mcp.get("tools"), dict):
                                mcp["tools"][label] = tools
                            item = {"id": tid, "type": "mcp_list_tools", "server_label": label, "tools": tools}
                        except Exception as exc:
                            if isinstance(mcp.get("tools"), dict):
                                mcp["tools"][label] = []
                            item = {"id": tid, "type": "mcp_list_tools", "server_label": label, "tools": [], "error": str(exc)}
                        outs.append(item)

                for _ in range(8):
                    lead = actlead(topts, tchoice)
                    prompt, paths = await mkprompt(msgs2, system=sysm, imgs=imgs, lead=lead)
                    res: Result = await cdx.run(Call(model=body.model, prompt=prompt, imgs=paths, effort=effort))
                    use = useadd(use, res.usage)

                    typ, name, args, finaltxt = actparse(res.text)
                    if typ == "final":
                        if (tchoice == "required" or forced) and called == 0:
                            msgs2.append(Msg(role="user", content="Tool use is required. Call a tool before finishing."))
                            continue
                        break

                    if not name or name not in topts:
                        msgs2.append(Msg(role="user", content=f"Tool '{name or ''}' is not enabled."))
                        continue

                    if forced and called == 0 and name != forced:
                        msgs2.append(Msg(role="user", content=f"You must call {forced} tool first."))
                        continue

                    if forced == "mcp" and called == 0:
                        asl = (args or {}).get("server_label")
                        anm = (args or {}).get("name") or (args or {}).get("tool") or (args or {}).get("tool_name")
                        asl = asl.strip() if isinstance(asl, str) else None
                        anm = anm.strip() if isinstance(anm, str) else None
                        if forcedsrv and asl and asl != forcedsrv:
                            msgs2.append(Msg(role="user", content=f"You must call mcp server_label={forcedsrv} first."))
                            continue
                        if forcedmcp and anm and anm != forcedmcp:
                            msgs2.append(Msg(role="user", content=f"You must call mcp name={forcedmcp} first."))
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

    @app.post("/v1/responses/input_tokens")
    @app.post("/responses/input_tokens")
    async def inputtokens(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}

        model = body.get("model")
        model = model.strip() if isinstance(model, str) and model.strip() else MODELS[0]
        bad = chkmodel(model)
        if bad:
            return bad

        instr = body.get("instructions")
        instr = instr if isinstance(instr, str) and instr.strip() else None

        prev_id = body.get("previous_response_id")
        prev_id = prev_id.strip() if isinstance(prev_id, str) and prev_id.strip() else None
        prev = request.app.state.store.get(prev_id) if prev_id else None

        raw_input = body.get("input")
        if raw_input is None:
            raw_input = ""

        try:
            msgs, _items = norminput(raw_input)
        except ValueError as exc:
            return err(400, f"invalid input: {exc}", param="input", code="invalid_input")

        if prev and prev.get("output_text"):
            msgs.insert(0, Msg(role="assistant", content=prev["output_text"]))

        imgs = Tmpimgs()
        try:
            prompt, _paths = await mkprompt(msgs, system=_system(instr), imgs=imgs, lead=BASELEAD)
        except ImgErr as exc:
            return err(400, f"image error: {exc}", param="input", code="invalid_image")
        finally:
            imgs.close()

        # Naive count: good enough for client-side budgeting without a tokenizer dependency.
        count = len(prompt.split())
        return JSONResponse({"object": "response.input_tokens", "input_tokens": count})

    @app.post("/v1/responses/compact")
    @app.post("/responses/compact")
    async def compact(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}

        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            return err(400, "model is required", param="model", code="missing_required_parameter")
        model = model.strip()
        bad = chkmodel(model)
        if bad:
            return bad

        instr = body.get("instructions")
        instr = instr if isinstance(instr, str) and instr.strip() else None

        prev_id = body.get("previous_response_id")
        prev_id = prev_id.strip() if isinstance(prev_id, str) and prev_id.strip() else None
        prev = request.app.state.store.get(prev_id) if prev_id else None

        raw_input = body.get("input")
        if raw_input is None:
            raw_input = ""

        try:
            msgs, _items = norminput(raw_input)
        except ValueError as exc:
            return err(400, f"invalid input: {exc}", param="input", code="invalid_input")

        if prev and prev.get("output_text"):
            msgs.insert(0, Msg(role="assistant", content=prev["output_text"]))

        imgs = Tmpimgs()
        try:
            prompt, _paths = await mkprompt(msgs, system=_system(instr), imgs=imgs, lead=BASELEAD)
        except ImgErr as exc:
            return err(400, f"image error: {exc}", param="input", code="invalid_image")
        finally:
            imgs.close()

        raw = prompt.encode("utf-8", errors="replace")
        enc = base64.b64encode(raw).decode("ascii")
        now = int(time.time())

        item = {
            "id": f"comp_{uuid4().hex}",
            "type": "compaction",
            # Not real encryption, but schema-compatible for SDK use.
            "encrypted_content": enc,
            "created_by": "barter",
        }

        usage = {
            "input_tokens": len(prompt.split()),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": len(prompt.split()),
        }

        return JSONResponse(
            {
                "id": f"cmp_{uuid4().hex}",
                "object": "response.compaction",
                "created_at": now,
                "output": [item],
                "usage": usage,
            }
        )

    @app.get("/v1/responses/{rid}")
    @app.get("/responses/{rid}")
    async def getresp(rid: str, request: Request):
        got = request.app.state.store.get(rid)
        if got is None:
            return err(404, f"response '{rid}' not found", param="response_id", code="not_found")
        return JSONResponse(got)

    @app.get("/v1/responses/{rid}/input_items")
    @app.get("/responses/{rid}/input_items")
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
    @app.post("/responses/{rid}/cancel")
    async def cancel(rid: str, request: Request):
        if rid not in request.app.state.store:
            return err(404, f"response '{rid}' not found", param="response_id", code="not_found")
        return err(409, "cancel is not supported", code="not_supported")

    @app.delete("/v1/responses/{rid}")
    @app.delete("/responses/{rid}")
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
