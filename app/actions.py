from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

from app.codex import Usage
from app.schema import Msg
from app.tools import Tools


def canon(typ: str) -> str | None:
    t = (typ or "").strip()
    if not t:
        return None
    if t == "mcp":
        return "mcp"
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

        name = canon(typ)
        if name is None:
            raise ValueError(f"unsupported tool type '{typ}'")

        if name == "web_search":
            defs.append(t)
            allow = None
            filt = t.get("filters")
            if isinstance(filt, dict) and isinstance(filt.get("allowed_domains"), list):
                allow = [d for d in filt.get("allowed_domains") if isinstance(d, str)]
            enabled[name] = {"allow": allow}
            continue

        if name == "mcp":
            label = t.get("server_label")
            url = t.get("server_url")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("mcp tool requires server_label")
            if not isinstance(url, str) or not url.strip():
                raise ValueError("mcp tool requires server_url")

            label = label.strip()
            url = url.strip()

            auth = t.get("authorization") if isinstance(t.get("authorization"), str) and t.get("authorization").strip() else None
            hdrs = t.get("headers") if isinstance(t.get("headers"), dict) else None
            headers: dict[str, str] | None = None
            if hdrs:
                headers = {str(k): str(v) for k, v in hdrs.items() if isinstance(k, str) and isinstance(v, str)}

            allow = t.get("allowed_tools") if isinstance(t.get("allowed_tools"), list) else None
            allowed = [x for x in (allow or []) if isinstance(x, str) and x.strip()] or None

            appr = t.get("require_approval")
            if appr is None:
                appr = "never"
            if isinstance(appr, str):
                appr = appr.strip().lower()
                if appr not in {"never"}:
                    raise ValueError("mcp require_approval only supports 'never'")
            else:
                raise ValueError("mcp require_approval must be a string")

            desc = t.get("server_description") if isinstance(t.get("server_description"), str) else None
            transport = t.get("transport") if isinstance(t.get("transport"), str) and t.get("transport").strip() else None

            cfg = enabled.setdefault("mcp", {"servers": {}, "tools": {}})
            if not isinstance(cfg.get("servers"), dict):
                cfg["servers"] = {}
            if not isinstance(cfg.get("tools"), dict):
                cfg["tools"] = {}

            existing = cfg["servers"].get(label)
            if existing and existing.get("url") != url:
                raise ValueError(f"mcp server_label '{label}' already registered with a different server_url")

            cfg["servers"][label] = {
                "label": label,
                "url": url,
                "authorization": auth,
                "headers": headers,
                "allowed_tools": allowed,
                "server_description": desc,
                "transport": transport,
            }

            safe = dict(t)
            safe.pop("authorization", None)
            safe.pop("headers", None)
            defs.append(safe)
            continue

        defs.append(t)
        enabled[name] = {}

    return defs, enabled


def toolchoice(raw: Any | None, enabled: dict[str, dict[str, Any]]) -> Any:
    if raw is None:
        return "auto" if enabled else "none"

    if isinstance(raw, str):
        s = raw.strip()
        if s in {"none", "auto", "required"}:
            return s
        name = canon(s)
        if name and name in enabled:
            return {"type": name}
        raise ValueError("tool_choice must be 'none', 'auto', 'required', or an object with type")

    if isinstance(raw, dict):
        typ = raw.get("type")
        if isinstance(typ, str):
            name = canon(typ)
            if name and name in enabled:
                if name != "mcp":
                    return {"type": name}

                sl = raw.get("server_label")
                nm = raw.get("name") or raw.get("tool_name") or raw.get("tool")
                if isinstance(raw.get("mcp"), dict):
                    sl = sl or raw["mcp"].get("server_label")
                    nm = nm or raw["mcp"].get("name") or raw["mcp"].get("tool_name")

                sl = sl.strip() if isinstance(sl, str) else None
                nm = nm.strip() if isinstance(nm, str) else None

                out: dict[str, Any] = {"type": "mcp"}
                if sl:
                    mcp = enabled.get("mcp") if isinstance(enabled.get("mcp"), dict) else None
                    servers = mcp.get("servers") if isinstance(mcp, dict) else None
                    if not isinstance(servers, dict) or sl not in servers:
                        raise ValueError(f"tool_choice mcp server_label '{sl}' is not enabled")
                    out["server_label"] = sl
                if nm:
                    out["name"] = nm
                return out
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
    forcedmcp: str | None = None
    forcedsrv: str | None = None
    if isinstance(choice, dict) and isinstance(choice.get("type"), str):
        forced = choice["type"]
        if forced == "mcp":
            forcedsrv = choice.get("server_label") if isinstance(choice.get("server_label"), str) else None
            forcedmcp = choice.get("name") if isinstance(choice.get("name"), str) else None
    if choice == "required":
        forced = names or ""

    must = f" You must call {forced}." if forced else ""
    if forced == "mcp" and (forcedsrv or forcedmcp):
        must = " You must call mcp"
        if forcedsrv:
            must += f" server_label={forcedsrv}"
        if forcedmcp:
            must += f" name={forcedmcp}"
        must += "."

    mcpinfo = ""
    mcp = enabled.get("mcp") if isinstance(enabled.get("mcp"), dict) else None
    if mcp and isinstance(mcp.get("servers"), dict) and isinstance(mcp.get("tools"), dict):
        lines: list[str] = []
        for label, cfg in sorted(mcp["servers"].items()):
            if not isinstance(cfg, dict):
                continue
            url = cfg.get("url")
            url = url if isinstance(url, str) else ""
            lines.append(f"- {label}: {url}".strip())
            tlist = mcp["tools"].get(label)
            if isinstance(tlist, list) and tlist:
                for tool in tlist[:50]:
                    if not isinstance(tool, dict):
                        continue
                    nm = tool.get("name")
                    desc = tool.get("description") or ""
                    if isinstance(nm, str) and nm.strip():
                        descs = str(desc)[:200].replace("\n", " ")
                        lines.append(f"  * {nm.strip()}: {descs}".rstrip())
        if lines:
            mcpinfo = "MCP servers and tools:\n" + "\n".join(lines) + "\n"

    return (
        "TOOL MODE. Decide the next step for the assistant.\n"
        f"Allowed tools: {names or 'none'}.{must}\n"
        f"{mcpinfo}"
        "Output exactly one JSON object and nothing else.\n"
        "To call a tool:\n"
        '  {"type":"tool","name":"web_search","arguments":{"query":"...","max_results":5}}\n'
        '  {"type":"tool","name":"code_interpreter","arguments":{"code":"..."}}\n'
        '  {"type":"tool","name":"mcp","arguments":{"server_label":"...","name":"...","arguments":{}}}\n'
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
        name = canon(str(name or "").strip()) or str(name or "").strip()
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
                "action": {"type": "search", "query": q, "queries": [q]},
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
                "action": {"type": "search", "query": q, "queries": [q]},
                "query": q,
                "error": str(exc),
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

    if name == "mcp":
        tid = tid or f"mcp_{uuid4().hex}"
        sl = args.get("server_label") or args.get("server") or args.get("label")
        sl = sl.strip() if isinstance(sl, str) and sl.strip() else None
        nm = args.get("name") or args.get("tool") or args.get("tool_name")
        nm = nm.strip() if isinstance(nm, str) and nm.strip() else None

        targs = args.get("arguments") or args.get("args") or {}
        if isinstance(targs, str):
            try:
                targs = json.loads(targs)
            except json.JSONDecodeError:
                targs = {}
        if targs is None:
            targs = {}
        if not isinstance(targs, dict):
            targs = {}

        mcp = opts.get("mcp") if isinstance(opts.get("mcp"), dict) else None
        srv = mcp.get("servers", {}).get(sl) if mcp and sl else None
        if not isinstance(srv, dict):
            item = {
                "id": tid,
                "type": "mcp_call",
                "server_label": sl or "",
                "name": nm or "",
                "arguments": json.dumps(targs, ensure_ascii=True, separators=(",", ":")),
                "output": "",
                "status": "failed",
                "error": "unknown mcp server_label",
            }
            msg = toolmsg("mcp", {"error": "unknown mcp server_label", "server_label": sl, "name": nm})
            return item, msg

        allowed = srv.get("allowed_tools")
        if isinstance(allowed, list) and nm and nm not in allowed:
            item = {
                "id": tid,
                "type": "mcp_call",
                "server_label": sl or "",
                "name": nm or "",
                "arguments": json.dumps(targs, ensure_ascii=True, separators=(",", ":")),
                "output": "",
                "status": "failed",
                "error": f"mcp tool '{nm}' is not allowed",
            }
            msg = toolmsg("mcp", {"error": f"tool '{nm}' is not allowed", "server_label": sl, "name": nm})
            return item, msg

        url = srv.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("mcp server url missing")

        headers: dict[str, str] = {}
        if isinstance(srv.get("headers"), dict):
            headers.update({str(k): str(v) for k, v in srv["headers"].items()})
        auth = srv.get("authorization")
        if isinstance(auth, str) and auth.strip() and "authorization" not in {k.lower() for k in headers}:
            val = auth.strip()
            headers["Authorization"] = val if " " in val else f"Bearer {val}"

        run = await t.mcpcall(
            url,
            nm or "",
            arguments=targs,
            headers=headers or None,
            transport=srv.get("transport") if isinstance(srv.get("transport"), str) else None,
        )
        out = str(run.get("output") or "")
        item = {
            "id": tid,
            "type": "mcp_call",
            "server_label": sl or "",
            "name": nm or "",
            "arguments": json.dumps(targs, ensure_ascii=True, separators=(",", ":")),
            "output": out,
            "status": "failed" if run.get("is_error") else "completed",
        }
        if run.get("is_error"):
            item["error"] = out or "mcp tool error"
        msg = toolmsg(
            "mcp",
            {
                "server_label": sl,
                "name": nm,
                "arguments": targs,
                "output": out,
                "is_error": bool(run.get("is_error")),
            },
        )
        return item, msg

    raise ValueError(f"unsupported tool '{name}'")

