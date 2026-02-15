from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from app.actions import useadd
from app.codex import Call, Result, Usage
from app.images import Tmpimgs
from app.prompt import mkprompt
from app.schema import Msg


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

