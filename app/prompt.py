from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from app.schema import ImageURL, Msg, Part
from app.images import Tmpimgs


async def mkprompt(
    msgs: list[Msg], *, system: str, imgs: Tmpimgs, lead: str | None = None
) -> tuple[str, list[Path]]:
    lead = lead or "Use the conversation below and respond to the latest user request."
    lines: list[str] = [system.strip(), "", lead, ""]

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

