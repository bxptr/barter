from __future__ import annotations

from typing import Any

from app.codex import Usage


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
    max_tool_calls: int | None = None,
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
        "max_tool_calls": max_tool_calls,
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
