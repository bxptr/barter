from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any, AsyncIterator


@dataclass
class Usage:
    prompt: int | None = None
    completion: int | None = None
    total: int | None = None


@dataclass
class Call:
    model: str
    prompt: str
    imgs: list[Path]
    effort: str | None = None


@dataclass
class Result:
    text: str
    usage: Usage | None = None


@dataclass
class Update:
    delta: str | None = None
    usage: Usage | None = None
    done: bool = False


class CodexError(RuntimeError):
    pass


class Codex:
    def __init__(
        self,
        *,
        bin: str = "codex",
        cwd: Path,
        timeout: int = 900,
    ) -> None:
        self.bin = bin
        self.cwd = cwd
        self.timeout = timeout
        self._flags = _flags(bin)

    async def run(self, call: Call) -> Result:
        parts: list[str] = []
        usage: Usage | None = None
        async for up in self.stream(call):
            if up.delta:
                parts.append(up.delta)
            if up.usage:
                usage = up.usage
        return Result(text="".join(parts), usage=usage)

    async def stream(self, call: Call) -> AsyncIterator[Update]:
        cmd = self._cmd(call)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        err_chunks: list[str] = []

        async def readerr() -> None:
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    return
                err_chunks.append(chunk.decode("utf-8", errors="replace"))

        err_task = asyncio.create_task(readerr())

        proc.stdin.write(call.prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        usage: Usage | None = None
        last: dict[str, str] = {}

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            evt = _json(line)
            if evt is None:
                continue

            got = _usage(evt)
            if got is not None:
                usage = got

            delta = _delta(evt, last)
            if delta:
                yield Update(delta=delta)

        try:
            await asyncio.wait_for(proc.wait(), timeout=max(self.timeout, 1))
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            await err_task
            raise CodexError("codex timed out") from exc

        await err_task

        if proc.returncode != 0:
            err = "".join(err_chunks).strip()
            raise CodexError(err or f"codex exited with {proc.returncode}")

        yield Update(done=True, usage=usage)

    def _cmd(self, call: Call) -> list[str]:
        cmd = [
            self.bin,
            "exec",
            "--json",
            "--model",
            call.model,
            "--sandbox",
            "read-only",
        ]

        if "--cd" in self._flags:
            cmd.extend(["--cd", str(self.cwd)])
        if "--skip-git-repo-check" in self._flags:
            cmd.append("--skip-git-repo-check")
        if "--ephemeral" in self._flags:
            cmd.append("--ephemeral")

        if call.effort:
            cmd.extend(["-c", f'model_reasoning_effort="{call.effort}"'])

        if call.imgs and "--image" in self._flags:
            for path in call.imgs:
                cmd.extend(["--image", str(path)])

        cmd.append("-")
        return cmd


def _flags(bin: str) -> set[str]:
    try:
        out = subprocess.run(
            [bin, "exec", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()

    text = f"{out.stdout}\n{out.stderr}"
    return set(re.findall(r"--[a-z0-9-]+", text))


def _json(line: str) -> dict[str, Any] | None:
    try:
        val = json.loads(line)
    except json.JSONDecodeError:
        return None
    return val if isinstance(val, dict) else None


def _delta(evt: dict[str, Any], last: dict[str, str]) -> str | None:
    item = evt.get("item")
    if not isinstance(item, dict):
        return _delta_field(evt)

    if item.get("type") != "agent_message":
        return None

    itemid = str(item.get("id") or "agent_message")

    explicit = _delta_field(item)
    if explicit:
        return explicit

    text = _text(item)
    if text is None:
        return None

    prev = last.get(itemid, "")
    last[itemid] = text

    if text.startswith(prev):
        d = text[len(prev) :]
        return d or None

    return text


def _text(item: dict[str, Any]) -> str | None:
    if isinstance(item.get("text"), str):
        return item["text"]

    content = item.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for ent in content:
            if not isinstance(ent, dict):
                continue
            if isinstance(ent.get("text"), str):
                parts.append(ent["text"])
            elif isinstance(ent.get("content"), str):
                parts.append(ent["content"])
        if parts:
            return "".join(parts)

    return None


def _delta_field(payload: dict[str, Any]) -> str | None:
    delta = payload.get("delta")
    if isinstance(delta, str):
        return delta

    if isinstance(delta, dict):
        if isinstance(delta.get("text"), str):
            return delta["text"]
        if isinstance(delta.get("content"), str):
            return delta["content"]

    if isinstance(payload.get("text_delta"), str):
        return payload["text_delta"]

    return None


def _usage(evt: dict[str, Any]) -> Usage | None:
    u = evt.get("usage")
    if not isinstance(u, dict):
        return None

    prompt = _int(u.get("input_tokens"))
    if prompt is None:
        prompt = _int(u.get("prompt_tokens"))

    comp = _int(u.get("output_tokens"))
    if comp is None:
        comp = _int(u.get("completion_tokens"))

    total = _int(u.get("total_tokens"))
    if total is None and prompt is not None and comp is not None:
        total = prompt + comp

    if prompt is None and comp is None and total is None:
        return None

    return Usage(prompt=prompt, completion=comp, total=total)


def _int(v: Any) -> int | None:
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return None
