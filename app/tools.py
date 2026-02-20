from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class PySandbox:
    """
    Process-level limits for the local Python code interpreter tool.

    Notes:
    - This is not a full VM/container boundary.
    - Limits are enforced in the child process before user code runs.
    """

    max_memory_mb: int = 256
    max_cpu_seconds: int = 10
    max_file_bytes: int = 10 * 1024 * 1024
    max_open_files: int = 64
    max_processes: int = 32
    max_output_bytes: int = 100_000

    @classmethod
    def env(cls) -> "PySandbox":
        return cls(
            max_memory_mb=_env_int("BARTER_CODE_SANDBOX_MAX_MEMORY_MB", 256, low=64, high=8192),
            max_cpu_seconds=_env_int("BARTER_CODE_SANDBOX_MAX_CPU_SECONDS", 10, low=1, high=300),
            max_file_bytes=_env_int("BARTER_CODE_SANDBOX_MAX_FILE_BYTES", 10 * 1024 * 1024, low=1024, high=1024 * 1024 * 1024),
            max_open_files=_env_int("BARTER_CODE_SANDBOX_MAX_OPEN_FILES", 64, low=16, high=1024),
            max_processes=_env_int("BARTER_CODE_SANDBOX_MAX_PROCESSES", 32, low=1, high=4096),
            max_output_bytes=_env_int("BARTER_CODE_SANDBOX_MAX_OUTPUT_BYTES", 100_000, low=1024, high=10_000_000),
        )


_RUNNER = """\
from __future__ import annotations

import builtins
import os
import resource
import subprocess
import sys


def _set(name: str, value: int) -> None:
    if value <= 0:
        return
    lim = getattr(resource, name, None)
    if lim is None:
        return
    try:
        resource.setrlimit(lim, (value, value))
    except Exception:
        # Platforms differ; best-effort limits are better than failing closed here.
        pass


_WRITE_FLAGS = 0
for _name in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND", "O_EXCL", "O_TMPFILE"):
    _WRITE_FLAGS |= int(getattr(os, _name, 0))


def _deny_fs(op: str) -> None:
    raise PermissionError(f"[barter sandbox] read-only filesystem: blocked {op}")


def _deny_proc(op: str) -> None:
    raise PermissionError(f"[barter sandbox] subprocess disabled: blocked {op}")


def _is_write_mode(mode: str) -> bool:
    return any(ch in mode for ch in ("w", "a", "x", "+"))


def _deny_fs_fn(name: str):
    def _blocked(*_args, **_kwargs):
        _deny_fs(name)

    return _blocked


def _deny_proc_fn(name: str):
    def _blocked(*_args, **_kwargs):
        _deny_proc(name)

    return _blocked


_ORIG_OPEN = builtins.open
_ORIG_OS_OPEN = os.open


def _open_ro(file, mode="r", *args, **kwargs):
    m = str(mode or "r")
    if _is_write_mode(m):
        _deny_fs(f"open({m})")
    return _ORIG_OPEN(file, mode, *args, **kwargs)


def _os_open_ro(path, flags, mode=0o777, *, dir_fd=None):
    try:
        f = int(flags)
    except Exception:
        f = 0
    if f & _WRITE_FLAGS:
        _deny_fs("os.open(write)")
    if dir_fd is None:
        return _ORIG_OS_OPEN(path, f, mode)
    return _ORIG_OS_OPEN(path, f, mode, dir_fd=dir_fd)


def _apply_read_only() -> None:
    builtins.open = _open_ro
    os.open = _os_open_ro

    for name in (
        "remove",
        "unlink",
        "rename",
        "replace",
        "rmdir",
        "removedirs",
        "mkdir",
        "makedirs",
        "chmod",
        "lchmod",
        "chown",
        "lchown",
        "chflags",
        "lchflags",
        "utime",
        "truncate",
        "ftruncate",
        "symlink",
        "link",
        "mknod",
        "mkfifo",
        "setxattr",
        "removexattr",
        "fsetxattr",
        "fremovexattr",
    ):
        if hasattr(os, name):
            setattr(os, name, _deny_fs_fn(f"os.{name}"))

    for name in ("system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp"):
        if hasattr(os, name):
            setattr(os, name, _deny_proc_fn(f"os.{name}"))

    subprocess.Popen = _deny_proc_fn("subprocess.Popen")

    blocked_events = {
        "os.remove",
        "os.unlink",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.mkdir",
        "os.chmod",
        "os.chown",
        "os.utime",
        "os.symlink",
        "os.link",
        "os.truncate",
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.posix_spawnp",
    }

    def _audit(event, args):
        if event == "open":
            mode = ""
            flags = 0
            if len(args) >= 2:
                arg = args[1]
                if isinstance(arg, str):
                    mode = arg
                else:
                    try:
                        flags = int(arg)
                    except Exception:
                        flags = 0
            if _is_write_mode(mode) or (flags & _WRITE_FLAGS):
                _deny_fs("open(write)")
            return

        if event in blocked_events:
            if event.startswith("os."):
                _deny_fs(event)
            _deny_proc(event)

        if event.startswith("os.exec") or event.startswith("os.spawn"):
            _deny_proc(event)

    sys.addaudithook(_audit)


code_path = sys.argv[1]
cpu = int(sys.argv[2])
mem = int(sys.argv[3])
fsize = int(sys.argv[4])
nofile = int(sys.argv[5])
nproc = int(sys.argv[6])

_set("RLIMIT_CORE", 0)
_set("RLIMIT_CPU", cpu)
_set("RLIMIT_AS", mem)
_set("RLIMIT_DATA", mem)
_set("RLIMIT_FSIZE", fsize)
_set("RLIMIT_NOFILE", nofile)
_set("RLIMIT_NPROC", nproc)
_apply_read_only()

g = {"__name__": "__main__", "__file__": code_path, "__package__": None}
with open(code_path, "rb") as fh:
    source = fh.read()
exec(compile(source, code_path, "exec"), g)
"""


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    if val < low:
        return low
    if val > high:
        return high
    return val


async def _read_cap(stream: asyncio.StreamReader, cap: int) -> tuple[bytes, bool]:
    out = bytearray()
    trunc = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        if len(out) + len(chunk) > cap:
            trunc = True
        if len(out) < cap:
            left = cap - len(out)
            out.extend(chunk[:left])
    return bytes(out), trunc


async def _watch_mem(proc: asyncio.subprocess.Process, limit: int) -> tuple[bool, int]:
    """
    Best-effort RSS watchdog. Uses psutil if available; otherwise no-op.
    Returns (limit_exceeded, peak_rss_bytes).
    """
    if limit <= 0:
        return False, 0
    try:
        import psutil  # type: ignore
    except Exception:
        return False, 0

    peak = 0
    try:
        root = psutil.Process(proc.pid)
    except Exception:
        return False, 0

    while proc.returncode is None:
        total = 0
        plist: list[Any] = []
        try:
            plist = [root, *root.children(recursive=True)]
        except Exception:
            plist = [root]

        for p in plist:
            try:
                total += int(p.memory_info().rss)
            except Exception:
                continue

        if total > peak:
            peak = total
        if total > limit:
            proc.kill()
            return True, peak
        await asyncio.sleep(0.02)

    return False, peak


class Tools:
    async def py(self, code: str, *, timeout: int = 10) -> dict[str, Any]:
        c = code if isinstance(code, str) else ""
        t = max(1, min(int(timeout or 10), 120))
        sb = PySandbox.env()
        cap = sb.max_output_bytes
        mem_bytes = int(sb.max_memory_mb) * 1024 * 1024
        cpu_seconds = min(int(sb.max_cpu_seconds), t)

        tmp = tempfile.TemporaryDirectory(prefix="codex-proxy-ci-")
        try:
            tmpdir = Path(tmp.name)
            code_path = tmpdir / "main.py"
            run_path = tmpdir / "_sandbox_runner.py"
            code_path.write_text(c, encoding="utf-8")
            run_path.write_text(_RUNNER, encoding="utf-8")

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(run_path),
                str(code_path),
                str(cpu_seconds),
                str(mem_bytes),
                str(sb.max_file_bytes),
                str(sb.max_open_files),
                str(sb.max_processes),
                cwd=tmp.name,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HOME": tmp.name,
                    "TMPDIR": tmp.name,
                    "PATH": os.getenv("PATH", ""),
                },
            )

            assert proc.stdout is not None
            assert proc.stderr is not None
            tout = asyncio.create_task(_read_cap(proc.stdout, cap))
            terr = asyncio.create_task(_read_cap(proc.stderr, cap))
            tmem = asyncio.create_task(_watch_mem(proc, mem_bytes))

            try:
                await asyncio.wait_for(proc.wait(), timeout=t)
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                timed_out = True

            memkilled = False
            peakrss = 0
            if tmem.done():
                try:
                    memkilled, peakrss = await tmem
                except asyncio.CancelledError:
                    memkilled, peakrss = False, 0
            else:
                try:
                    memkilled, peakrss = await asyncio.wait_for(tmem, timeout=0.05)
                except asyncio.TimeoutError:
                    tmem.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await tmem
                except asyncio.CancelledError:
                    memkilled, peakrss = False, 0

            outb, outtrunc = await tout
            errb, errtrunc = await terr
            stderr = errb.decode("utf-8", errors="replace")
            if outtrunc:
                stderr = f"{stderr}\n[barter sandbox] stdout truncated at {cap} bytes".strip()
            if errtrunc:
                stderr = f"{stderr}\n[barter sandbox] stderr truncated at {cap} bytes".strip()
            if memkilled:
                stderr = f"{stderr}\n[barter sandbox] memory limit exceeded ({sb.max_memory_mb} MB)".strip()

            return {
                "ok": (not timed_out) and (not memkilled) and proc.returncode == 0,
                "timeout": timed_out,
                "exit_code": None if timed_out else proc.returncode,
                "stdout": outb.decode("utf-8", errors="replace")[:cap],
                "stderr": stderr[:cap],
                "sandboxed": True,
                "read_only_fs": True,
                "memory_killed": memkilled,
                "peak_rss_bytes": peakrss,
                "limits": {
                    "max_memory_mb": sb.max_memory_mb,
                    "max_cpu_seconds": cpu_seconds,
                    "max_file_bytes": sb.max_file_bytes,
                    "max_open_files": sb.max_open_files,
                    "max_processes": sb.max_processes,
                    "max_output_bytes": sb.max_output_bytes,
                },
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
