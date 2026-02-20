from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path
from typing import Any

from app.codex import Codex
from app.config import Cfg, EFFORTS
from app.server import mkapp
from app.tools import Tools


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _mkparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="barter", description="OpenAI-compatible API proxy backed by Codex CLI")
    sub = p.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="Run the API server")
    serve.add_argument("--host", default=os.getenv("BARTER_HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=_env_int("BARTER_PORT", 8000))
    serve.add_argument("--reload", action="store_true", default=False)
    serve.add_argument("--log-level", default=os.getenv("BARTER_LOG_LEVEL", "info"))

    serve.add_argument("--admin-key", default=os.getenv("BARTER_ADMIN_KEY", ""))
    serve.add_argument("--db", dest="db_path", default=os.getenv("BARTER_DB", ""))
    serve.add_argument("--default-effort", choices=EFFORTS, default=os.getenv("BARTER_DEFAULT_EFFORT", "medium"))
    serve.add_argument(
        "--tool-call-max-bad-turns",
        type=int,
        default=_env_int("BARTER_TOOL_CALL_MAX_BAD_TURNS", 8),
        help="Max consecutive invalid tool decisions before the proxy stops tool-calling and answers.",
    )

    serve.add_argument("--codex-binary", default=os.getenv("CODEX_BINARY", "codex"))
    serve.add_argument("--codex-workdir", default=os.getenv("CODEX_WORKDIR", os.getcwd()))
    serve.add_argument("--codex-timeout-seconds", type=int, default=_env_int("CODEX_TIMEOUT_SECONDS", 900))

    g = serve.add_mutually_exclusive_group()
    g.add_argument("--codex-disable-shell-tool", dest="codex_disable_shell_tool", action="store_true")
    g.add_argument("--codex-enable-shell-tool", dest="codex_disable_shell_tool", action="store_false")
    serve.set_defaults(codex_disable_shell_tool=None)

    sub.add_parser("gen-admin-key", help="Print a random admin key (starts with brtr-)")
    return p


def _serve(args: Any) -> int:
    import uvicorn

    db_path: Path | None = None
    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()

    admin_key = (args.admin_key or "").strip()
    if admin_key and not admin_key.startswith("brtr-"):
        raise SystemExit("error: --admin-key must start with 'brtr-'")

    disable_shell_tool = args.codex_disable_shell_tool
    if disable_shell_tool is None:
        disable_shell_tool = _env_bool("BARTER_CODEX_DISABLE_SHELL_TOOL", False)

    cfg = Cfg(
        bin=str(args.codex_binary),
        cwd=Path(args.codex_workdir).expanduser().resolve(),
        timeout=max(1, int(args.codex_timeout_seconds)),
    )

    codex = Codex(
        bin=cfg.bin,
        cwd=cfg.cwd,
        timeout=cfg.timeout,
        disable_shell_tool=disable_shell_tool,
    )

    app = mkapp(
        cfg=cfg,
        codex=codex,
        tools=Tools(),
        db_path=db_path,
        admin_key=admin_key or None,
        default_effort=str(args.default_effort),
        max_bad_tool_turns=max(0, int(args.tool_call_max_bad_turns)),
    )

    uvicorn.run(
        app,
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
        log_level=str(args.log_level),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = _mkparser()
    args = p.parse_args(argv)

    if not args.cmd:
        args = p.parse_args((argv or []) + ["serve"])

    if args.cmd == "serve":
        return _serve(args)

    if args.cmd == "gen-admin-key":
        print("brtr-" + secrets.token_urlsafe(32))
        return 0

    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
