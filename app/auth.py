from __future__ import annotations

from fastapi import Request


def bearer(req: Request) -> str | None:
    h = req.headers.get("authorization") or ""
    if h.lower().startswith("bearer "):
        val = h[7:].strip()
        return val or None

    return None

