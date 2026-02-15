from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

MODELS = [
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.1-codex-mini",
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


def defeff(raw: str | None) -> str | None:
    """
    Codex CLI can pick up a default reasoning effort from user config; in practice
    this can break requests when the default is unsupported by the chosen model.
    We default to a safe, OpenAI-SDK-compatible effort unless the client specifies
    one explicitly.
    """
    if raw is not None:
        return raw

    val = os.getenv("BARTER_DEFAULT_EFFORT", "medium")
    val = val.strip().lower()
    if not val or val == "none":
        return None
    return val

