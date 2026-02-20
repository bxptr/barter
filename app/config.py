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

MODEL_ALIASES = {
    "fast": "gpt-5.3-codex-spark",
}

EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]

SYSTEM = "You are a general-purpose, helpful AI assistant"

_EFFRANK = {e: i for i, e in enumerate(EFFORTS)}


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


@dataclass(frozen=True)
class ToolCallLimits:
    """
    Limits for tool calls in the /v1/responses tool loop (web_search, code_interpreter, mcp).

    Notes:
    - Limits are enforced by the proxy (barter), not by the Codex CLI.
    - `None` means unlimited.
    """

    cutoff_effort: str = "medium"
    max_calls_at_or_below_cutoff: int | None = 5
    max_calls_above_cutoff: int | None = None
    max_bad_tool_turns: int = 8

    @classmethod
    def env(cls) -> "ToolCallLimits":
        cutoff = (os.getenv("BARTER_TOOL_CALL_CUTOFF_EFFORT", "medium") or "").strip().lower()
        if cutoff not in EFFORTS:
            cutoff = "medium"

        def _int_or_none(key: str, default: int | None) -> int | None:
            raw = (os.getenv(key, "") or "").strip().lower()
            if not raw:
                return default
            if raw in {"none", "null", "unlimited", "inf", "infinite"}:
                return None
            try:
                v = int(raw)
            except ValueError:
                return default
            return v

        low = _int_or_none("BARTER_TOOL_CALL_LIMIT", 5)
        high = _int_or_none("BARTER_TOOL_CALL_LIMIT_ABOVE_CUTOFF", None)
        bad = _int_or_none("BARTER_TOOL_CALL_MAX_BAD_TURNS", 8)
        if bad is None:
            bad = 8

        # Clamp negatives to zero to avoid weird loops.
        if isinstance(low, int) and low < 0:
            low = 0
        if isinstance(high, int) and high < 0:
            high = 0
        if isinstance(bad, int) and bad < 0:
            bad = 0

        return cls(
            cutoff_effort=cutoff,
            max_calls_at_or_below_cutoff=low,
            max_calls_above_cutoff=high,
            max_bad_tool_turns=int(bad),
        )

    def _rank(self, eff: str | None) -> int:
        e = (eff or "none").strip().lower()
        return _EFFRANK.get(e, _EFFRANK["none"])

    def max_calls_for(self, eff: str | None) -> int | None:
        cutoff = self._rank(self.cutoff_effort)
        cur = self._rank(eff)
        return self.max_calls_above_cutoff if cur > cutoff else self.max_calls_at_or_below_cutoff


def defeff(raw: str | None, *, default: str | None = None) -> str | None:
    """
    Codex CLI can pick up a default reasoning effort from user config; in practice
    this can break requests when the default is unsupported by the chosen model.
    We default to a safe, OpenAI-SDK-compatible effort unless the client specifies
    one explicitly.
    """
    if raw is not None:
        return raw

    val = default if default is not None else os.getenv("BARTER_DEFAULT_EFFORT", "medium")
    val = val.strip().lower()
    if not val or val == "none":
        return None
    return val


def modelalias(raw: str) -> str:
    val = raw.strip()
    return MODEL_ALIASES.get(val, val)
