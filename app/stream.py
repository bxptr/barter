from __future__ import annotations

import json
from typing import Any


def ssedata(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"


def sseevt(typ: str, payload: dict[str, Any]) -> str:
    data = dict(payload)
    data["type"] = typ
    return f"event: {typ}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"

