from __future__ import annotations

from app.config import Cfg, MODELS
from app.server import mkapp

# uvicorn entrypoint: `uvicorn app.main:app`
app = mkapp()

__all__ = ["app", "mkapp", "Cfg", "MODELS"]

