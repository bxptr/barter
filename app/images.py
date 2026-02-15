from __future__ import annotations

import base64
import binascii
import mimetypes
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx


class ImgErr(ValueError):
    pass


DATAURL = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE)
MAXIMG = 20 * 1024 * 1024


class Tmpimgs:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="codex-proxy-img-")
        self._cache: dict[str, Path] = {}

    @property
    def dir(self) -> Path:
        return Path(self._tmp.name)

    def close(self) -> None:
        self._tmp.cleanup()

    async def get(self, url: str) -> Path:
        cached = self._cache.get(url)
        if cached is not None:
            return cached

        p = urlparse(url)

        if url.startswith("data:"):
            path = self._write(url)
            self._cache[url] = path
            return path

        if p.scheme in {"http", "https"}:
            path = await self._dl(url)
            self._cache[url] = path
            return path

        if p.scheme == "file":
            path = Path(unquote(p.path)).expanduser().resolve()
            if not path.exists():
                raise ImgErr(f"missing local file: {path}")
            self._cache[url] = path
            return path

        path = Path(url).expanduser().resolve()
        if path.exists():
            self._cache[url] = path
            return path

        raise ImgErr("unsupported image url")

    def _write(self, url: str) -> Path:
        m = DATAURL.match(url)
        if not m:
            raise ImgErr("invalid data url")

        mime = m.group("mime")
        suf = mimetypes.guess_extension(mime) or ".img"

        try:
            raw = base64.b64decode(m.group("data"), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImgErr("invalid base64") from exc

        if len(raw) > MAXIMG:
            raise ImgErr("image too large")

        path = self.dir / f"img-{uuid4().hex}{suf}"
        path.write_bytes(raw)
        return path

    async def _dl(self, url: str) -> Path:
        timeout = httpx.Timeout(20.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        raw = resp.content
        if len(raw) > MAXIMG:
            raise ImgErr("image too large")

        ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        suf = mimetypes.guess_extension(ct) or Path(urlparse(url).path).suffix or ".img"

        path = self.dir / f"img-{uuid4().hex}{suf}"
        path.write_bytes(raw)
        return path

