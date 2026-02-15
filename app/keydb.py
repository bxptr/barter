from __future__ import annotations

import hashlib
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Any
from uuid import uuid4


class Keydb:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._init()
        try:
            self.path.chmod(0o600)
        except Exception:
            pass

    def _init(self) -> None:
        cur = self.con.cursor()
        cur.execute(
            """
            create table if not exists keys(
                id text primary key,
                name text,
                hash text not null unique,
                admin integer not null default 0,
                created_at integer not null,
                last_used_at integer,
                revoked integer not null default 0,
                revoked_at integer
            )
            """
        )
        self.con.commit()

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    def _hash(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def add(self, *, name: str | None = None, admin: bool = False) -> dict[str, Any]:
        now = int(time.time())
        kid = f"key_{uuid4().hex}"
        secret = "brtr-" + secrets.token_urlsafe(32)
        h = self._hash(secret)

        cur = self.con.cursor()
        cur.execute(
            "insert into keys(id,name,hash,admin,created_at) values (?,?,?,?,?)",
            (kid, name, h, 1 if admin else 0, now),
        )
        self.con.commit()

        return {
            "id": kid,
            "object": "api_key",
            "created_at": now,
            "name": name,
            "admin": admin,
            "revoked": False,
            "key": secret,
        }

    def check(self, key: str) -> dict[str, Any] | None:
        if not (key or "").startswith("brtr-"):
            return None

        h = self._hash(key)
        cur = self.con.cursor()
        row = cur.execute(
            "select id,admin,revoked from keys where hash=?",
            (h,),
        ).fetchone()

        if row is None or int(row["revoked"] or 0) != 0:
            return None

        now = int(time.time())
        cur.execute("update keys set last_used_at=? where id=?", (now, str(row["id"])))
        self.con.commit()

        return {"id": str(row["id"]), "admin": bool(row["admin"])}

    def list(self) -> list[dict[str, Any]]:
        cur = self.con.cursor()
        rows = cur.execute(
            "select id,name,admin,created_at,last_used_at,revoked,revoked_at from keys order by created_at desc"
        ).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": str(r["id"]),
                    "object": "api_key",
                    "created_at": int(r["created_at"]),
                    "name": r["name"],
                    "admin": bool(r["admin"]),
                    "last_used_at": int(r["last_used_at"]) if r["last_used_at"] is not None else None,
                    "revoked": bool(r["revoked"]),
                    "revoked_at": int(r["revoked_at"]) if r["revoked_at"] is not None else None,
                }
            )

        return out

    def revoke(self, kid: str) -> bool:
        now = int(time.time())
        cur = self.con.cursor()
        res = cur.execute(
            "update keys set revoked=1,revoked_at=? where id=? and revoked=0",
            (now, kid),
        )
        self.con.commit()
        return bool(res.rowcount)

