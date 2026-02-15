from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


def errpayload(
    msg: str,
    *,
    type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {"error": {"message": msg, "type": type, "param": param, "code": code}}


def err(
    status: int,
    msg: str,
    *,
    type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(status_code=status, content=errpayload(msg, type=type, param=param, code=code))


def autherr() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content=errpayload(
            "missing or invalid api key",
            type="authentication_error",
            code="invalid_api_key",
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )

