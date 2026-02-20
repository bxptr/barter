from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class ImageURL(BaseModel):
    url: str
    detail: str | None = None

    model_config = ConfigDict(extra="allow")


class Part(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None

    model_config = ConfigDict(extra="allow")


class Msg(BaseModel):
    role: str
    content: str | list[Part] | None = None
    name: str | None = None

    model_config = ConfigDict(extra="allow")


class Streamopts(BaseModel):
    include_usage: bool = False

    model_config = ConfigDict(extra="allow")


class Chatreq(BaseModel):
    model: str
    messages: list[Msg]
    stream: bool = False
    reasoning_effort: str | None = None
    stream_options: Streamopts | None = None
    response_format: dict[str, Any] | None = None

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("messages")
    @classmethod
    def nonempty(cls, v: list[Msg]) -> list[Msg]:
        if not v:
            raise ValueError("messages must be non-empty")
        return v


class Reason(BaseModel):
    effort: str | None = None

    model_config = ConfigDict(extra="allow")


class Respreq(BaseModel):
    model: str
    input: str | list[dict[str, Any]] | None = None
    instructions: str | None = None
    stream: bool = False
    store: bool = True
    reasoning: Reason | None = None
    previous_response_id: str | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, Any] | None = None
    background: bool | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    max_tool_calls: int | None = None
    text: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None

    temperature: float | None = None
    top_p: float | None = None

    model_config = ConfigDict(extra="allow")
