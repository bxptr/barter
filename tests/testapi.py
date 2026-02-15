from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.codex import Call, Result, Update, Usage
from app.main import Cfg, MODELS, mkapp


class FakeCodex:
    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.usage = Usage(prompt=10, completion=5, total=15)

    async def run(self, call: Call) -> Result:
        self.calls.append(call)
        return Result(text="fake answer", usage=self.usage)

    async def stream(self, call: Call):
        self.calls.append(call)
        yield Update(delta="fake ")
        yield Update(delta="answer")
        yield Update(done=True, usage=self.usage)


class ToolCodex:
    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.usage = Usage(prompt=3, completion=2, total=5)
        self.toolstep = 0

    async def run(self, call: Call) -> Result:
        self.calls.append(call)
        if "TOOL MODE." in call.prompt:
            self.toolstep += 1
            if self.toolstep == 1:
                return Result(
                    text='{"type":"tool","name":"web_search","arguments":{"query":"kittens","max_results":2}}',
                    usage=self.usage,
                )
            return Result(text='{"type":"final","text":"done"}', usage=self.usage)

        return Result(text="tool answer", usage=self.usage)

    async def stream(self, call: Call):
        self.calls.append(call)
        yield Update(delta="streamed ")
        yield Update(delta="answer")
        yield Update(done=True, usage=self.usage)


class FakeTools:
    async def web(self, query: str, *, maxres: int = 5, allow=None):
        _ = (query, maxres, allow)
        return [
            {
                "title": "Example",
                "url": "https://example.com",
                "snippet": "snippet",
            }
        ]

    async def py(self, code: str, *, timeout: int = 10):
        _ = (code, timeout)
        return {
            "ok": True,
            "timeout": False,
            "exit_code": 0,
            "stdout": "2\n",
            "stderr": "",
        }


class JsonCodex:
    def __init__(self, text: str) -> None:
        self.calls: list[Call] = []
        self.text = text
        self.usage = Usage(prompt=1, completion=1, total=2)

    async def run(self, call: Call) -> Result:
        self.calls.append(call)
        return Result(text=self.text, usage=self.usage)

    async def stream(self, call: Call):
        self.calls.append(call)
        yield Update(delta=self.text)
        yield Update(done=True, usage=self.usage)


class RetryCodex:
    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.usage = Usage(prompt=1, completion=1, total=2)
        self.n = 0

    async def run(self, call: Call) -> Result:
        self.calls.append(call)
        self.n += 1
        if self.n == 1:
            return Result(text="not json", usage=self.usage)
        return Result(text='{"answer": 1}', usage=self.usage)

    async def stream(self, call: Call):
        self.calls.append(call)
        yield Update(delta="not used")
        yield Update(done=True, usage=self.usage)


def cfg(tmp_path: Path) -> Cfg:
    return Cfg(bin="codex", cwd=tmp_path, timeout=30)


def test_healthz(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex())
    client = TestClient(app)

    resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_models(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex())
    client = TestClient(app)

    resp = client.get("/v1/models")

    assert resp.status_code == 200
    payload = resp.json()
    ids = [item["id"] for item in payload["data"]]
    assert ids == MODELS


def test_rejects_unknown_model(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex())
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "nope",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "model"


def test_rejects_unknown_reasoning_effort_chat(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex())
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODELS[0],
            "reasoning_effort": "ultra",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "reasoning_effort"


def test_chat_non_stream(tmp_path: Path) -> None:
    fake = FakeCodex()
    app = mkapp(cfg=cfg(tmp_path), codex=fake)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODELS[0],
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "fake answer"
    assert fake.calls


def test_chat_stream(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex())
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": MODELS[0],
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    ) as resp:
        body = "\n".join(line for line in resp.iter_lines() if line)

    assert resp.status_code == 200
    assert "chat.completion.chunk" in body
    assert "data: [DONE]" in body


def test_image_data_url_is_materialized(tmp_path: Path) -> None:
    fake = FakeCodex()
    app = mkapp(cfg=cfg(tmp_path), codex=fake)
    client = TestClient(app)

    tiny_png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODELS[0],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": tiny_png}},
                    ],
                }
            ],
        },
    )

    assert resp.status_code == 200
    assert fake.calls
    assert len(fake.calls[-1].imgs) == 1


def test_responses_non_stream(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex())
    client = TestClient(app)

    resp = client.post(
        "/v1/responses",
        json={
            "model": MODELS[0],
            "input": "Tell me one fact.",
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["output_text"] == "fake answer"


def test_responses_stream(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex())
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": MODELS[0],
            "stream": True,
            "input": "hello",
        },
    ) as resp:
        body = "\n".join(line for line in resp.iter_lines() if line)

    assert resp.status_code == 200
    assert "event: response.output_text.delta" in body
    assert "event: response.completed" in body


def test_responses_image_input_data_url(tmp_path: Path) -> None:
    fake = FakeCodex()
    app = mkapp(cfg=cfg(tmp_path), codex=fake)
    client = TestClient(app)

    tiny_png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    resp = client.post(
        "/v1/responses",
        json={
            "model": MODELS[0],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is in this image?"},
                        {"type": "input_image", "image_url": tiny_png},
                    ],
                }
            ],
        },
    )

    assert resp.status_code == 200
    assert fake.calls
    assert len(fake.calls[-1].imgs) == 1


def test_responses_store_and_retrieve_and_delete(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex())
    client = TestClient(app)

    created = client.post(
        "/v1/responses",
        json={"model": MODELS[0], "input": "hello"},
    )
    assert created.status_code == 200
    rid = created.json()["id"]

    got = client.get(f"/v1/responses/{rid}")
    assert got.status_code == 200
    assert got.json()["id"] == rid

    items = client.get(f"/v1/responses/{rid}/input_items")
    assert items.status_code == 200
    assert items.json()["object"] == "list"

    deleted = client.delete(f"/v1/responses/{rid}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    missing = client.get(f"/v1/responses/{rid}")
    assert missing.status_code == 404


def test_responses_rejects_unknown_reasoning_effort(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex())
    client = TestClient(app)

    resp = client.post(
        "/v1/responses",
        json={
            "model": MODELS[0],
            "reasoning": {"effort": "ultra"},
            "input": "hello",
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "reasoning.effort"


def test_responses_tools_non_stream(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=ToolCodex(), tools=FakeTools())
    client = TestClient(app)

    resp = client.post(
        "/v1/responses",
        json={
            "model": MODELS[0],
            "input": "Search for kittens then answer.",
            "tools": [{"type": "web_search"}],
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["tool_choice"] == "auto"
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["output"][0]["type"] == "web_search_call"
    assert payload["output"][0]["status"] == "completed"
    assert payload["output"][0]["results"]
    assert payload["output_text"] == "tool answer"


def test_responses_tools_stream(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=ToolCodex(), tools=FakeTools())
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": MODELS[0],
            "stream": True,
            "input": "Search for kittens then answer.",
            "tools": [{"type": "web_search"}],
        },
    ) as resp:
        body = "\n".join(line for line in resp.iter_lines() if line)

    assert resp.status_code == 200
    assert "event: response.web_search_call.in_progress" in body
    assert "event: response.output_text.delta" in body
    assert "event: response.completed" in body


def test_responses_rejects_unknown_tool(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex(), tools=FakeTools())
    client = TestClient(app)

    resp = client.post(
        "/v1/responses",
        json={
            "model": MODELS[0],
            "input": "hello",
            "tools": [{"type": "file_search"}],
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "tools"


def test_responses_structured_json_schema(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=JsonCodex('{"answer":1}'), tools=FakeTools())
    client = TestClient(app)

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    resp = client.post(
        "/v1/responses",
        json={
            "model": MODELS[0],
            "input": "Return {answer:int}.",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ans",
                    "strict": True,
                    "schema": schema,
                }
            },
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert json.loads(payload["output_text"]) == {"answer": 1}


def test_responses_structured_stream(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=JsonCodex('{"answer":1}'), tools=FakeTools())
    client = TestClient(app)

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": MODELS[0],
            "stream": True,
            "input": "Return {answer:int}.",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ans",
                    "strict": True,
                    "schema": schema,
                }
            },
        },
    ) as resp:
        body = "\n".join(line for line in resp.iter_lines() if line)

    assert resp.status_code == 200
    assert "event: response.output_text.delta" in body
    assert "event: response.completed" in body


def test_responses_structured_retries(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=RetryCodex(), tools=FakeTools())
    client = TestClient(app)

    resp = client.post(
        "/v1/responses",
        json={
            "model": MODELS[0],
            "input": "Return JSON.",
            "text": {"format": {"type": "json_object"}},
        },
    )

    assert resp.status_code == 200
    assert json.loads(resp.json()["output_text"]) == {"answer": 1}


def test_chat_structured_json_object(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=JsonCodex('{"answer":1}'), tools=FakeTools())
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODELS[0],
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {"type": "json_object"},
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert json.loads(payload["choices"][0]["message"]["content"]) == {"answer": 1}


def test_responses_rejects_bad_text_format(tmp_path: Path) -> None:
    app = mkapp(cfg=cfg(tmp_path), codex=FakeCodex(), tools=FakeTools())
    client = TestClient(app)

    resp = client.post(
        "/v1/responses",
        json={
            "model": MODELS[0],
            "input": "hello",
            "text": {"format": "json_schema"},
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "text.format"
