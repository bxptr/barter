# barter
proxy codex as an openai-compatible api

barter is a full-featured openai-compatible api proxy that forwards requests to a standalone codex, allowing you to _barter_ codex usage that comes for a chatgpt plan for api-esque credits.
it supports the latest openai api spec (including streaming, images, structured responses) and exposes `web_search` and `code_interpreter` as tools.

to use, set a `BARTER_ADMIN_KEY` to bootstrap and issue service keys via `/v1/keys`. the admin key works for api requests too.

