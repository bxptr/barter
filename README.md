# barter
proxy codex as an openai-compatible api

barter is a full-featured openai-compatible api proxy that forwards requests to a standalone codex, allowing you to _barter_ codex usage that comes for a chatgpt plan for api-esque credits.
it supports the latest openai api spec (including streaming, images, structured responses) and native codex live web search. 
codex runs with read-only filesystem "sandboxing" (only compute and filesystem limits are enforced for now). 

to use, create an admin key with `./barter gen-admin-key` to bootstrap and issue service keys via `/v1/keys`. the admin key works for api requests too.

serve the proxy with `./barter serve --admin-key ...`. see `.env.example` for configuration knobs.
