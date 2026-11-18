#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import secrets
import uuid
from pathlib import Path
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from multidict import CIMultiDict


LOG = logging.getLogger("deepseek-public-gateway")
MODEL_ID = "deepseek-v4-flash-local"
MODEL_ALIAS = "gpt-5.6-luna"
PUBLIC_MODEL_IDS = (MODEL_ID, MODEL_ALIAS)
TOKEN_FILE = Path(
    os.environ.get(
        "DEEPSEEK_GATEWAY_TOKEN_FILE",
        "/run/secrets/deepseek-public-api-token",
    )
)
UPSTREAM_URL = os.environ.get(
    "DEEPSEEK_GATEWAY_UPSTREAM_URL",
    "http://stream-proxy:3002",
).rstrip("/")
MAX_REQUEST_BYTES = int(
    os.environ.get("DEEPSEEK_GATEWAY_MAX_REQUEST_BYTES", str(32 * 1024 * 1024))
)
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
FORWARDED_REQUEST_HEADERS = {
    "accept",
    "content-type",
    "x-forwarded-for",
    "x-forwarded-proto",
    "x-request-id",
}


def openai_error(message: str, error_type: str, status: int) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": error_type}},
        status=status,
    )


def read_token(path: Path = TOKEN_FILE) -> str:
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 48:
        raise RuntimeError("DeepSeek gateway token must contain at least 48 characters")
    return token


class DeepSeekGateway:
    def __init__(self, token: str, upstream_url: str = UPSTREAM_URL) -> None:
        self.token = token
        self.upstream_url = upstream_url.rstrip("/")
        self.session: ClientSession | None = None

    async def start(self, _app: web.Application) -> None:
        self.session = ClientSession(
            timeout=ClientTimeout(total=None, connect=15, sock_connect=15, sock_read=None)
        )

    async def close(self, _app: web.Application) -> None:
        if self.session is not None:
            await self.session.close()

    def authorized(self, request: web.Request) -> bool:
        scheme, separator, supplied = request.headers.get("Authorization", "").partition(" ")
        return bool(
            separator
            and scheme.lower() == "bearer"
            and secrets.compare_digest(supplied.strip(), self.token)
        )

    @staticmethod
    def response_headers(upstream: Any) -> CIMultiDict:
        headers = CIMultiDict()
        for raw_name, raw_value in upstream.raw_headers:
            name = raw_name.decode("latin-1")
            if name.lower() not in HOP_HEADERS | {"content-length"}:
                headers.add(name, raw_value.decode("latin-1"))
        return headers

    @staticmethod
    def request_headers(request: web.Request, request_id: str) -> CIMultiDict:
        headers = CIMultiDict()
        for name, value in request.headers.items():
            if name.lower() in FORWARDED_REQUEST_HEADERS:
                headers.add(name, value)
        headers["Content-Type"] = "application/json"
        headers["X-Request-ID"] = request_id
        # Stream's local latest-wins policy fingerprints User-Agent. A unique
        # value prevents independent agents sharing one public token from
        # superseding each other's active streams.
        headers["User-Agent"] = f"deepseek-public-gateway/{request_id}"
        return headers

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def models(self, request: web.Request) -> web.Response:
        if not self.authorized(request):
            return openai_error("Invalid API key", "authentication_error", 401)
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "local",
                    }
                    for model_id in PUBLIC_MODEL_IDS
                ],
            }
        )

    async def inference(self, request: web.Request) -> web.StreamResponse:
        if not self.authorized(request):
            return openai_error("Invalid API key", "authentication_error", 401)
        try:
            body = await request.read()
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return openai_error("Request body must be valid JSON", "invalid_request_error", 400)
        if not isinstance(payload, dict):
            return openai_error("Request body must be a JSON object", "invalid_request_error", 400)
        requested_model = payload.get("model")
        if requested_model not in PUBLIC_MODEL_IDS:
            return openai_error(
                f"This endpoint only serves {', '.join(PUBLIC_MODEL_IDS)}",
                "invalid_model",
                400,
            )
        if requested_model == MODEL_ALIAS:
            payload["model"] = MODEL_ID
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

        request_id = request.headers.get("X-Request-ID", "").strip() or uuid.uuid4().hex
        upstream_path = request.path
        if request.query_string:
            upstream_path = f"{upstream_path}?{request.query_string}"
        assert self.session is not None
        try:
            async with self.session.post(
                f"{self.upstream_url}{upstream_path}",
                data=body,
                headers=self.request_headers(request, request_id),
                allow_redirects=False,
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=self.response_headers(upstream),
                )
                response.headers["X-Request-ID"] = request_id
                await response.prepare(request)
                async for chunk in upstream.content.iter_chunked(1024 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (ClientError, OSError, ConnectionError) as exc:
            LOG.warning("upstream_failed request_id=%s error=%r", request_id, exc)
            return openai_error("DeepSeek upstream unavailable", "upstream_error", 503)


def build_app(*, token: str | None = None, upstream_url: str = UPSTREAM_URL) -> web.Application:
    gateway = DeepSeekGateway(token or read_token(), upstream_url)
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    app.router.add_get("/health", gateway.health)
    app.router.add_get("/v1/models", gateway.models)
    app.router.add_post("/v1/chat/completions", gateway.inference)
    app.router.add_post("/v1/responses", gateway.inference)
    app.on_startup.append(gateway.start)
    app.on_cleanup.append(gateway.close)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    web.run_app(build_app(), host="0.0.0.0", port=8080)
