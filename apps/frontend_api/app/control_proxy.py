# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Same-origin frontend gateway to the single management service."""

import os
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

router = APIRouter()


def manager_url():
    return os.getenv(
        "DT_MANAGER_URL", os.getenv("EDGE_MANAGER_URL", "http://127.0.0.1:8001")
    ).rstrip("/")


async def forward(request: Request, path: str):
    if path.split("/")[0] not in {"regions", "edges", "healthz"}:
        raise HTTPException(404)
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "다른 사이트에서 보낸 관리 요청은 허용하지 않습니다.")
    body = await request.body()
    if len(body) > 1024 * 1024:
        raise HTTPException(413)
    headers = {"content-type": "application/json"}
    if os.getenv("DT_MANAGER_TOKEN"):
        headers["authorization"] = "Bearer " + os.environ["DT_MANAGER_TOKEN"]
    client = httpx.AsyncClient(timeout=15)
    try:
        outbound = client.build_request(
            request.method,
            f"{manager_url()}/{path}",
            params=request.query_params,
            content=body,
            headers=headers,
        )
        response = await client.send(outbound, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(
            503, "관리 서비스에 연결할 수 없습니다. 연결 상태를 확인하세요."
        ) from exc
    if path.endswith("/download") and response.is_success:

        async def chunks():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            chunks(),
            media_type="application/x-ndjson",
            headers={
                "content-disposition": response.headers.get(
                    "content-disposition", "attachment"
                )
            },
        )
    try:
        content = await response.aread()
        return Response(
            content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )
    finally:
        await response.aclose()
        await client.aclose()


@router.api_route(
    "/api/v1/control/{path:path}", methods=["GET", "POST", "PUT", "DELETE"]
)
async def control(request: Request, path: str):
    return await forward(request, path)


@router.get("/api/v1/edges")
async def edges(request: Request):
    return await forward(request, "edges")


@router.get("/api/v1/system/status")
async def system_status(request: Request):
    return await forward(request, "regions")
