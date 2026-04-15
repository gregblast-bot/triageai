from __future__ import annotations

import asyncio

import httpx

from .config import REQUEST_TIMEOUT

_shared_client: httpx.AsyncClient | None = None


def _get_client(timeout: float = REQUEST_TIMEOUT) -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(timeout=timeout)
    return _shared_client


async def close_client() -> None:
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


async def request(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json: dict | None = None,
    headers: dict | None = None,
    retries: int = 2,
    timeout: float = REQUEST_TIMEOUT,
) -> httpx.Response:
    client = _get_client(timeout)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await client.request(
                method,
                url,
                params=params,
                json=json,
                headers=headers,
            )
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            await asyncio.sleep(0.15 * (attempt + 1))
    raise RuntimeError(f"Request failed for {url}: {last_exc}")


async def request_json(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json: dict | None = None,
    headers: dict | None = None,
    retries: int = 2,
    timeout: float = REQUEST_TIMEOUT,
) -> tuple[int, dict]:
    response = await request(
        method,
        url,
        params=params,
        json=json,
        headers=headers,
        retries=retries,
        timeout=timeout,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}
    return response.status_code, payload
