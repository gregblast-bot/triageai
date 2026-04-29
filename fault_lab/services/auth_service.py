from __future__ import annotations

import asyncio
import random
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fault_lab.common.config import DEFAULT_USERS
from fault_lab.common.runtime import ServiceRuntime

TOKEN_TTL_SEC = 3600
MAX_TOKENS = 2048

app = FastAPI(title="Auth Service")
runtime = ServiceRuntime("auth-service")
TOKENS: dict[str, dict] = {}


def _evict_expired_tokens() -> None:
    now = time.monotonic()
    expired = [k for k, v in TOKENS.items() if now - v.get("created_at", 0) > TOKEN_TTL_SEC]
    for key in expired:
        del TOKENS[key]
    if len(TOKENS) > MAX_TOKENS:
        by_age = sorted(TOKENS, key=lambda k: TOKENS[k].get("created_at", 0))
        for key in by_age[: len(TOKENS) - MAX_TOKENS]:
            del TOKENS[key]


class LoginRequest(BaseModel):
    email: str
    password: str


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/login")
async def login(payload: LoginRequest) -> dict:
    context = runtime.begin_request()
    status_code = 200
    auth_error = False
    extra_cpu = 0.2
    try:
        faults = await runtime.get_faults()
        if faults.get("latency_spike"):
            await asyncio.sleep(0.12 + 0.45 * faults["latency_spike"])
        if faults.get("auth_failure") and random.random() < (0.35 + 0.55 * faults["auth_failure"]):
            auth_error = True
            status_code = 503
            raise HTTPException(status_code=503, detail="Auth dependency is failing")

        user = DEFAULT_USERS.get(payload.email)
        if not user or user["password"] != payload.password:
            auth_error = True
            status_code = 401
            raise HTTPException(status_code=401, detail="Invalid credentials")

        _evict_expired_tokens()
        token = f"tok-{uuid.uuid4().hex}"
        TOKENS[token] = {"email": payload.email, "name": user["name"], "created_at": time.monotonic()}
        return {"token": token, "user": {"email": payload.email, "name": user["name"]}}
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    finally:
        await runtime.emit_telemetry(
            context,
            path="/login",
            status_code=status_code,
            auth_error=auth_error,
            extra_cpu=extra_cpu,
        )


@app.get("/validate")
async def validate(token: str) -> dict:
    context = runtime.begin_request()
    status_code = 200
    auth_error = False
    try:
        faults = await runtime.get_faults()
        if faults.get("latency_spike"):
            await asyncio.sleep(0.08 + 0.35 * faults["latency_spike"])
        if faults.get("auth_failure") and random.random() < (0.25 + 0.50 * faults["auth_failure"]):
            auth_error = True
            status_code = 503
            raise HTTPException(status_code=503, detail="Token validation unavailable")

        entry = TOKENS.get(token)
        if not entry or time.monotonic() - entry.get("created_at", 0) > TOKEN_TTL_SEC:
            if entry:
                del TOKENS[token]
            auth_error = True
            status_code = 401
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"valid": True, "user": {"email": entry["email"], "name": entry["name"]}}
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    finally:
        await runtime.emit_telemetry(
            context,
            path="/validate",
            status_code=status_code,
            auth_error=auth_error,
            extra_cpu=0.1,
        )
