from __future__ import annotations

import asyncio
import random
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fault_lab.common.config import DEFAULT_USERS
from fault_lab.common.runtime import ServiceRuntime


app = FastAPI(title="Auth Service")
runtime = ServiceRuntime("auth-service")
TOKENS: dict[str, dict] = {}


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

        token = f"tok-{uuid.uuid4().hex}"
        TOKENS[token] = {"email": payload.email, "name": user["name"]}
        return {"token": token, "user": TOKENS[token]}
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

        user = TOKENS.get(token)
        if not user:
            auth_error = True
            status_code = 401
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"valid": True, "user": user}
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
