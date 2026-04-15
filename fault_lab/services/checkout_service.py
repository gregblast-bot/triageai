from __future__ import annotations

import asyncio
import random
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fault_lab.common.clients import request_json
from fault_lab.common.config import AUTH_BASE_URL, CART_BASE_URL, CATALOG_BASE_URL
from fault_lab.common.runtime import ServiceRuntime, async_busy_wait


app = FastAPI(title="Checkout Service")
runtime = ServiceRuntime("checkout-service")


class CheckoutRequest(BaseModel):
    client_id: str
    session_token: str | None = None


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/checkout")
async def checkout(payload: CheckoutRequest) -> dict:
    context = runtime.begin_request()
    status_code = 200
    faults: dict[str, float] = {}
    try:
        faults = await runtime.get_faults()
        if not payload.session_token:
            status_code = 401
            raise HTTPException(status_code=401, detail="Login required before checkout")

        validate_status, validate_payload = await request_json(
            "GET",
            f"{AUTH_BASE_URL}/validate",
            params={"token": payload.session_token},
        )
        if validate_status != 200:
            status_code = validate_status
            raise HTTPException(status_code=validate_status, detail=validate_payload.get("detail", "Auth failed"))

        cart_status, cart_payload = await request_json(
            "GET",
            f"{CART_BASE_URL}/cart/{payload.client_id}",
        )
        if cart_status != 200:
            status_code = cart_status
            raise HTTPException(status_code=cart_status, detail=cart_payload.get("detail", "Cart unavailable"))

        items = cart_payload.get("items", {})
        if not items:
            status_code = 400
            raise HTTPException(status_code=400, detail="Cart is empty")

        product_ids = ",".join(items.keys())
        catalog_status, catalog_payload = await request_json(
            "GET",
            f"{CATALOG_BASE_URL}/products",
            params={"ids": product_ids},
        )
        if catalog_status != 200:
            status_code = catalog_status
            raise HTTPException(status_code=catalog_status, detail=catalog_payload.get("detail", "Catalog unavailable"))

        if faults.get("cpu_exhaustion"):
            runtime.add_queue_pressure(2.0 + faults["cpu_exhaustion"] * 4.0)
            await async_busy_wait(0.08 + 0.25 * faults["cpu_exhaustion"])
        if faults.get("cascading_failure"):
            runtime.add_queue_pressure(3.0 + faults["cascading_failure"] * 6.0)
            await asyncio.sleep(0.20 + 0.70 * faults["cascading_failure"])
            if random.random() < (0.10 + 0.50 * faults["cascading_failure"]):
                status_code = 503
                raise HTTPException(status_code=503, detail="Checkout cascade reached upstream timeout")

        product_map = {item["id"]: item for item in catalog_payload.get("items", [])}
        subtotal = 0.0
        lines = []
        for product_id, quantity in items.items():
            product = product_map.get(product_id)
            if not product:
                continue
            line_total = float(product["price"]) * int(quantity)
            subtotal += line_total
            lines.append(
                {
                    "product_id": product_id,
                    "name": product["name"],
                    "quantity": quantity,
                    "line_total": round(line_total, 2),
                }
            )

        await request_json("POST", f"{CART_BASE_URL}/cart/{payload.client_id}/clear")
        return {
            "order_id": f"ord-{uuid.uuid4().hex[:10]}",
            "subtotal": round(subtotal, 2),
            "line_items": lines,
            "user": validate_payload["user"],
        }
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    finally:
        extra_cpu = 0.4 + 2.4 * faults.get("cpu_exhaustion", 0.0)
        await runtime.emit_telemetry(
            context,
            path="/checkout",
            status_code=status_code,
            extra_cpu=extra_cpu,
        )
