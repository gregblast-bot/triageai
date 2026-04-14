from __future__ import annotations

import asyncio
import threading

from fastapi import FastAPI
from pydantic import BaseModel

from fault_lab.common.runtime import ServiceRuntime


app = FastAPI(title="Cart Service")
runtime = ServiceRuntime("cart-service")

CARTS: dict[str, dict[str, int]] = {}
cart_lock = threading.Lock()


class CartItemRequest(BaseModel):
    product_id: str
    quantity: int = 1


def get_cart(client_id: str) -> dict[str, int]:
    with cart_lock:
        return dict(CARTS.get(client_id, {}))


def cart_totals(client_id: str) -> tuple[int, int]:
    items = get_cart(client_id)
    return len(items), sum(items.values())


def build_cart_snapshot(client_id: str) -> dict:
    items = get_cart(client_id)
    unique_items, total_quantity = cart_totals(client_id)
    return {
        "client_id": client_id,
        "items": items,
        "unique_items": unique_items,
        "total_quantity": total_quantity,
    }


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/cart/{client_id}")
async def fetch_cart(client_id: str) -> dict:
    context = runtime.begin_request()
    status_code = 200
    try:
        faults = await runtime.get_faults()
        if faults.get("memory_leak"):
            runtime.leak_memory_mb(1.5 * faults["memory_leak"])
        if faults.get("queue_congestion"):
            runtime.add_queue_pressure(1.6 + faults["queue_congestion"] * 4.0)
            await asyncio.sleep(0.05 + 0.35 * faults["queue_congestion"])
        return build_cart_snapshot(client_id)
    finally:
        await runtime.emit_telemetry(
            context,
            path="/cart/get",
            status_code=status_code,
            extra_cpu=0.15,
        )


@app.post("/cart/{client_id}/items")
async def add_item(client_id: str, payload: CartItemRequest) -> dict:
    context = runtime.begin_request()
    status_code = 200
    try:
        faults = await runtime.get_faults()
        if faults.get("memory_leak"):
            runtime.leak_memory_mb(2.0 * faults["memory_leak"])
        if faults.get("queue_congestion"):
            runtime.add_queue_pressure(2.8 + faults["queue_congestion"] * 5.0)
            await asyncio.sleep(0.08 + 0.45 * faults["queue_congestion"])
        with cart_lock:
            CARTS.setdefault(client_id, {})
            CARTS[client_id][payload.product_id] = CARTS[client_id].get(payload.product_id, 0) + max(1, payload.quantity)
        return build_cart_snapshot(client_id)
    finally:
        await runtime.emit_telemetry(
            context,
            path="/cart/add",
            status_code=status_code,
            extra_cpu=0.35,
        )


@app.delete("/cart/{client_id}/items/{product_id}")
async def remove_item(client_id: str, product_id: str) -> dict:
    context = runtime.begin_request()
    status_code = 200
    try:
        faults = await runtime.get_faults()
        if faults.get("memory_leak"):
            runtime.leak_memory_mb(0.8 * faults["memory_leak"])
        if faults.get("queue_congestion"):
            runtime.add_queue_pressure(1.2 + faults["queue_congestion"] * 3.0)
            await asyncio.sleep(0.03 + 0.25 * faults["queue_congestion"])
        with cart_lock:
            if client_id in CARTS:
                CARTS[client_id].pop(product_id, None)
        return build_cart_snapshot(client_id)
    finally:
        await runtime.emit_telemetry(
            context,
            path="/cart/remove",
            status_code=status_code,
            extra_cpu=0.2,
        )


@app.post("/cart/{client_id}/clear")
async def clear_cart(client_id: str) -> dict:
    context = runtime.begin_request()
    status_code = 200
    try:
        with cart_lock:
            CARTS[client_id] = {}
        return {"ok": True}
    finally:
        await runtime.emit_telemetry(
            context,
            path="/cart/clear",
            status_code=status_code,
            extra_cpu=0.12,
        )
