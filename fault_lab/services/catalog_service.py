from __future__ import annotations

import asyncio
import random
from typing import Optional

from fastapi import FastAPI, HTTPException

from fault_lab.common.runtime import ServiceRuntime


app = FastAPI(title="Catalog Service")
runtime = ServiceRuntime("catalog-service")

PRODUCTS = [
    {
        "id": "p-100",
        "name": "Noise-Cancelling Headphones",
        "price": 179.0,
        "category": "Audio",
        "inventory": 19,
        "description": "Wireless headphones with active noise cancellation.",
    },
    {
        "id": "p-200",
        "name": "Mechanical Keyboard",
        "price": 129.0,
        "category": "Accessories",
        "inventory": 12,
        "description": "Tactile keyboard for engineers and gamers.",
    },
    {
        "id": "p-300",
        "name": "4K Monitor",
        "price": 349.0,
        "category": "Displays",
        "inventory": 8,
        "description": "27-inch monitor with USB-C docking support.",
    },
    {
        "id": "p-400",
        "name": "USB-C Dock",
        "price": 99.0,
        "category": "Accessories",
        "inventory": 24,
        "description": "Dock with dual display output and Ethernet.",
    },
]


def maybe_filter_products(query: Optional[str] = None, ids: Optional[str] = None) -> list[dict]:
    if ids:
        wanted = {item.strip() for item in ids.split(",") if item.strip()}
        return [product for product in PRODUCTS if product["id"] in wanted]
    if not query:
        return PRODUCTS
    needle = query.lower()
    return [
        product
        for product in PRODUCTS
        if needle in product["name"].lower()
        or needle in product["description"].lower()
        or needle in product["category"].lower()
    ]


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/products")
async def list_products(q: Optional[str] = None, ids: Optional[str] = None) -> dict:
    context = runtime.begin_request()
    status_code = 200
    extra_cpu = 0.25
    try:
        faults = await runtime.get_faults()
        if faults.get("dependency_delay"):
            await asyncio.sleep(0.10 + 0.50 * faults["dependency_delay"])
        if faults.get("dependency_outage") and random.random() < (0.18 + 0.55 * faults["dependency_outage"]):
            status_code = 503
            raise HTTPException(status_code=503, detail="Catalog dependency outage")
        products = maybe_filter_products(query=q, ids=ids)
        return {"items": products}
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    finally:
        await runtime.emit_telemetry(
            context,
            path="/products",
            status_code=status_code,
            extra_cpu=extra_cpu,
        )
