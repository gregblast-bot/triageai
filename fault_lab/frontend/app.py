from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fault_lab.common.clients import request_json
from fault_lab.common.config import (
    AUTH_BASE_URL,
    CART_BASE_URL,
    CATALOG_BASE_URL,
    CHECKOUT_BASE_URL,
    CONTROL_BASE_URL,
)


app = FastAPI(title="Fault Lab Storefront")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")


def redirect_with_message(path: str, message: str, level: str = "info") -> RedirectResponse:
    encoded = quote_plus(message)
    response = RedirectResponse(f"{path}?message={encoded}&level={level}", status_code=303)
    return response


def ensure_client_id(request: Request, response: object | None = None) -> str:
    client_id = request.cookies.get("client_id")
    if client_id:
        return client_id
    client_id = f"client-{uuid.uuid4().hex[:10]}"
    if response is not None and hasattr(response, "set_cookie"):
        response.set_cookie("client_id", client_id, httponly=False)
    return client_id


async def load_home_context(request: Request, client_id: str, q: str | None = None) -> dict:
    status_code, payload = await request_json(
        "GET",
        f"{CATALOG_BASE_URL}/products",
        params={"q": q} if q else None,
    )
    products = payload.get("items", []) if status_code == 200 else []
    cart_status, cart_payload = await request_json("GET", f"{CART_BASE_URL}/cart/{client_id}")
    cart_count = cart_payload.get("total_quantity", 0) if cart_status == 200 else 0
    return {
        "request": request,
        "products": products,
        "cart_count": cart_count,
        "query": q or "",
        "message": request.query_params.get("message"),
        "level": request.query_params.get("level", "info"),
        "user_email": request.cookies.get("user_email"),
        "service_issue": None if status_code == 200 else payload.get("detail", "Catalog unavailable"),
    }


async def load_cart_context(request: Request, client_id: str) -> dict:
    cart_status, cart_payload = await request_json("GET", f"{CART_BASE_URL}/cart/{client_id}")
    cart_items = cart_payload.get("items", {}) if cart_status == 200 else {}
    ids = ",".join(cart_items.keys()) if cart_items else ""
    catalog_status, catalog_payload = await request_json(
        "GET",
        f"{CATALOG_BASE_URL}/products",
        params={"ids": ids} if ids else None,
    )
    product_map = {item["id"]: item for item in catalog_payload.get("items", [])} if catalog_status == 200 else {}
    line_items = []
    subtotal = 0.0
    for product_id, quantity in cart_items.items():
        product = product_map.get(product_id, {"name": product_id, "price": 0.0})
        line_total = float(product["price"]) * int(quantity)
        subtotal += line_total
        line_items.append(
            {
                "product_id": product_id,
                "name": product["name"],
                "quantity": quantity,
                "unit_price": product["price"],
                "line_total": round(line_total, 2),
            }
        )
    return {
        "request": request,
        "line_items": line_items,
        "subtotal": round(subtotal, 2),
        "message": request.query_params.get("message"),
        "level": request.query_params.get("level", "info"),
        "user_email": request.cookies.get("user_email"),
    }


async def load_admin_context(request: Request) -> dict:
    faults_status, faults_payload = await request_json("GET", f"{CONTROL_BASE_URL}/api/faults")
    telemetry_status, telemetry_payload = await request_json("GET", f"{CONTROL_BASE_URL}/api/telemetry/summary")
    return {
        "request": request,
        "faults": faults_payload.get("faults", {}) if faults_status == 200 else {},
        "scenarios": faults_payload.get("scenarios", []) if faults_status == 200 else [],
        "summary": telemetry_payload if telemetry_status == 200 else {},
        "message": request.query_params.get("message"),
        "level": request.query_params.get("level", "info"),
        "download_url": "http://localhost:8001/api/telemetry/window.csv?limit=120",
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, q: str | None = None):
    client_id = ensure_client_id(request)
    context = await load_home_context(request, client_id, q=q)
    response = templates.TemplateResponse("home.html", context)
    ensure_client_id(request, response)
    return response


@app.post("/search")
async def search(query: str = Form("")):
    return RedirectResponse(f"/?q={quote_plus(query)}", status_code=303)


@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    status_code, payload = await request_json(
        "POST",
        f"{AUTH_BASE_URL}/login",
        json={"email": email, "password": password},
    )
    if status_code != 200:
        return redirect_with_message("/", payload.get("detail", "Login failed"), "error")

    response = redirect_with_message("/", f"Logged in as {payload['user']['name']}", "success")
    ensure_client_id(request, response)
    response.set_cookie("session_token", payload["token"], httponly=True)
    response.set_cookie("user_email", payload["user"]["email"] if "email" in payload["user"] else email, httponly=False)
    return response


@app.post("/logout")
async def logout():
    response = redirect_with_message("/", "Signed out", "info")
    response.delete_cookie("session_token")
    response.delete_cookie("user_email")
    return response


@app.post("/cart/add/{product_id}")
async def add_to_cart(request: Request, product_id: str):
    response = redirect_with_message("/", "Added item to cart", "success")
    client_id = ensure_client_id(request, response)
    status_code, payload = await request_json(
        "POST",
        f"{CART_BASE_URL}/cart/{client_id}/items",
        json={"product_id": product_id, "quantity": 1},
    )
    if status_code != 200:
        return redirect_with_message("/", payload.get("detail", "Cart update failed"), "error")
    return response


@app.get("/cart", response_class=HTMLResponse)
async def cart(request: Request):
    client_id = ensure_client_id(request)
    context = await load_cart_context(request, client_id)
    response = templates.TemplateResponse("cart.html", context)
    ensure_client_id(request, response)
    return response


@app.post("/cart/remove/{product_id}")
async def remove_from_cart(request: Request, product_id: str):
    client_id = ensure_client_id(request)
    status_code, payload = await request_json(
        "DELETE",
        f"{CART_BASE_URL}/cart/{client_id}/items/{product_id}",
    )
    if status_code != 200:
        return redirect_with_message("/cart", payload.get("detail", "Remove failed"), "error")
    return redirect_with_message("/cart", "Removed item from cart", "success")


@app.post("/checkout")
async def checkout(request: Request):
    client_id = ensure_client_id(request)
    token = request.cookies.get("session_token")
    status_code, payload = await request_json(
        "POST",
        f"{CHECKOUT_BASE_URL}/checkout",
        json={"client_id": client_id, "session_token": token},
    )
    if status_code != 200:
        return redirect_with_message("/cart", payload.get("detail", "Checkout failed"), "error")
    message = f"Checkout complete. Order {payload['order_id']} for ${payload['subtotal']}"
    return redirect_with_message("/cart", message, "success")


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    context = await load_admin_context(request)
    response = templates.TemplateResponse("admin.html", context)
    ensure_client_id(request, response)
    return response


@app.post("/admin/scenario")
async def set_scenario(scenario: str = Form(...)):
    status_code, payload = await request_json(
        "POST",
        f"{CONTROL_BASE_URL}/api/faults/scenario",
        json={"scenario": scenario},
    )
    if status_code != 200 or not payload.get("ok"):
        return redirect_with_message("/admin", payload.get("detail", "Could not update scenario"), "error")
    return redirect_with_message("/admin", f"Scenario set to {scenario}", "success")


@app.post("/admin/fault")
async def toggle_fault(
    service: str = Form(...),
    fault: str = Form(...),
    enabled: str = Form(...),
    intensity: float = Form(...),
):
    status_code, payload = await request_json(
        "POST",
        f"{CONTROL_BASE_URL}/api/faults/toggle",
        json={
            "service": service,
            "fault": fault,
            "enabled": enabled == "true",
            "intensity": intensity,
        },
    )
    if status_code != 200 or not payload.get("ok"):
        return redirect_with_message("/admin", "Fault update failed", "error")
    return redirect_with_message("/admin", f"Updated {service}:{fault}", "success")


async def simulate_browse_burst(client_id: str) -> None:
    tasks = []
    for _ in range(8):
        tasks.append(request_json("GET", f"{CATALOG_BASE_URL}/products", params={"q": "dock"}))
        tasks.append(
            request_json(
                "POST",
                f"{CART_BASE_URL}/cart/{client_id}/items",
                json={"product_id": "p-100", "quantity": 1},
            )
        )
    await asyncio.gather(*tasks)


async def simulate_checkout_burst(client_id: str, session_token: str | None) -> None:
    token = session_token
    if not token:
        login_status, login_payload = await request_json(
            "POST",
            f"{AUTH_BASE_URL}/login",
            json={"email": "demo@triage.ai", "password": "demo123"},
        )
        if login_status == 200:
            token = login_payload["token"]

    await request_json(
        "POST",
        f"{CART_BASE_URL}/cart/{client_id}/items",
        json={"product_id": "p-200", "quantity": 1},
    )
    tasks = []
    for _ in range(6):
        tasks.append(
            request_json(
                "POST",
                f"{CHECKOUT_BASE_URL}/checkout",
                json={"client_id": client_id, "session_token": token},
            )
        )
    await asyncio.gather(*tasks)


@app.post("/admin/burst")
async def run_burst(request: Request, burst_type: str = Form(...)):
    client_id = ensure_client_id(request)
    token = request.cookies.get("session_token")
    if burst_type == "browse":
        await simulate_browse_burst(client_id)
        return redirect_with_message("/admin", "Browse burst executed", "success")
    if burst_type == "checkout":
        await simulate_checkout_burst(client_id, token)
        return redirect_with_message("/admin", "Checkout burst executed", "success")
    return redirect_with_message("/admin", "Unknown burst type", "error")


@app.post("/admin/reset-telemetry")
async def reset_telemetry():
    await request_json("POST", f"{CONTROL_BASE_URL}/api/telemetry/reset")
    return redirect_with_message("/admin", "Telemetry reset", "success")
