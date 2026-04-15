from __future__ import annotations

import streamlit as st

from src.config import METRIC_COLUMNS
from src.simulator import (
    PAGES,
    SCENARIO_DESCRIPTIONS,
    TRAFFIC_LEVELS,
    default_simulation_state,
    event_log_frame,
    history_frame,
    metrics_csv_bytes,
    record_event,
    reset_state,
    set_scenario,
    set_traffic_level,
    simulate_background_traffic,
    warm_start_state,
)
from src.triage import models_ready, triage_custom_metrics


st.set_page_config(page_title="ShopSim", layout="wide")


PRODUCTS = [
    {"name": "Noise-Cancelling Headphones", "price": "$179", "tag": "Popular"},
    {"name": "Mechanical Keyboard", "price": "$129", "tag": "New"},
    {"name": "4K Monitor", "price": "$349", "tag": "Limited"},
]


def get_sim_state() -> dict:
    if "shop_sim_state" not in st.session_state:
        st.session_state["shop_sim_state"] = default_simulation_state()
        warm_start_state(st.session_state["shop_sim_state"])
    return st.session_state["shop_sim_state"]


def render_sidebar(state: dict) -> str:
    st.sidebar.title("ShopSim")
    st.sidebar.caption("A small ecommerce simulator that generates telemetry while you browse.")

    scenario = st.sidebar.selectbox(
        "Anomaly scenario",
        options=list(SCENARIO_DESCRIPTIONS.keys()),
        format_func=lambda value: value.replace("_", " ").title(),
        index=list(SCENARIO_DESCRIPTIONS.keys()).index(state["scenario"]),
    )
    set_scenario(state, scenario)
    st.sidebar.write(SCENARIO_DESCRIPTIONS[scenario])

    traffic_level = st.sidebar.select_slider(
        "Traffic level",
        options=list(TRAFFIC_LEVELS.keys()),
        value=state["traffic_level"],
    )
    set_traffic_level(state, traffic_level)

    selected_page = st.sidebar.radio("Navigate app", options=PAGES, index=PAGES.index(state["current_page"]))
    if selected_page != state["last_navigation_page"]:
        record_event(state, page=selected_page, action="navigate", note=f"Navigated to {selected_page}")
        state["last_navigation_page"] = selected_page

    st.sidebar.subheader("Traffic controls")
    if st.sidebar.button("Simulate 5 background requests", use_container_width=True):
        simulate_background_traffic(state, count=5, page=selected_page)
    if st.sidebar.button("Simulate 20 background requests", use_container_width=True):
        simulate_background_traffic(state, count=20, page=selected_page)
    if st.sidebar.button("Reset simulator", use_container_width=True):
        reset_state(state)
        st.rerun()

    st.sidebar.subheader("Export")
    st.sidebar.download_button(
        "Download latest telemetry CSV",
        data=metrics_csv_bytes(state, limit=120),
        file_name=f"shopsim_{state['scenario']}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    return selected_page


def render_header(state: dict):
    st.title("ShopSim")
    st.caption(
        "Browse the storefront, trigger anomalies, and export a telemetry window compatible with TriageAI."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Scenario", state["scenario"].replace("_", " ").title())
    col2.metric("Traffic", state["traffic_level"])
    col3.metric("Samples collected", len(state["history"]))
    col4.metric("Cart items", state["cart_items"])


def render_page_content(state: dict, page: str):
    st.subheader(page)
    if page == "Home":
        st.write("A storefront landing page with recommendations and traffic banners.")
        product_cols = st.columns(3)
        for column, product in zip(product_cols, PRODUCTS):
            column.markdown(f"**{product['name']}**")
            column.caption(f"{product['tag']} · {product['price']}")
            if column.button(f"Open {product['name']}", key=f"home-open-{product['name']}"):
                record_event(state, page="Product", action="open_product", note=f"Opened {product['name']}")
                state["current_page"] = "Product"
                state["last_navigation_page"] = "Product"
                st.rerun()
    elif page == "Search":
        query = st.text_input("Search catalog", value="wireless headset")
        if st.button("Run search", key="search-run"):
            record_event(state, page="Search", action="search", note=f"Searched for '{query}'")
        st.write("Top results")
        st.table(
            [
                {"Result": "Wireless Headphones", "Price": "$179"},
                {"Result": "Gaming Headset", "Price": "$129"},
                {"Result": "Bluetooth Speaker", "Price": "$89"},
            ]
        )
    elif page == "Product":
        st.markdown("**Noise-Cancelling Headphones**")
        st.write("Premium wireless headphones with active noise cancellation and 30-hour battery life.")
        col1, col2 = st.columns(2)
        if col1.button("Add to cart", key="product-add"):
            record_event(state, page="Product", action="add_to_cart", note="Added headphones to cart")
        if col2.button("Refresh inventory", key="product-refresh"):
            record_event(state, page="Product", action="inventory_refresh", note="Refreshed product inventory")
    elif page == "Cart":
        st.write("Review selected items and apply discounts before checkout.")
        col1, col2 = st.columns(2)
        if col1.button("Apply coupon", key="cart-coupon"):
            record_event(state, page="Cart", action="coupon", note="Applied promo code")
        if col2.button("Remove item", key="cart-remove"):
            record_event(state, page="Cart", action="remove_from_cart", note="Removed an item from cart")
    elif page == "Login":
        st.text_input("Email", value="customer@example.com")
        st.text_input("Password", value="hunter2", type="password")
        if st.button("Sign in", key="login-submit"):
            record_event(state, page="Login", action="login_attempt", note="Submitted login form")
    elif page == "Checkout":
        st.write("Finalize payment and place the order.")
        col1, col2 = st.columns(2)
        if col1.button("Place order", key="checkout-submit"):
            record_event(state, page="Checkout", action="checkout_submit", note="Submitted checkout")
        if col2.button("Validate coupon", key="checkout-coupon"):
            record_event(state, page="Checkout", action="coupon", note="Validated coupon during checkout")
    elif page == "Admin":
        st.write("Internal dashboard for operational tasks and analytics.")
        col1, col2 = st.columns(2)
        if col1.button("Open sales report", key="admin-report"):
            record_event(state, page="Admin", action="admin_report", note="Opened admin sales report")
        if col2.button("Refresh inventory cache", key="admin-refresh"):
            record_event(state, page="Admin", action="inventory_refresh", note="Refreshed inventory cache")


def render_telemetry(state: dict):
    history = history_frame(state, limit=120)
    st.subheader("Live telemetry")
    st.line_chart(history.set_index("minute")[METRIC_COLUMNS], use_container_width=True)

    st.subheader("Recent events")
    st.dataframe(event_log_frame(state, limit=12), use_container_width=True, hide_index=True)


def render_triage_preview(state: dict):
    st.subheader("TriageAI preview")
    if not models_ready():
        st.info("Train TriageAI first in `app.py` to enable live triage for simulator telemetry.")
        return

    history = history_frame(state, limit=120)
    result = triage_custom_metrics(
        history,
        title="ShopSim session",
        description=(
            f"Simulated ecommerce telemetry with scenario {state['scenario']} under {state['traffic_level']} traffic."
        ),
        incident_id="SHOPSIM-LIVE-001",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly", "Abnormal" if result["unusual"] else "Normal")
    col2.metric("Predicted fault", result["predicted_fault_type"])
    col3.metric("Predicted service", result["predicted_root_cause_service"])
    st.write(
        f"**Anomaly score:** {result['anomaly_score']:.3f} | "
        f"**Fault confidence:** {result['fault_confidence']:.3f} | "
        f"**Root-cause confidence:** {result['root_cause_confidence']:.3f}"
    )
    if result["top_similar_incidents"]:
        st.write("**Closest training incidents**")
        for item in result["top_similar_incidents"]:
            st.write(
                f"- `{item['incident_id']}` | score={item['similarity']:.3f} | "
                f"fault={item['fault_type']} | service={item['root_cause_service']}"
            )


def main():
    state = get_sim_state()
    selected_page = render_sidebar(state)
    render_header(state)

    left, right = st.columns([1.3, 1.0])
    with left:
        render_page_content(state, selected_page)
    with right:
        render_triage_preview(state)

    render_telemetry(state)


if __name__ == "__main__":
    main()
