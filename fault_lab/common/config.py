from __future__ import annotations

import os
from pathlib import Path


CONTROL_BASE_URL = os.getenv("CONTROL_BASE_URL", "http://control-plane:8000")
AUTH_BASE_URL = os.getenv("AUTH_BASE_URL", "http://auth-service:8000")
CATALOG_BASE_URL = os.getenv("CATALOG_BASE_URL", "http://catalog-service:8000")
CART_BASE_URL = os.getenv("CART_BASE_URL", "http://cart-service:8000")
CHECKOUT_BASE_URL = os.getenv("CHECKOUT_BASE_URL", "http://checkout-service:8000")
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown-service")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "6.0"))
FAULT_CACHE_TTL_SEC = float(os.getenv("FAULT_CACHE_TTL_SEC", "0.75"))
TELEMETRY_DB_PATH = Path(os.getenv("TELEMETRY_DB_PATH", "/data/fault_lab.db"))

# Demo-only plaintext credentials — NOT suitable for production use.
DEFAULT_USERS = {
    "demo@triage.ai": {
        "password": "demo123",
        "name": "Demo User",
    },
    "ops@triage.ai": {
        "password": "ops123",
        "name": "Ops Engineer",
    },
}

SERVICE_FAULTS = {
    "auth-service": {
        "auth_failure": "Return intermittent auth failures and broken validation.",
        "latency_spike": "Add authentication latency to login and validate calls.",
    },
    "catalog-service": {
        "dependency_delay": "Make catalog queries slow like a degraded dependency.",
        "dependency_outage": "Return intermittent upstream-style catalog failures.",
    },
    "cart-service": {
        "memory_leak": "Leak process memory on each cart request.",
        "queue_congestion": "Build queue pressure and request backlog in cart flows.",
    },
    "checkout-service": {
        "cpu_exhaustion": "Burn CPU during checkout and raise checkout latency.",
        "cascading_failure": "Mix queue growth, latency, and failure propagation.",
    },
}

SCENARIO_PRESETS = {
    "healthy": {},
    "login_outage": {
        "auth-service": {
            "auth_failure": 0.95,
            "latency_spike": 0.60,
        }
    },
    "catalog_brownout": {
        "catalog-service": {
            "dependency_delay": 0.85,
            "dependency_outage": 0.45,
        }
    },
    "cart_memory_leak": {
        "cart-service": {
            "memory_leak": 0.90,
            "queue_congestion": 0.35,
        }
    },
    "checkout_cpu_hot": {
        "checkout-service": {
            "cpu_exhaustion": 0.95,
        }
    },
    "cascading_checkout_failure": {
        "catalog-service": {
            "dependency_delay": 0.60,
            "dependency_outage": 0.30,
        },
        "cart-service": {
            "queue_congestion": 0.85,
            "memory_leak": 0.45,
        },
        "checkout-service": {
            "cpu_exhaustion": 0.70,
            "cascading_failure": 0.95,
        },
        "auth-service": {
            "latency_spike": 0.40,
        },
    },
}
