from __future__ import annotations

import os
from pathlib import Path


CONTROL_BASE_URL = os.getenv("CONTROL_BASE_URL", "http://control-plane:8000")
# Public (browser-reachable) URL of the control plane, used only for links
# emitted into HTML. Inside docker, services talk to control-plane via
# CONTROL_BASE_URL, but browsers on the host need the host-mapped port.
CONTROL_PUBLIC_URL = os.getenv("CONTROL_PUBLIC_URL", "http://localhost:8001")
AUTH_BASE_URL = os.getenv("AUTH_BASE_URL", "http://auth-service:8000")
CATALOG_BASE_URL = os.getenv("CATALOG_BASE_URL", "http://catalog-service:8000")
CART_BASE_URL = os.getenv("CART_BASE_URL", "http://cart-service:8000")
CHECKOUT_BASE_URL = os.getenv("CHECKOUT_BASE_URL", "http://checkout-service:8000")
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown-service")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "6.0"))
FAULT_CACHE_TTL_SEC = float(os.getenv("FAULT_CACHE_TTL_SEC", "0.75"))
TELEMETRY_DB_PATH = Path(os.getenv("TELEMETRY_DB_PATH", "/data/fault_lab.db"))

# Bucket window size in seconds. Training features are computed over 120
# samples, so a 60-second bucket matches the "1 minute" semantic in the
# training data. Lower this to shorten the observation window.
TELEMETRY_BUCKET_SEC = float(os.getenv("TELEMETRY_BUCKET_SEC", "60"))

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


# Expected model predictions per fault-lab scenario. The TriageAI models are
# trained on RCAEval labels (fault_type in {healthy, cpu, mem, disk, delay,
# loss, socket}) and RCAEval service names (cartservice, checkoutservice...).
# This dictionary is the "ground truth" used when the control plane reports
# what a given scenario *should* produce, so the UI can show expected vs.
# predicted.
SCENARIO_EXPECTATION = {
    "healthy": {"fault_type": "healthy", "root_cause_service": "none"},
    "login_outage": {"fault_type": "loss", "root_cause_service": "adservice"},
    "catalog_brownout": {"fault_type": "delay", "root_cause_service": "productcatalogservice"},
    "cart_memory_leak": {"fault_type": "mem", "root_cause_service": "cartservice"},
    "checkout_cpu_hot": {"fault_type": "cpu", "root_cause_service": "checkoutservice"},
    "cascading_checkout_failure": {
        "fault_type": "delay",
        "root_cause_service": "checkoutservice",
    },
}

# When fault-lab emits telemetry it tags each event with its internal service
# name (e.g. "cart-service"). The training data uses RCAEval canonical names.
# This map is applied at the point we need to expose a "predicted service"
# comparison. It is intentionally a display-only concept; the ML models still
# see raw internal names only via the text channel.
FAULT_LAB_SERVICE_ALIAS = {
    "auth-service": "adservice",
    "cart-service": "cartservice",
    "catalog-service": "productcatalogservice",
    "checkout-service": "checkoutservice",
}
