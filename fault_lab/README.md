# Fault Lab

This is a real local microservice stack you can run next to TriageAI. It gives you:

- a working storefront web app
- real HTTP backend services
- deliberate fault injection
- telemetry export in the same metric schema TriageAI expects

## Services

- `storefront`: user-facing web app
- `auth-service`: login and token validation
- `catalog-service`: product catalog and search
- `cart-service`: cart storage
- `checkout-service`: cross-service checkout flow
- `control-plane`: fault injection + telemetry aggregation

## Fault Scenarios

- `healthy`
- `login_outage`
- `catalog_brownout`
- `cart_memory_leak`
- `checkout_cpu_hot`
- `cascading_checkout_failure`

## Run

```bash
docker-compose -f fault_lab/docker-compose.yml up --build
```

Then open:

- Storefront: `http://localhost:8090`
- Control plane API: `http://localhost:8001`

## Export Telemetry For TriageAI

The admin page includes a download link for the latest TriageAI-compatible CSV.

You can also download it directly:

```bash
curl http://localhost:8001/api/telemetry/window.csv?limit=120 -o fault_lab_window.csv
```
