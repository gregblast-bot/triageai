from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil

from .clients import request_json
from .config import CONTROL_BASE_URL, FAULT_CACHE_TTL_SEC


# Total logical cores on the host. Used to normalize `Process.cpu_percent`
# (which can exceed 100 on multi-core) to host-wide %. Memoized because
# `cpu_count` returns a stable value and reading it is cheap but not free.
_CPU_COUNT = max(1, psutil.cpu_count(logical=True) or 1)


@dataclass
class RequestContext:
    start_time: float
    queue_depth: float


class ServiceRuntime:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._lock = threading.Lock()
        self._inflight = 0
        self._queue_pressure = 0.0
        self._memory_leak_blobs: list[bytearray] = []
        self._fault_cache: dict[str, float] = {}
        self._fault_cache_time = 0.0
        self._process = psutil.Process()
        self._process.cpu_percent(None)
        # Seed disk IO so we can emit deltas per request instead of cumulative
        # counters. psutil can't always get io_counters for a user-space
        # process (macOS, some container setups), so fall back to zero there.
        self._last_io_bytes = self._read_io_bytes()

    def _read_io_bytes(self) -> int:
        try:
            counters = self._process.io_counters()
            return int(counters.read_bytes + counters.write_bytes)
        except (psutil.AccessDenied, AttributeError, NotImplementedError, OSError):
            return 0

    def _open_socket_count(self) -> int:
        try:
            return len(self._process.net_connections(kind="inet"))
        except (psutil.AccessDenied, NotImplementedError, OSError):
            return 0

    def begin_request(self) -> RequestContext:
        start = time.perf_counter()
        with self._lock:
            self._inflight += 1
            queue_depth = max(0.0, self._inflight - 1 + self._queue_pressure)
        return RequestContext(start_time=start, queue_depth=queue_depth)

    def add_queue_pressure(self, amount: float) -> None:
        with self._lock:
            self._queue_pressure = min(24.0, self._queue_pressure + amount)

    def decay_queue_pressure(self) -> None:
        with self._lock:
            self._queue_pressure = max(0.0, self._queue_pressure * 0.72 - 0.25)

    def leak_memory_mb(self, amount_mb: float) -> None:
        bytes_to_add = int(max(0.0, amount_mb) * 1024 * 1024)
        if bytes_to_add <= 0:
            return
        self._memory_leak_blobs.append(bytearray(bytes_to_add))
        max_blobs = 48
        if len(self._memory_leak_blobs) > max_blobs:
            self._memory_leak_blobs = self._memory_leak_blobs[-max_blobs:]

    async def get_faults(self) -> dict[str, float]:
        now = time.monotonic()
        if now - self._fault_cache_time <= FAULT_CACHE_TTL_SEC:
            return self._fault_cache

        status_code, payload = await request_json(
            "GET",
            f"{CONTROL_BASE_URL}/api/faults/{self.service_name}",
            retries=1,
        )
        if status_code == 200:
            self._fault_cache = payload.get("faults", {})
            self._fault_cache_time = now
        return self._fault_cache

    async def emit_telemetry(
        self,
        context: RequestContext,
        *,
        path: str,
        status_code: int,
        auth_error: bool = False,
        extra_cpu: float = 0.0,
    ) -> None:
        latency_ms = (time.perf_counter() - context.start_time) * 1000.0
        raw_cpu = self._process.cpu_percent(None)
        cpu_pct = max(0.0, raw_cpu / _CPU_COUNT + extra_cpu)
        memory_mb = self._process.memory_info().rss / (1024 * 1024)

        # Delta since the last emission. It's process-wide, not request-scoped
        # (psutil can't tell us that), but it still surfaces IO pressure nicely
        # once we bucket a batch of events together.
        current_io = self._read_io_bytes()
        io_delta = max(0, current_io - self._last_io_bytes)
        self._last_io_bytes = current_io
        socket_count = self._open_socket_count()

        payload = {
            "service": self.service_name,
            "path": path,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "cpu_pct": cpu_pct,
            "memory_mb": memory_mb,
            "queue_depth": max(context.queue_depth, self._queue_pressure),
            "error": status_code >= 400,
            "auth_error": auth_error,
            "disk_io_bytes": io_delta,
            "socket_count": socket_count,
        }

        try:
            await request_json(
                "POST",
                f"{CONTROL_BASE_URL}/api/telemetry/events",
                json=payload,
                retries=1,
            )
        finally:
            with self._lock:
                self._inflight = max(0, self._inflight - 1)
            self.decay_queue_pressure()


def busy_wait(seconds: float) -> None:
    """Spin the CPU for a bit—crude but enough to show a cpu-style fault in the demo."""
    deadline = time.perf_counter() + max(0.0, seconds)
    while time.perf_counter() < deadline:
        pass


async def async_busy_wait(seconds: float) -> None:
    """Same idea as busy_wait, but in a thread pool so the event loop stays responsive."""
    import asyncio

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, busy_wait, seconds)
