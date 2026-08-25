"""Low-overhead JSONL telemetry for the server inference process."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import queue
import socket
import subprocess
import threading
import time
from typing import Any
from uuid import uuid4

import torch


logger = logging.getLogger(__name__)

TELEMETRY_SCHEMA_VERSION = 1
_STOP = object()


def utc_now_iso() -> str:
    """Return an RFC 3339 UTC timestamp suitable for lexical sorting."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def collect_git_metadata(project_root: Path) -> dict[str, Any]:
    """Collect reproducibility metadata without making Git a runtime dependency."""

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    revision = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "revision": revision,
        "branch": branch,
        "tracked_files_dirty": bool(status) if status is not None else None,
    }


def collect_runtime_metadata(
    device: torch.device,
    project_root: Path,
) -> dict[str, Any]:
    """Describe the software and accelerator used for a benchmark run."""
    gpu_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [properties.major, properties.minor],
                }
            )

    return {
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "process": {
            "pid": os.getpid(),
            "python_version": platform.python_version(),
        },
        "software": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "git": collect_git_metadata(project_root),
        },
        "accelerator": {
            "selected_device": str(device),
            "devices": gpu_devices,
        },
    }


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: float | int | None) -> None:
        if value is None:
            return
        numeric = float(value)
        self.count += 1
        self.total += numeric
        self.minimum = numeric if self.minimum is None else min(self.minimum, numeric)
        self.maximum = numeric if self.maximum is None else max(self.maximum, numeric)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else None,
            "min": self.minimum,
            "max": self.maximum,
        }


class TelemetryRecorder:
    """Write telemetry on a background thread so inference never waits on disk."""

    def __init__(
        self,
        root_dir: str | Path,
        run_metadata: dict[str, Any],
        *,
        queue_size: int = 2048,
        flush_every: int = 10,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")

        self.run_id = uuid4().hex
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(root_dir).expanduser().resolve() / (
            f"{stamp}_{self.run_id[:8]}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(queue_size)
        self._flush_every = flush_every
        self._closed = False
        self._write_error: BaseException | None = None
        self._dropped_records = 0
        self._started_wall = time.time()
        self._started_monotonic = time.monotonic()
        self._event_count = 0
        self._batch_count = 0
        self._persisted_batch_count = 0
        self._successful_batches = 0
        self._failed_batches = 0
        self._processing_ms = RunningStats()
        self._people = RunningStats()

        metadata = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at_utc": utc_now_iso(),
            **run_metadata,
        }
        self._write_json(self.run_dir / "run.json", metadata)

        self._worker = threading.Thread(
            target=self._write_loop,
            name="inference-telemetry-writer",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(
                value,
                output_file,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            output_file.write("\n")

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    def _write_loop(self) -> None:
        try:
            with self.events_path.open("a", encoding="utf-8") as output_file:
                pending_flush = 0
                while True:
                    item = self._queue.get()
                    if item is _STOP:
                        break
                    try:
                        serialized = json.dumps(
                            item,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                    except (TypeError, ValueError):
                        self._dropped_records += 1
                        logger.exception("Invalid telemetry record was dropped")
                        continue
                    output_file.write(serialized + "\n")
                    pending_flush += 1
                    if pending_flush >= self._flush_every:
                        output_file.flush()
                        pending_flush = 0
                output_file.flush()
        except BaseException as exc:
            self._write_error = exc
            logger.exception("Telemetry writer failed")

    def record_event(self, event: dict[str, Any]) -> bool:
        if self._closed or self._write_error is not None:
            self._dropped_records += 1
            return False

        record = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "recorded_at_utc": utc_now_iso(),
            **event,
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_records += 1
            return False
        self._event_count += 1
        return True

    def record_batch(
        self,
        event: dict[str, Any],
        *,
        persist: bool = True,
    ) -> bool:
        """Aggregate every batch while writing only selected batch records."""
        self._batch_count += 1
        if event.get("status") == "ok":
            self._successful_batches += 1
        else:
            self._failed_batches += 1
        timings = event.get("timings_ms") or {}
        workload = event.get("workload") or {}
        self._processing_ms.add(timings.get("processing_total"))
        self._people.add(workload.get("output_people"))
        if not persist:
            return True
        self._persisted_batch_count += 1
        return self.record_event({"event": "batch", **event})

    def close(self, extra_summary: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(_STOP, timeout=2)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._dropped_records += 1
                self._queue.put_nowait(_STOP)
            except queue.Empty:
                self._queue.put_nowait(_STOP)
        self._worker.join(timeout=5)
        if self._worker.is_alive() and self._write_error is None:
            self._write_error = TimeoutError("Telemetry writer did not stop")

        duration = max(0.0, time.monotonic() - self._started_monotonic)
        summary = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at_unix_seconds": self._started_wall,
            "ended_at_utc": utc_now_iso(),
            "duration_seconds": duration,
            "events_enqueued": self._event_count,
            "telemetry_records_dropped": self._dropped_records,
            "writer_error": (
                None if self._write_error is None else repr(self._write_error)
            ),
            "batches": {
                "total": self._batch_count,
                "successful": self._successful_batches,
                "failed": self._failed_batches,
                "persisted": self._persisted_batch_count,
                "successful_batches_per_second": (
                    self._successful_batches / duration if duration else None
                ),
                "processing_ms": self._processing_ms.to_dict(),
                "output_people": self._people.to_dict(),
            },
            **(extra_summary or {}),
        }
        try:
            self._write_json(self.summary_path, summary)
        except Exception:
            logger.exception("Failed to write telemetry summary")


class ResourceSampler:
    """Sample host, process, network, and CUDA counters using optional APIs."""

    def __init__(
        self,
        device: torch.device,
        *,
        interval_seconds: float = 1.0,
        include_device_metrics: bool = False,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must not be negative")
        self.device = device
        self.interval_seconds = interval_seconds
        self.include_device_metrics = include_device_metrics
        self._last_sample_time: float | None = None
        self._last_process_cpu: float | None = None
        self._last_system_cpu: tuple[int, int] | None = None
        self._last_network: tuple[int, int] | None = None
        self._last_io: tuple[int, int] | None = None

    @staticmethod
    def _read_key_values(path: Path) -> dict[str, int]:
        values = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                key, raw_value = line.split(":", 1)
                token = raw_value.strip().split()[0]
                values[key] = int(token)
        except (OSError, ValueError, IndexError):
            return {}
        return values

    @staticmethod
    def _read_system_cpu() -> tuple[int, int] | None:
        try:
            fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[
                0
            ].split()
            values = [int(value) for value in fields[1:]]
        except (OSError, ValueError, IndexError):
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    @staticmethod
    def _read_network() -> tuple[int, int] | None:
        received = 0
        transmitted = 0
        try:
            lines = Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[
                2:
            ]
            for line in lines:
                interface, raw_values = line.split(":", 1)
                if interface.strip() == "lo":
                    continue
                fields = raw_values.split()
                received += int(fields[0])
                transmitted += int(fields[8])
        except (OSError, ValueError, IndexError):
            return None
        return received, transmitted

    @staticmethod
    def _read_rss_bytes() -> int | None:
        try:
            resident_pages = int(
                Path("/proc/self/statm").read_text(encoding="utf-8").split()[1]
            )
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None

    def _gpu_sample(self) -> dict[str, Any] | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device()
        result: dict[str, Any] = {"device_index": index}
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            result.update(
                {
                    "memory_free_bytes": int(free_bytes),
                    "memory_total_bytes": int(total_bytes),
                    "process_memory_allocated_bytes": int(
                        torch.cuda.memory_allocated(index)
                    ),
                    "process_memory_reserved_bytes": int(
                        torch.cuda.memory_reserved(index)
                    ),
                    "process_max_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(index)
                    ),
                }
            )
        except Exception:
            pass

        if not self.include_device_metrics:
            return result

        optional_metrics = (
            ("utilization_percent", "utilization", 1.0),
            ("memory_utilization_percent", "memory_usage", 1.0),
            ("temperature_celsius", "temperature", 1.0),
            ("power_watts", "power_draw", 0.001),
            ("clock_rate_mhz", "clock_rate", 1.0),
        )
        for output_name, function_name, scale in optional_metrics:
            function = getattr(torch.cuda, function_name, None)
            if function is None:
                continue
            try:
                result[output_name] = float(function(index)) * scale
            except Exception:
                continue
        return result

    def sample(self, *, force: bool = False) -> dict[str, Any] | None:
        now = time.monotonic()
        if (
            not force
            and self._last_sample_time is not None
            and now - self._last_sample_time < self.interval_seconds
        ):
            return None

        elapsed = (
            None if self._last_sample_time is None else now - self._last_sample_time
        )
        process_cpu = time.process_time()
        system_cpu = self._read_system_cpu()
        network = self._read_network()
        io_values = self._read_key_values(Path("/proc/self/io"))
        process_io = (
            io_values.get("read_bytes", 0),
            io_values.get("write_bytes", 0),
        )

        process_cpu_percent = None
        if elapsed and self._last_process_cpu is not None:
            process_cpu_percent = (
                (process_cpu - self._last_process_cpu) / elapsed * 100.0
            )

        system_cpu_percent = None
        if system_cpu is not None and self._last_system_cpu is not None:
            total_delta = system_cpu[0] - self._last_system_cpu[0]
            idle_delta = system_cpu[1] - self._last_system_cpu[1]
            if total_delta > 0:
                system_cpu_percent = (1.0 - idle_delta / total_delta) * 100.0

        network_sample: dict[str, Any] | None = None
        if network is not None:
            network_sample = {
                "scope": "host_non_loopback_interfaces",
                "received_bytes_total": network[0],
                "transmitted_bytes_total": network[1],
                "received_mbps": None,
                "transmitted_mbps": None,
            }
            if elapsed and self._last_network is not None:
                network_sample["received_mbps"] = max(
                    0.0, (network[0] - self._last_network[0]) * 8 / elapsed / 1e6
                )
                network_sample["transmitted_mbps"] = max(
                    0.0, (network[1] - self._last_network[1]) * 8 / elapsed / 1e6
                )

        io_sample = {
            "read_bytes_total": process_io[0],
            "write_bytes_total": process_io[1],
            "read_mbps": None,
            "write_mbps": None,
        }
        if elapsed and self._last_io is not None:
            io_sample["read_mbps"] = max(
                0.0, (process_io[0] - self._last_io[0]) / elapsed / 1e6
            )
            io_sample["write_mbps"] = max(
                0.0, (process_io[1] - self._last_io[1]) / elapsed / 1e6
            )

        memory = self._read_key_values(Path("/proc/meminfo"))
        memory_total = memory.get("MemTotal")
        memory_available = memory.get("MemAvailable")
        try:
            load_average = list(os.getloadavg())
        except OSError:
            load_average = None

        self._last_sample_time = now
        self._last_process_cpu = process_cpu
        self._last_system_cpu = system_cpu
        self._last_network = network
        self._last_io = process_io

        cpu_count = os.cpu_count() or 1
        rss_bytes = self._read_rss_bytes()
        return {
            "sample_interval_seconds": elapsed,
            "process": {
                "cpu_percent": process_cpu_percent,
                "cpu_percent_of_host_capacity": (
                    None
                    if process_cpu_percent is None
                    else process_cpu_percent / cpu_count
                ),
                "rss_bytes": rss_bytes,
                "io": io_sample,
            },
            "system": {
                "cpu_percent": system_cpu_percent,
                "load_average": load_average,
                "memory_total_bytes": (
                    None if memory_total is None else memory_total * 1024
                ),
                "memory_available_bytes": (
                    None if memory_available is None else memory_available * 1024
                ),
                "memory_used_percent": (
                    None
                    if not memory_total or memory_available is None
                    else (memory_total - memory_available) / memory_total * 100.0
                ),
            },
            "network": network_sample,
            "gpu": self._gpu_sample(),
        }
