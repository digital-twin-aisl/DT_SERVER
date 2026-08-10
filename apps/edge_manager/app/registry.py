import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class EdgeRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._thread_lock = threading.RLock()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"version": 1, "edges": {}}

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("edges"), dict):
            raise ValueError(f"invalid edge registry: {self.path}")
        return data

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _with_lock(
        self,
        operation: Callable[[dict[str, Any]], tuple[T, bool]],
    ) -> T:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                data = self._load_unlocked()
                result, changed = operation(data)
                if changed:
                    self._save_unlocked(data)
                return result
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def snapshot(self) -> dict[str, Any]:
        return self._with_lock(lambda data: (data, False))

    def observe_status(
        self,
        edge_id: str,
        message: dict[str, Any],
        received_at: float | None = None,
    ) -> tuple[bool, bool, dict[str, Any]]:
        received_at = received_at or time.time()

        def update(
            data: dict[str, Any],
        ) -> tuple[tuple[bool, bool, dict[str, Any]], bool]:
            edges = data["edges"]
            created = edge_id not in edges
            was_online = bool(edges.get(edge_id, {}).get("online"))
            record = edges.setdefault(
                edge_id,
                {
                    "edge_id": edge_id,
                    "display_name": edge_id,
                    "approved": False,
                    "online": True,
                    "first_seen_at": received_at,
                },
            )
            record.pop("enabled", None)
            record.update(
                {
                    "online": True,
                    "last_seen_at": received_at,
                    "last_status": message.get("data") or {},
                    "edge_sent_at": message.get("sent_at"),
                }
            )
            came_online = not created and not was_online
            return (created, came_online, dict(record)), True

        return self._with_lock(update)

    def mark_all_offline(self) -> list[str]:
        """Reset ephemeral connection state when the manager starts."""

        def update(data: dict[str, Any]) -> tuple[list[str], bool]:
            changed = []
            migrated = False
            for edge_id, record in data["edges"].items():
                if record.pop("enabled", None) is not None:
                    migrated = True
                if record.get("online"):
                    record["online"] = False
                    changed.append(edge_id)
            return changed, bool(changed) or migrated

        return self._with_lock(update)

    def update(self, edge_id: str, **fields: Any) -> dict[str, Any]:
        def update_record(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            try:
                record = data["edges"][edge_id]
            except KeyError as exc:
                raise KeyError(f"unknown edge: {edge_id}") from exc
            record.update(fields)
            return dict(record), True

        return self._with_lock(update_record)

    def remove(self, edge_id: str) -> None:
        def remove_record(data: dict[str, Any]) -> tuple[None, bool]:
            try:
                del data["edges"][edge_id]
            except KeyError as exc:
                raise KeyError(f"unknown edge: {edge_id}") from exc
            return None, True

        self._with_lock(remove_record)

    def mark_offline(self, offline_after: float, now: float | None = None) -> list[str]:
        now = now or time.time()

        def update(data: dict[str, Any]) -> tuple[list[str], bool]:
            changed = []
            for edge_id, record in data["edges"].items():
                last_seen = float(record.get("last_seen_at", 0))
                if record.get("online") and now - last_seen > offline_after:
                    record["online"] = False
                    changed.append(edge_id)
            return changed, bool(changed)

        return self._with_lock(update)

    def record_ack(self, edge_id: str, message: dict[str, Any]) -> None:
        def update(data: dict[str, Any]) -> tuple[None, bool]:
            record = data["edges"].get(edge_id)
            if record is None:
                return None, False
            record["last_ack"] = message
            return None, True

        self._with_lock(update)
