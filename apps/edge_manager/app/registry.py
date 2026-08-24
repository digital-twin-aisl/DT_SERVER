import fcntl
from copy import deepcopy
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
        received_at = time.time() if received_at is None else received_at

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
            display_name = str(
                (message.get("data") or {}).get("display_name") or ""
            ).strip()
            if display_name:
                record["display_name"] = display_name
            came_online = not created and not was_online
            return (created, came_online, dict(record)), True

        return self._with_lock(update)

    def observe_cameras(
        self,
        edge_id: str,
        cameras: list[dict[str, Any]],
        *,
        edge_sent_at: float | None = None,
        received_at: float | None = None,
    ) -> tuple[bool, bool, dict[str, Any]]:
        """Store an authoritative, sanitized camera snapshot for one edge."""
        received_at = time.time() if received_at is None else received_at

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
            previous_value = record.get("cameras") or {}
            previous = previous_value if isinstance(previous_value, dict) else {}
            current = {camera["camera_id"]: dict(camera) for camera in cameras}
            for camera_id, camera in current.items():
                old_camera = previous.get(camera_id)
                if isinstance(old_camera, dict) and old_camera.get(
                    "service_metadata"
                ) is not None:
                    camera["service_metadata"] = deepcopy(
                        old_camera["service_metadata"]
                    )
            for camera_id, old_camera in previous.items():
                if camera_id not in current:
                    current[camera_id] = {
                        "camera_id": camera_id,
                        "exists": False,
                        "ping": False,
                        "calibration": old_camera.get("calibration")
                        or {
                            "intrinsic": None,
                            "extrinsic": None,
                            "distortion_coefficients": None,
                        },
                    }
            record.update(
                {
                    "online": True,
                    "last_seen_at": received_at,
                    "cameras": current,
                    "last_camera_update_at": received_at,
                    "camera_sent_at": edge_sent_at,
                }
            )
            return (created, not created and not was_online, dict(record)), True

        return self._with_lock(update)

    @staticmethod
    def _clear_camera_pings(record: dict[str, Any]) -> bool:
        changed = False
        cameras = record.get("cameras") or {}
        values = cameras.values() if isinstance(cameras, dict) else cameras
        for camera in values:
            if not isinstance(camera, dict):
                continue
            if camera.get("ping"):
                camera["ping"] = False
                changed = True
        return changed

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
                if self._clear_camera_pings(record):
                    migrated = True
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
                    self._clear_camera_pings(record)
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

    def record_calibration_result(
        self,
        edge_id: str,
        result: dict[str, Any],
        *,
        result_path: str,
        completed_at: float | None = None,
        edge_sync_status: str = "pending",
    ) -> dict[str, Any]:
        """Persist detailed camera calibration and an edge-level run summary."""
        json.dumps(result, allow_nan=False)
        completed_at = time.time() if completed_at is None else completed_at

        def update(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            try:
                record = data["edges"][edge_id]
            except KeyError as exc:
                raise KeyError(f"unknown edge: {edge_id}") from exc
            result_cameras = result.get("cameras")
            if not isinstance(result_cameras, list) or not result_cameras:
                raise ValueError("calibration result cameras must be a non-empty list")
            cameras = record.setdefault("cameras", {})
            if not isinstance(cameras, dict):
                raise ValueError(f"edge {edge_id} cameras must be an object")

            updated_ids: list[str] = []
            for camera in result_cameras:
                if not isinstance(camera, dict):
                    raise ValueError("calibration result camera must be an object")
                camera_id = str(camera.get("edge_camera_id") or "")
                expected_key = f"{edge_id}/camera/{camera_id}"
                if camera.get("camera_id") != expected_key:
                    raise ValueError(
                        f"calibration camera key does not match edge: {camera.get('camera_id')}"
                    )
                current = cameras.get(camera_id)
                if not isinstance(current, dict) or not current.get("exists"):
                    raise ValueError(f"unknown registered camera: camera/{camera_id}")
                calibration = current.setdefault("calibration", {})
                existing_intrinsic = calibration.get("intrinsic")
                intrinsic = (
                    deepcopy(existing_intrinsic)
                    if isinstance(existing_intrinsic, dict)
                    else {}
                )
                intrinsic.update(
                    {
                        "camera_matrix": deepcopy(camera.get("camera_matrix")),
                        "undistorted_camera_matrix": deepcopy(
                            camera.get("undistorted_camera_matrix")
                        ),
                        "image_size": deepcopy(camera.get("source_image_size")),
                    }
                )
                calibration["intrinsic"] = intrinsic
                calibration["distortion_coefficients"] = deepcopy(
                    camera.get("distortion_coefficients")
                )
                calibration["extrinsic"] = {
                    "schema_version": 1,
                    "request_id": (result.get("input") or {}).get("request_id"),
                    "calibrated_at": completed_at,
                    "marker_tree": result.get("marker_tree"),
                    "coordinate_convention": deepcopy(
                        result.get("coordinate_convention")
                    ),
                    "world_to_camera": deepcopy(camera.get("world_to_camera")),
                    "camera_to_world": deepcopy(camera.get("camera_to_world")),
                    "position_m": deepcopy(camera.get("position_m")),
                    "alignment": deepcopy(result.get("alignment")),
                }
                current["service_metadata"] = {
                    "name": camera.get("name"),
                    "location": camera.get("location"),
                    "twin_id": camera.get("twin_id"),
                }
                updated_ids.append(camera_id)

            record["last_calibration"] = {
                "schema_version": 1,
                "request_id": (result.get("input") or {}).get("request_id"),
                "completed_at": completed_at,
                "result_path": result_path,
                "marker_tree": result.get("marker_tree"),
                "checkpoint_sha256": (result.get("input") or {}).get(
                    "checkpoint_sha256"
                ),
                "reference_video_frame_indices": deepcopy(
                    (result.get("input") or {}).get(
                        "reference_video_frame_indices"
                    )
                ),
                "alignment": deepcopy(result.get("alignment")),
                "camera_ids": sorted(updated_ids, key=int),
                "edge_sync": {
                    "status": edge_sync_status,
                    "updated_at": completed_at,
                    "error": None,
                },
            }
            return dict(record), True

        return self._with_lock(update)

    def record_calibration_sync(
        self,
        edge_id: str,
        request_id: str,
        *,
        success: bool,
        error: str | None = None,
        updated_at: float | None = None,
    ) -> None:
        updated_at = time.time() if updated_at is None else updated_at

        def update(data: dict[str, Any]) -> tuple[None, bool]:
            record = data["edges"].get(edge_id)
            if record is None:
                raise KeyError(f"unknown edge: {edge_id}")
            calibration = record.get("last_calibration")
            if not isinstance(calibration, dict) or calibration.get(
                "request_id"
            ) != request_id:
                raise ValueError(
                    f"calibration request is no longer current: {request_id}"
                )
            calibration["edge_sync"] = {
                "status": "applied" if success else "failed",
                "updated_at": updated_at,
                "error": error,
            }
            return None, True

        self._with_lock(update)
