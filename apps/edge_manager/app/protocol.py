import json
import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import zenoh


SCHEMA_VERSION = 1
DEFAULT_TOPIC_ROOT = "dt/edges"
MAX_CALIBRATION_BUNDLE_BYTES = 256 * 1024 * 1024
EDGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_edge_id(edge_id: str) -> str:
    if not EDGE_ID_PATTERN.fullmatch(edge_id):
        raise ValueError(
            "edge_id must be 1-64 characters using letters, numbers, '.', '_' or '-'"
        )
    return edge_id


@dataclass(frozen=True, slots=True)
class EdgeTopics:
    edge_id: str
    root: str = DEFAULT_TOPIC_ROOT

    def __post_init__(self) -> None:
        validate_edge_id(self.edge_id)
        normalized_root = self.root.strip("/")
        if not normalized_root:
            raise ValueError("topic root must not be empty")
        object.__setattr__(self, "root", normalized_root)

    @property
    def base(self) -> str:
        return f"{self.root}/{self.edge_id}"

    @property
    def status(self) -> str:
        return f"{self.base}/status"

    @property
    def inference(self) -> str:
        return f"{self.base}/inference"

    @property
    def command(self) -> str:
        return f"{self.base}/command"

    @property
    def config(self) -> str:
        return f"{self.base}/config"

    @property
    def ack(self) -> str:
        return f"{self.base}/ack"

    @property
    def cameras(self) -> str:
        return f"{self.base}/cameras"

    @property
    def calibration(self) -> str:
        return f"{self.base}/calibration"

    def calibration_chunk(self, request_id: str, index: int | str = "*") -> str:
        return f"{self.calibration}/{request_id}/chunks/{index}"


def make_zenoh_config(
    endpoint: str | None = None,
    config_path: str | None = None,
) -> zenoh.Config:
    if config_path:
        return zenoh.Config.from_file(config_path)
    if endpoint:
        if "/" not in endpoint:
            endpoint = f"tcp/{endpoint}"
        return zenoh.Config.from_json5(
            json.dumps(
                {
                    "mode": "client",
                    "connect": {"endpoints": [endpoint]},
                }
            )
        )
    return zenoh.Config()


def encode_json(message: dict[str, Any]) -> bytes:
    return json.dumps(
        message,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_json(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "to_bytes"):
        payload = payload.to_bytes()
    message = json.loads(bytes(payload).decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("message must be a JSON object")
    if message.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    return message


def camera_records_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and whitelist the only camera fields the server may persist."""
    if message.get("kind") != "camera_status":
        raise ValueError("unexpected message kind")
    data = message.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("cameras"), list):
        raise ValueError("camera status data.cameras must be a list")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(data["cameras"]):
        if not isinstance(raw, dict):
            raise ValueError(f"camera status item {index} must be an object")
        camera_id = str(raw.get("camera_id") or "")
        if not camera_id.isdigit() or int(camera_id) <= 0:
            raise ValueError(f"camera status item {index} has an invalid camera_id")
        camera_id = str(int(camera_id))
        if camera_id in seen:
            raise ValueError(f"duplicate camera_id: {camera_id}")
        seen.add(camera_id)
        if not isinstance(raw.get("exists"), bool) or not isinstance(
            raw.get("ping"), bool
        ):
            raise ValueError(f"camera/{camera_id} exists and ping must be booleans")

        calibration = raw.get("calibration")
        if calibration is None:
            calibration = {}
        if not isinstance(calibration, dict):
            raise ValueError(f"camera/{camera_id} calibration must be an object")
        sanitized_calibration = {
            key: deepcopy(calibration.get(key))
            for key in ("intrinsic", "extrinsic", "distortion_coefficients")
        }
        # Validate that nested calibration values can safely be represented as JSON.
        json.dumps(sanitized_calibration, allow_nan=False)
        records.append(
            {
                "camera_id": camera_id,
                "exists": raw["exists"],
                "ping": raw["ping"],
                "calibration": sanitized_calibration,
            }
        )
    return records


class ChunkCollector:
    def __init__(self, count: int, sha256: str, byte_count: int) -> None:
        if count <= 0 or byte_count <= 0 or byte_count > MAX_CALIBRATION_BUNDLE_BYTES:
            raise ValueError("invalid calibration transfer dimensions")
        self.expected_count = count
        self.expected_sha256 = sha256
        self.expected_size = byte_count
        self._chunks: dict[int, bytes] = {}

    def add(self, index: int, payload: bytes) -> None:
        if index < 0 or index >= self.expected_count:
            raise ValueError("chunk index is out of range")
        self._chunks.setdefault(index, bytes(payload))

    @property
    def complete(self) -> bool:
        return len(self._chunks) == self.expected_count

    def assemble(self) -> bytes:
        if not self.complete:
            raise ValueError("calibration transfer is incomplete")
        payload = b"".join(self._chunks[index] for index in range(self.expected_count))
        if len(payload) != self.expected_size:
            raise ValueError("calibration transfer size mismatch")
        if hashlib.sha256(payload).hexdigest() != self.expected_sha256:
            raise ValueError("calibration transfer checksum mismatch")
        return payload
