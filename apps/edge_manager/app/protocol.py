import json
import re
from dataclasses import dataclass
from typing import Any

import zenoh


SCHEMA_VERSION = 1
DEFAULT_TOPIC_ROOT = "dt/edges"
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
