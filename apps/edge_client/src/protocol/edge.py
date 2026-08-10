import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = 1
DEFAULT_TOPIC_ROOT = "dt/edges"
DEFAULT_IDENTITY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "edge.local.json"
)
EDGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_edge_id(edge_id: str) -> str:
    if not EDGE_ID_PATTERN.fullmatch(edge_id):
        raise ValueError(
            "edge_id must be 1-64 characters using letters, numbers, '.', '_' or '-'"
        )
    return edge_id


def load_or_create_edge_id(
    path: str | Path = DEFAULT_IDENTITY_PATH,
    requested_edge_id: str | None = None,
) -> str:
    identity_path = Path(path)
    if identity_path.exists():
        data = json.loads(identity_path.read_text(encoding="utf-8"))
        edge_id = validate_edge_id(str(data["edge_id"]))
        if requested_edge_id is not None and requested_edge_id != edge_id:
            raise ValueError(
                f"stored edge_id is {edge_id!r}; refusing requested {requested_edge_id!r}"
            )
        return edge_id

    edge_id = validate_edge_id(requested_edge_id or f"edge-{uuid4().hex[:12]}")
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(
        json.dumps({"edge_id": edge_id}, indent=2) + "\n",
        encoding="utf-8",
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
    def ack(self) -> str:
        return f"{self.base}/ack"


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
