import argparse
import os
from pathlib import Path
import platform
import re
import signal
import socket
import sys
import threading
import time
from typing import Any

import zenoh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.edge_client.src.protocol.edge import (  # noqa: E402
    DEFAULT_IDENTITY_PATH,
    DEFAULT_TOPIC_ROOT,
    SCHEMA_VERSION,
    EdgeTopics,
    decode_json,
    encode_json,
    load_edge_metadata,
    load_or_create_edge_id,
    save_edge_metadata,
)
from apps.edge_client.src.protocol.zenoh import make_zenoh_config  # noqa: E402
from apps.edge_client.src.camera_status import (  # noqa: E402
    DEFAULT_CAMERA_PING_TIMEOUT,
    build_camera_status_message,
)
from dt_common.calibration.protocol import (  # noqa: E402
    payload_sha256,
    split_payload,
)
from apps.edge_client.src.calibration_features import capture_and_encode  # noqa: E402
from apps.edge_client.src.calibration_result import (  # noqa: E402
    apply_calibration_result,
)


AGENT_VERSION = "0.1.0"


class EdgeAgent:
    def __init__(
        self,
        edge_id: str,
        endpoint: str | None,
        config_path: str | None,
        topic_root: str,
        heartbeat_interval: float,
        camera_count: int,
        identity_path: str | Path,
        camera_config: str | Path,
        calibration_checkpoint: str | Path | None,
        calibration_device: str,
        calibration_image_size: int,
        camera_ping_timeout: float,
    ) -> None:
        self.edge_id = edge_id
        self.endpoint = endpoint
        self.config_path = config_path
        self.topics = EdgeTopics(edge_id, topic_root)
        self.heartbeat_interval = heartbeat_interval
        self.camera_count = camera_count
        self.identity_path = Path(identity_path)
        self.camera_config = Path(camera_config)
        self.calibration_checkpoint = (
            Path(calibration_checkpoint).expanduser()
            if calibration_checkpoint
            else None
        )
        self.calibration_device = calibration_device
        self.calibration_image_size = calibration_image_size
        self._calibration_lock = threading.Lock()
        self.camera_ping_timeout = camera_ping_timeout
        self._camera_status_lock = threading.Lock()
        self.started_at = time.time()
        self.stop_event = threading.Event()
        self._session: Any = None

    def _status(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "status",
            "edge_id": self.edge_id,
            "sent_at": time.time(),
            "data": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "agent_version": AGENT_VERSION,
                "camera_count": self.camera_count,
                "agent_started_at": self.started_at,
                "zenoh_endpoint": self.endpoint,
                "topic_root": self.topics.root,
            },
        }

    def _publish_ack(
        self,
        command_id: str,
        command: str,
        success: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        kind: str = "command_ack",
    ) -> None:
        if self._session is None:
            return
        message = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "edge_id": self.edge_id,
            "command": command,
            "success": success,
            "completed_at": time.time(),
            "result": result,
            "error": error,
        }
        message["config_id" if kind == "config_ack" else "command_id"] = command_id
        self._session.put(self.topics.ack, encode_json(message))

    def _publish_camera_status(self, trigger: str) -> int:
        with self._camera_status_lock:
            message = build_camera_status_message(
                self.edge_id,
                self.camera_config,
                ping_timeout=self.camera_ping_timeout,
            )
            cameras = message["data"]["cameras"]
            self.camera_count = len(cameras)
            if self._session is None:
                raise RuntimeError("Zenoh session is not connected")
            self._session.put(self.topics.cameras, encode_json(message))
            print(
                f"[cameras] published count={len(cameras)} trigger={trigger}",
                flush=True,
            )
            return len(cameras)

    def _handle_ping(self, command_id: str) -> None:
        try:
            camera_count = self._publish_camera_status("manager_heartbeat")
            self._publish_ack(
                command_id,
                "ping",
                True,
                {
                    "message": "pong",
                    "received_at": time.time(),
                    "camera_count": camera_count,
                },
            )
        except Exception as exc:
            self._publish_ack(command_id, "ping", False, error=str(exc))
            print(
                f"[cameras] heartbeat sync failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _on_command(self, sample: Any) -> None:
        try:
            message = decode_json(sample.payload)
            if message.get("kind") != "command":
                raise ValueError("unexpected message kind")
            if message.get("edge_id") != self.edge_id:
                raise ValueError("command edge_id does not match")

            command_id = str(message["command_id"])
            command = str(message["command"])
            print(f"[command] id={command_id} command={command}", flush=True)

            if command == "ping":
                worker = threading.Thread(
                    target=self._handle_ping,
                    args=(command_id,),
                    daemon=True,
                )
                worker.start()
            elif command == "info":
                self._publish_ack(command_id, command, True, self._status()["data"])
            elif command == "shutdown":
                self._publish_ack(command_id, command, True, {"message": "stopping"})
                self.stop_event.set()
            elif command == "capture_calibration_features":
                parameters = message.get("parameters") or {}
                if not isinstance(parameters, dict):
                    raise ValueError("command parameters must be an object")
                worker = threading.Thread(
                    target=self._capture_calibration_features,
                    args=(command_id, parameters),
                    daemon=True,
                )
                worker.start()
            elif command == "apply_calibration_result":
                parameters = message.get("parameters") or {}
                if not isinstance(parameters, dict):
                    raise ValueError("command parameters must be an object")
                worker = threading.Thread(
                    target=self._apply_calibration_result,
                    args=(command_id, parameters),
                    daemon=True,
                )
                worker.start()
            else:
                self._publish_ack(
                    command_id,
                    command,
                    False,
                    error=f"unsupported command: {command}",
                )
        except Exception as exc:
            print(f"[command] invalid message: {exc}", file=sys.stderr, flush=True)

    def _capture_calibration_features(
        self, command_id: str, parameters: dict[str, Any]
    ) -> None:
        if not self._calibration_lock.acquire(blocking=False):
            self._publish_ack(
                command_id,
                "capture_calibration_features",
                False,
                error="another calibration capture is already running",
            )
            return
        try:
            if not load_edge_metadata(self.identity_path).get("approved"):
                raise PermissionError("edge is not approved for calibration capture")
            checkpoint = self.calibration_checkpoint
            if checkpoint is None:
                raise ValueError(
                    "calibration checkpoint is not configured; pass "
                    "--calibration-checkpoint when starting the agent"
                )
            request_id = str(parameters.get("request_id") or command_id)
            if not re.fullmatch(r"[A-Fa-f0-9]{16,64}", request_id):
                raise ValueError("invalid calibration request_id")
            payload = capture_and_encode(
                self.camera_config,
                checkpoint,
                self.edge_id,
                request_id,
                device=self.calibration_device,
                image_size=self.calibration_image_size,
            )
            chunks = split_payload(payload)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "kind": "calibration_features",
                "edge_id": self.edge_id,
                "request_id": request_id,
                "chunk_count": len(chunks),
                "byte_count": len(payload),
                "sha256": payload_sha256(payload),
                "created_at": time.time(),
            }
            self._session.put(self.topics.calibration, encode_json(manifest))
            for index, chunk in enumerate(chunks):
                self._session.put(
                    self.topics.calibration_chunk(request_id, index), chunk
                )
            self._publish_ack(
                command_id,
                "capture_calibration_features",
                True,
                {
                    "request_id": request_id,
                    "camera_config": str(self.camera_config),
                    "chunk_count": len(chunks),
                    "byte_count": len(payload),
                },
            )
        except Exception as exc:
            self._publish_ack(
                command_id,
                "capture_calibration_features",
                False,
                error=str(exc),
            )
            print(f"[calibration] failed: {exc}", file=sys.stderr, flush=True)
        finally:
            self._calibration_lock.release()

    def _apply_calibration_result(
        self, command_id: str, parameters: dict[str, Any]
    ) -> None:
        try:
            if not load_edge_metadata(self.identity_path).get("approved"):
                raise PermissionError("edge is not approved for calibration updates")
            camera_ids = apply_calibration_result(
                self.camera_config,
                self.edge_id,
                parameters,
            )
            self._publish_ack(
                command_id,
                "apply_calibration_result",
                True,
                {
                    "request_id": parameters.get("request_id"),
                    "saved_camera_ids": camera_ids,
                    "camera_config": str(self.camera_config),
                },
            )
        except Exception as exc:
            self._publish_ack(
                command_id,
                "apply_calibration_result",
                False,
                error=str(exc),
            )
            print(
                f"[calibration] result apply failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return
        try:
            self._publish_camera_status("calibration_result")
        except Exception as exc:
            print(
                f"[calibration] saved but camera status refresh failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"[calibration] saved request={parameters.get('request_id')} "
            f"cameras={','.join(camera_ids)}",
            flush=True,
        )

    def _on_config(self, sample: Any) -> None:
        config_id = "unknown"
        try:
            message = decode_json(sample.payload)
            if message.get("kind") != "edge_config":
                raise ValueError("unexpected message kind")
            if message.get("edge_id") != self.edge_id:
                raise ValueError("config edge_id does not match")
            config_id = str(message["config_id"])
            data = message.get("data")
            if not isinstance(data, dict):
                raise ValueError("config data must be a JSON object")

            topic_root = str(data["topic_root"])
            EdgeTopics(self.edge_id, topic_root)
            endpoint = str(data["zenoh_endpoint"])
            if not endpoint:
                raise ValueError("zenoh_endpoint must not be empty")

            current = load_edge_metadata(self.identity_path)
            current.update(
                {
                    "approved": bool(data.get("approved")),
                    "display_name": str(data.get("display_name") or self.edge_id),
                    "zenoh_endpoint": endpoint,
                    "topic_root": topic_root.strip("/"),
                    "topics": data.get("topics") or {},
                    "config_id": config_id,
                    "configured_at": time.time(),
                }
            )
            save_edge_metadata(self.identity_path, current)
            self._publish_ack(
                config_id,
                "apply_config",
                True,
                {
                    "message": "configuration saved",
                    "restart_required": (
                        endpoint != self.endpoint or topic_root != self.topics.root
                    ),
                },
                kind="config_ack",
            )
            print(f"[config] saved config_id={config_id}", flush=True)
        except Exception as exc:
            self._publish_ack(
                config_id,
                "apply_config",
                False,
                error=str(exc),
                kind="config_ack",
            )
            print(f"[config] failed: {exc}", file=sys.stderr, flush=True)

    def run(self, once: bool = False) -> None:
        config = make_zenoh_config(self.endpoint, self.config_path)
        with zenoh.open(config) as session:
            self._session = session
            command_subscriber = session.declare_subscriber(
                self.topics.command,
                self._on_command,
            )
            config_subscriber = session.declare_subscriber(
                self.topics.config,
                self._on_config,
            )
            print(
                f"Edge agent started: edge_id={self.edge_id} "
                f"status={self.topics.status} command={self.topics.command}",
                flush=True,
            )
            try:
                self._publish_camera_status("connected")
            except Exception as exc:
                print(
                    f"[cameras] initial sync failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                while not self.stop_event.is_set():
                    session.put(self.topics.status, encode_json(self._status()))
                    if once:
                        time.sleep(0.2)
                        break
                    self.stop_event.wait(self.heartbeat_interval)
            finally:
                _ = command_subscriber, config_subscriber
                self._session = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Zenoh edge agent")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    for name in ("init", "show-id"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--identity-file",
            default=str(DEFAULT_IDENTITY_PATH),
        )
        if name == "init":
            command.add_argument("--edge-id")

    run = subparsers.add_parser("run")
    run.add_argument("--edge-id")
    run.add_argument("--identity-file", default=str(DEFAULT_IDENTITY_PATH))
    run.add_argument("--endpoint", default=os.getenv("ZENOH_ENDPOINT"))
    run.add_argument("--zenoh-config")
    run.add_argument("--topic-root", default=os.getenv("EDGE_TOPIC_ROOT"))
    run.add_argument("--heartbeat", type=float, default=5.0)
    run.add_argument("--camera-count", type=int, default=0)
    run.add_argument(
        "--camera-config",
        default=str(Path(__file__).parent / "config" / "cameras.local.yaml"),
    )
    run.add_argument(
        "--calibration-checkpoint",
        default=os.getenv("VGGT_OMEGA_CHECKPOINT"),
    )
    run.add_argument("--calibration-device", default="cuda")
    run.add_argument("--calibration-image-size", type=int, default=512)
    run.add_argument(
        "--camera-ping-timeout",
        type=float,
        default=DEFAULT_CAMERA_PING_TIMEOUT,
    )
    run.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.subcommand == "init":
        edge_id = load_or_create_edge_id(args.identity_file, args.edge_id)
        print(edge_id)
        return
    if args.subcommand == "show-id":
        path = Path(args.identity_file)
        if not path.exists():
            raise SystemExit(f"identity file does not exist: {path}")
        print(load_edge_metadata(path)["edge_id"])
        return

    if args.heartbeat <= 0:
        raise SystemExit("--heartbeat must be greater than zero")
    if args.camera_count < 0:
        raise SystemExit("--camera-count must not be negative")
    if args.calibration_image_size <= 0 or args.calibration_image_size % 16:
        raise SystemExit("--calibration-image-size must be a positive multiple of 16")
    if args.camera_ping_timeout <= 0:
        raise SystemExit("--camera-ping-timeout must be greater than zero")

    edge_id = load_or_create_edge_id(args.identity_file, args.edge_id)
    metadata = load_edge_metadata(args.identity_file)
    endpoint = args.endpoint or metadata.get("zenoh_endpoint")
    topic_root = args.topic_root or metadata.get("topic_root") or DEFAULT_TOPIC_ROOT
    agent = EdgeAgent(
        edge_id=edge_id,
        endpoint=endpoint,
        config_path=args.zenoh_config,
        topic_root=topic_root,
        heartbeat_interval=args.heartbeat,
        camera_count=args.camera_count,
        identity_path=args.identity_file,
        camera_config=args.camera_config,
        calibration_checkpoint=args.calibration_checkpoint,
        calibration_device=args.calibration_device,
        calibration_image_size=args.calibration_image_size,
        camera_ping_timeout=args.camera_ping_timeout,
    )

    def stop_agent(*_: Any) -> None:
        agent.stop_event.set()

    signal.signal(signal.SIGINT, stop_agent)
    signal.signal(signal.SIGTERM, stop_agent)
    agent.run(args.once)


if __name__ == "__main__":
    main()
