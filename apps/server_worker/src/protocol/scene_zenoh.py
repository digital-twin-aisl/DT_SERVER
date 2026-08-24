import json
import logging
import queue
import threading
from typing import Any

import zenoh


logger = logging.getLogger(__name__)
DEFAULT_SCENE_TOPIC = "meta-sejong/scene/v1"
_STOP = object()


def encode_scene(scene: dict[str, Any]) -> bytes:
    """Serialize a scene using the JSON contract consumed by Isaac Sim."""
    return json.dumps(
        scene,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


class ZenohScenePublisher:
    """Publish the newest scene without blocking the inference loop."""

    def __init__(
        self,
        topic: str = DEFAULT_SCENE_TOPIC,
        endpoint: str | None = None,
        config_path: str | None = None,
        queue_size: int = 2,
    ) -> None:
        if not topic:
            raise ValueError("topic must not be empty")
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")

        self.topic = topic
        self.endpoint = endpoint
        self.config_path = config_path
        self.queue = queue.Queue[bytes | object](maxsize=queue_size)
        self.closed = False
        self.dropped = 0
        self.sent_messages = 0
        self.sent_bytes = 0
        self.router_ids: tuple[str, ...] = ()
        self.link_count = 0
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.worker = threading.Thread(
            target=self._run,
            name="zenoh-scene-publisher",
            daemon=True,
        )
        self.worker.start()

        if not self._ready.wait(timeout=10):
            self.close()
            raise TimeoutError("Zenoh scene publisher initialization timed out")
        if self._error is not None:
            self.close()
            raise RuntimeError(
                "Zenoh scene publisher initialization failed"
            ) from self._error

    def _config(self) -> zenoh.Config:
        if self.config_path:
            return zenoh.Config.from_file(self.config_path)
        if self.endpoint:
            endpoint = self.endpoint
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

    def _run(self) -> None:
        try:
            with zenoh.open(self._config()) as session:
                publisher = session.declare_publisher(self.topic)
                self.router_ids = tuple(
                    str(router_id) for router_id in session.info.routers_zid()
                )
                self.link_count = len(session.info.links())
                logger.info(
                    "Zenoh scene publisher is ready: topic=%s endpoint=%s",
                    self.topic,
                    self.endpoint,
                )
                self._ready.set()
                while True:
                    payload = self.queue.get()
                    if payload is _STOP:
                        break
                    publisher.put(payload)
                    self.sent_messages += 1
                    self.sent_bytes += len(payload)
        except Exception as exc:
            self._error = exc
            logger.exception("Zenoh scene publisher worker failed")
            self._ready.set()

    def send_scene(self, scene: dict[str, Any]) -> bool:
        return self.send_payload(encode_scene(scene))

    def send_payload(self, payload: bytes) -> bool:
        """Queue an already encoded scene and avoid duplicate serialization."""
        if self.closed:
            return False
        if self._error is not None:
            logger.error(
                "Cannot publish scene because the Zenoh worker failed: %s",
                self._error,
            )
            return False

        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(payload)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self.worker.is_alive() and threading.current_thread() is not self.worker:
            self.worker.join(timeout=5)

    def __enter__(self) -> "ZenohScenePublisher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
