import json
import queue
import threading

import zenoh

TOPIC = "edge/topic1"
_STOP = object()


def make_zenoh_config(endpoint=None, config_path=None):
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


class ZenohSender:
    """추론 스레드와 겹쳐 실행되는 최신 결과 우선 Zenoh 송신기."""

    def __init__(
        self,
        topic=TOPIC,
        endpoint=None,
        config_path=None,
        queue_size=2,
    ):
        self.topic = topic
        self.endpoint = endpoint
        self.config_path = config_path
        self.queue = queue.Queue(maxsize=queue_size)
        self.closed = False
        self.dropped = 0
        self.router_ids = ()
        self.link_count = 0
        self._ready = threading.Event()
        self._error = None
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

        if not self._ready.wait(timeout=10):
            raise TimeoutError("Zenoh sender initialization timed out")
        if self._error:
            raise RuntimeError("Failed to initialize Zenoh sender") from self._error

    def send(self, payload: bytes):
        if self.closed:
            raise RuntimeError("ZenohSender is closed")
        if self._error:
            raise RuntimeError("Zenoh sender thread failed") from self._error

        # 송신이 밀리면 오래된 추론 결과를 버리고 최신 결과를 유지한다.
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
        self.queue.put_nowait(payload)

    def _config(self):
        return make_zenoh_config(self.endpoint, self.config_path)

    def _run(self):
        try:
            with zenoh.open(self._config()) as session:
                publisher = session.declare_publisher(self.topic)
                self.router_ids = tuple(
                    str(router_id) for router_id in session.info.routers_zid()
                )
                self.link_count = len(session.info.links())
                print(f"[Zenoh] Publishing topic: {self.topic}", flush=True)
                self._ready.set()
                while True:
                    payload = self.queue.get()
                    if payload is _STOP:
                        break
                    publisher.put(payload)
        except Exception as exc:
            self._error = exc
            self._ready.set()

    @property
    def router_connected(self):
        """Whether the opened Zenoh session currently has a network link."""
        return self.link_count > 0

    def close(self):
        if self.closed:
            return
        self.closed = True

        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put_nowait(_STOP)
        self.worker.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# 사용 예:
# with ZenohSender(topic="edge/topic1", endpoint="127.0.0.1:7447") as sender:
#     while True:
#         payload = run_inference()  # bytes
#         sender.send(payload)
